import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
from torch.nn.utils import weight_norm
import matplotlib

matplotlib.use('Agg')
from argparse import Namespace

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(file_path))


class PursuitNet(pl.LightningModule):
    def __init__(self, args: Namespace):
        super(PursuitNet, self).__init__()
        self.args = args
        self.save_hyperparameters()

        self.encoder_transformer = EncoderTransformer(self.args)
        self.agent_gnn = AgentGnn(self.args)
        self.multihead_self_attention = MultiheadSelfAttention(self.args)
        self.decoder_residual = DecoderResidual(self.args)
        # Initialize the TCN module
        self.tcn = TCN(
            n_inputs=2,  
            n_outputs=128, 
            kernel_size=2,
            stride=1,
            dilation=2,
            padding=2,  
            dropout=0.2
)
        # Initialize Temporal Attention module
        self.temporal_attention = TemporalAttention(input_size=8)
        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.is_frozen = False

    @staticmethod
    def init_args(parent_parser):
        parser_dataset = parent_parser.add_argument_group("/storage1/wqq/dataset")
        parser_dataset.add_argument(
            "--train_split", type=str, default=os.path.join(
                root_path, "/storage1/wqq/dataset", "pec", "train", "data"))
        parser_dataset.add_argument(
            "--val_split", type=str, default=os.path.join(
                root_path, "/storage1/wqq/dataset", "pec", "val", "data"))
        parser_dataset.add_argument(
            "--test_split", type=str, default=os.path.join(
                root_path, "/storage1/wqq/dataset", "pec", "test_obs", "data"))
        parser_dataset.add_argument(
            "--train_split_pre", type=str, default=os.path.join(
                root_path, "/storage1/wqq/dataset", "pec", "train_pre.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre", type=str, default=os.path.join(
                root_path, "/storage1/wqq/dataset", "pec", "val_pre.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre", type=str, default=os.path.join(
                root_path, "/storage1/wqq/dataset", "pec", "test_pre.pkl"))
        parser_dataset.add_argument(
            "--reduce_dataset_size", type=int, default=0)
        parser_dataset.add_argument(
            "--use_preprocessed", type=bool, default=True)
        parser_dataset.add_argument(
            "--align_image_with_target_x", type=bool, default=True)

        parser_training = parent_parser.add_argument_group("training")
        parser_training.add_argument("--num_epochs", type=int, default=72)
        parser_training.add_argument(
            "--lr_values", type=list, default=[1e-3, 1e-4, 1e-3, 1e-4])
        parser_training.add_argument(
            "--lr_step_epochs", type=list, default=[32, 36, 68])
        parser_training.add_argument("--wd", type=float, default=0.001)
        parser_training.add_argument("--batch_size", type=int, default=16)
        parser_training.add_argument("--val_batch_size", type=int, default=16)
        parser_training.add_argument("--workers", type=int, default=0)
        parser_training.add_argument("--val_workers", type=int, default=0)
        parser_training.add_argument("--gpus", type=int, default=1)

        parser_model = parent_parser.add_argument_group("model")
        parser_model.add_argument("--latent_size", type=int, default=128)
        parser_model.add_argument("--num_preds", type=int, default=12)
        parser_model.add_argument("--mod_steps", type=list, default=[1, 5])
        parser_model.add_argument("--mod_freeze_epoch", type=int, default=36)

        return parent_parser

    def forward(self, batch):
        if self.is_frozen:
            for module in self.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()

        displ, centers = batch["displ"], batch["centers"]
        rotation, origin = batch["rotation"], batch["origin"]
        sp, acc = batch["speed"], batch["acceleration"]
        agents_per_sample = [x.shape[0] for x in displ]
        displ_cat = torch.cat(displ, dim=0)
        centers_cat = torch.cat(centers, dim=0)
        sp_cat = torch.cat(sp, dim=0).permute(0, 2, 1)
        acc_cat = torch.cat(acc, dim=0).permute(0, 2, 1)
        tcn_input = torch.cat((sp_cat, acc_cat), dim=1)
        tcn_features = self.tcn(tcn_input)
        tcn_features, _ = self.temporal_attention(tcn_features)
        tcn_features = torch.mean(tcn_features, dim=2)
        # print(tcn_features.shape)

        out_encoder_transformer = self.encoder_transformer(displ_cat, agents_per_sample)
        out_agent_gnn = self.agent_gnn(out_encoder_transformer, centers_cat, agents_per_sample, tcn_features)
        out_self_attention = self.multihead_self_attention(out_agent_gnn, agents_per_sample)
        out_self_attention = torch.stack([x[0] for x in out_self_attention])
        out_linear = self.decoder_residual(out_self_attention, self.is_frozen)

        out = out_linear.view(len(displ), 1, -1, self.args.num_preds, 2)
        
        for i in range(len(out)):
            out[i] = torch.matmul(out[i], rotation[i]) + origin[i].view(1, 1, 1, -1)

        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.decoder_residual.unfreeze_layers()

        self.is_frozen = True

    def prediction_loss(self, preds, gts):

        num_mods = preds.shape[2]
        preds = torch.cat([x[0] for x in preds], 0)
        gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0)
        gt_target = torch.repeat_interleave(gt_target, num_mods, dim=0)
        loss_single = self.reg_loss(preds, gt_target)
        loss_single = torch.sum(torch.sum(loss_single, dim=2), dim=1)

        loss_single = torch.split(loss_single, num_mods)

        loss_single = torch.stack(list(loss_single), dim=0)
        min_loss_index = torch.argmin(loss_single, dim=1)
        min_loss_combined = [x[min_loss_index[i]]
                             for i, x in enumerate(loss_single)]

        loss_out = torch.sum(torch.stack(min_loss_combined))

        return loss_out

    def configure_optimizers(self):
        if self.current_epoch == self.args.mod_freeze_epoch:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), weight_decay=self.args.wd)

        return optimizer

    def on_train_epoch_start(self):
        # Trigger weight freeze and optimizer reinit on mod_freeze_epoch
        if self.current_epoch == self.args.mod_freeze_epoch:
            self.freeze()
            self.trainer.accelerator.setup_optimizers(self.trainer)

        optimizer = self.optimizers() 
        if isinstance(optimizer, list):
            optimizer = optimizer[0]  

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.get_lr(self.current_epoch)

    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch)
        loss = self.prediction_loss(out, train_batch["gt"])
        self.log("loss_train", loss / len(out))
        return loss

    def get_lr(self, epoch):
        lr_index = 0
        for lr_epoch in self.args.lr_step_epochs:
            if epoch < lr_epoch:
                break
            lr_index += 1
        return self.args.lr_values[lr_index]

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch)
        loss = self.prediction_loss(out, val_batch["gt"])
        self.log("loss_val", loss / len(out))

        # Extract target agent only
        pred = [x[0].detach().cpu().numpy() for x in out]
        gt = [x[0].detach().cpu().numpy() for x in val_batch["gt"]]
        return pred, gt

    def validation_epoch_end(self, validation_outputs):
        # Extract predictions
        pred = [out[0] for out in validation_outputs]
        pred = np.concatenate(pred, 0)
        gt = [out[1] for out in validation_outputs]
        gt = np.concatenate(gt, 0)
        ade1, fde1, ade, fde = self.calc_prediction_metrics(pred, gt)
        mae1, smape1, mae, smape = self.calc_mae(pred, gt)
        self.log("ade1_val", ade1, prog_bar=True)
        self.log("fde1_val", fde1, prog_bar=True)
        self.log("ade_val", ade, prog_bar=True)
        self.log("fde_val", fde, prog_bar=True)
        self.log("mae_val", mae, prog_bar=True)
        self.log("smape_val", smape, prog_bar=True)

    def calc_prediction_metrics(self, preds, gts):
        error_per_t = np.linalg.norm(preds - np.expand_dims(gts, axis=1), axis=-1)
        # print("error_per_t shape:", error_per_t.shape)
        # Calculate the error for the first mode (at index 0)
        fde_1 = np.average(error_per_t[:, 0, -1])
        ade_1 = np.average(error_per_t[:, 0, :])

        # Calculate the error for all modes
        # Best mode is always the one with the lowest final displacement
        lowest_final_error_indices = np.argmin(error_per_t[:, :, -1], axis=1)
        error_per_t = error_per_t[np.arange(
            preds.shape[0]), lowest_final_error_indices]
        fde = np.average(error_per_t[:, -1])
        ade = np.average(error_per_t[:, :])
        return ade_1, fde_1, ade, fde

    def calc_mae(self, preds, gts):
        # print("preds shape:", preds.shape)
        # print("gts shape:", gts.shape)

        error_per_t = np.abs(preds - gts[:, np.newaxis, :, :])[:, :, :, 0]
        # print("error_per_t shape:", error_per_t.shape)

        mae_1 = np.average(error_per_t[:, 0, :])

        abs_diff = np.abs(preds - gts[:, np.newaxis, :, :])
        sum_values = np.abs(preds) + np.abs(gts[:, np.newaxis, :, :])

        smape_1 = 200 * np.average(
            np.divide(abs_diff[:, 0, :, :], sum_values[:, 0, :, :], out=np.zeros_like(abs_diff[:, 0, :, :]),
                      where=sum_values[:, 0, :, :] != 0))

        lowest_final_error_indices = np.argmin(error_per_t[:, :, -1], axis=1)
        error_per_t_all_modes = error_per_t[np.arange(len(lowest_final_error_indices)), lowest_final_error_indices, :]
        mae = np.average(error_per_t_all_modes)

        abs_diff = np.abs(preds - gts[:, np.newaxis, :, :])
        sum_values = np.abs(preds) + np.abs(gts[:, np.newaxis, :, :])
        smape = 200 * np.average(np.divide(abs_diff, sum_values, out=np.zeros_like(abs_diff), where=sum_values != 0))

        return mae_1, smape_1, mae, smape


class EncoderTransformer(nn.Module):
    def __init__(self, args):
        super(EncoderTransformer, self).__init__()
        self.args = args

        self.input_size = 3
        self.hidden_size = args.latent_size
        self.num_layers = 2

        self.input_proj = nn.Linear(self.input_size, self.hidden_size)

        self.rope = RotaryPositionEmbedding(self.hidden_size)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=2)

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=self.num_layers)

    def forward(self, transformer_in, agents_per_sample):
        transformer_in = self.input_proj(transformer_in)

        transformer_in = self.rope(transformer_in)
        transformer_out = self.transformer_encoder(transformer_in)
        return transformer_out[:, -1, :]


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=40):
        super(RotaryPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('rotary_emb', self._get_rotary_embedding())

    def _get_rotary_embedding(self):
        # Create the sinusoidal rotary embeddings
        rotary_emb = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(40.0)) / self.d_model))
        rotary_emb[:, 0::2] = torch.sin(position * div_term)
        rotary_emb[:, 1::2] = torch.cos(position * div_term)
        return rotary_emb.unsqueeze(0)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        rotary_emb = self.rotary_emb[:, :seq_len, :].clone().detach()
        return rotary_emb * x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCN(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCN, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        if out.shape != res.shape:
            # Here you can either adjust or raise an error
            raise ValueError(f"Shape mismatch: out {out.shape} vs res {res.shape}")

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TCN(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalAttention(nn.Module):
    def __init__(self, input_size):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Calculate attention scores
        scores = torch.bmm(query, key.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attention_weights = self.softmax(scores)

        # Apply attention weights to values
        context = torch.bmm(attention_weights, value)
        return context, attention_weights


class AgentGnn(nn.Module):
    def __init__(self, args):
        super(AgentGnn, self).__init__()
        self.args = args
        self.latent_size = args.latent_size

        self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
        self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)

        # Additional linear layer to process TCN features
        self.tcn_fc = nn.Linear(128, self.latent_size)  # Adjust size based on TCN output

    def forward(self, gnn_in, centers, agents_per_sample, tcn_features=None):
        x = gnn_in

        if tcn_features is not None:
            tcn_features = self.tcn_fc(tcn_features)
            x = x + tcn_features  # Integrate TCN features with GNN input

        edge_index = self.build_fully_connected_edge_idx(agents_per_sample).to(x.device)
        edge_attr = self.build_edge_attr(edge_index, centers).to(x.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr))
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

        return gnn_out

    def build_fully_connected_edge_idx(self, agents_per_sample):
        edge_index = []

        # In the for loop one subgraph is built (no self edges!)
        # The subgraph gets offsetted and the full graph over all samples in the batch
        # gets appended with the offsetted subgrah
        offset = 0
        for i in range(len(agents_per_sample)):
            num_nodes = agents_per_sample[i]

            adj_matrix = torch.ones((num_nodes, num_nodes))
            adj_matrix = adj_matrix.fill_diagonal_(0)

            sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())
            edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)

            # Offset the list
            edge_index_subgraph = torch.Tensor(
                np.asarray(edge_index_subgraph) + offset)
            offset += agents_per_sample[i]

            edge_index.append(edge_index_subgraph)

        # Concat the single subgraphs into one
        edge_index = torch.LongTensor(np.column_stack(edge_index))
        return edge_index

    def build_edge_attr(self, edge_index, data):
        edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)

        rows, cols = edge_index
        # goal - origin
        edge_attr = data[cols] - data[rows]

        return edge_attr


class MultiheadSelfAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadSelfAttention, self).__init__()
        self.args = args

        self.latent_size = self.args.latent_size

        self.multihead_attention = nn.MultiheadAttention(self.latent_size, 2)

    def forward(self, att_in, agents_per_sample):
        att_out_batch = []

        # Upper path is faster for multiple samples in the batch and vice versa
        if len(agents_per_sample) > 1:
            max_agents = max(agents_per_sample)

            padded_att_in = torch.zeros(
                (len(agents_per_sample), max_agents, self.latent_size), device=att_in[0].device)
            mask = torch.arange(max_agents) < torch.tensor(
                agents_per_sample)[:, None]

            padded_att_in[mask] = att_in

            mask_inverted = ~mask
            mask_inverted = mask_inverted.to(att_in.device)

            padded_att_in_swapped = torch.swapaxes(padded_att_in, 0, 1)

            padded_att_in_swapped, _ = self.multihead_attention(
                padded_att_in_swapped, padded_att_in_swapped, padded_att_in_swapped, key_padding_mask=mask_inverted)

            padded_att_in_reswapped = torch.swapaxes(
                padded_att_in_swapped, 0, 1)

            att_out_batch = [x[0:agents_per_sample[i]]
                             for i, x in enumerate(padded_att_in_reswapped)]
        else:
            att_in = torch.split(att_in, agents_per_sample)
            for i, sample in enumerate(att_in):
                # Add the batch dimension (this has to be the second dimension, because attention requires it)
                att_in_formatted = sample.unsqueeze(1)
                att_out, weights = self.multihead_attention(
                    att_in_formatted, att_in_formatted, att_in_formatted)

                # Remove the "1" batch dimension
                att_out = att_out.squeeze()
                att_out_batch.append(att_out)

        return att_out_batch


class DecoderResidual(nn.Module):
    def __init__(self, args):
        super(DecoderResidual, self).__init__()

        self.args = args

        output = []
        for i in range(sum(args.mod_steps)):
            output.append(PredictionNet(args))
        self.output = nn.ModuleList(output)

    def forward(self, decoder_in, is_frozen):
        sample_wise_out = []

        if self.training is False:
            for out_subnet in self.output:
                sample_wise_out.append(out_subnet(decoder_in))
        elif is_frozen:
            for i in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
                sample_wise_out.append(self.output[i](decoder_in))
        else:
            sample_wise_out.append(self.output[0](decoder_in))

        decoder_out = torch.stack(sample_wise_out)
        decoder_out = torch.swapaxes(decoder_out, 0, 1)

        return decoder_out

    def unfreeze_layers(self):
        for layer in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
            for param in self.output[layer].parameters():
                param.requires_grad = True


class PredictionNet(nn.Module):
    def __init__(self, args):
        super(PredictionNet, self).__init__()

        self.args = args

        self.latent_size = args.latent_size

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.GroupNorm(1, self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.GroupNorm(1, self.latent_size)

        self.output_fc = nn.Linear(self.latent_size, args.num_preds * 2)
        self.regularization = args.wd

    def forward(self, prednet_in):
        # Residual layer
        x = self.weight1(prednet_in)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.weight2(x)
        x = self.norm2(x)
        x += prednet_in
        x = F.relu(x)

        prednet_out = self.output_fc(x)

        return prednet_out
