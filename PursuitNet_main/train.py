import os
import argparse
import sys
import logging
import torch

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.pec.pec_csv_dataset import PECCSVDataset
from data.pec.utils.torch_utils import collate_fn_dict
from model.pursuitnet import PursuitNet

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

log_dir = os.path.dirname(os.path.abspath(__file__))
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser = PursuitNet.init_args(parser)


def main():
    args = parser.parse_args()

    dataset = PECCSVDataset(args.val_split, args.val_split_pre, args)
    val_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        collate_fn=collate_fn_dict,
        pin_memory=True
    )

    dataset = PECCSVDataset(args.train_split, args.train_split_pre, args)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=True,
        shuffle=True 
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{loss_train:.2f}-{loss_val:.2f}"
        "{ade1_val:.2f}-{fde1_val:.2f}-{ade_val:.2f}-{fde_val:.2f}"
        "{mae1_val:.2f}-{smape1_val:.2f}-{mae_val:.2f}-{smape_val:.2f}-"
        "{distance_accuracy:.2f}-{angle_accuracy:.2f}-{speed_accuracy:.2f}",
        monitor="loss_val",
        save_top_k=-1,
    )

    model = PursuitNet(args).to(device)
    
    trainer = pl.Trainer(  
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
        gpus=1,
        weights_save_path=None,
        max_epochs=args.num_epochs
    )

    trainer.fit(model, train_loader, val_loader)
    


if __name__ == "__main__":
    main()
