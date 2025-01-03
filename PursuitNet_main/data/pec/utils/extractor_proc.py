import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path
from sklearn.preprocessing import StandardScaler

@lru_cache(300000)
def _read_csv(path):
    return pd.read_csv(path)

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

class PECDataExtractor:
    def __init__(self, args):
        self.align_image_with_target_x = args.align_image_with_target_x

    def get_displ(self, data):
        """
        Get x and y displacements (proportional to discrete velocities) for
        a given trajectory and update the valid flag for observed timesteps.

        Args:
            data: Trajectories of all agents.

        Returns:
            Displacements of all agents.
        """
        num_agents, num_timesteps, num_features = data.shape
        res = np.zeros((num_agents, num_timesteps - 1, num_features))

        for i in range(num_agents):
            diff = data[i, 1:, :2] - data[i, :-1, :2]

            valid = np.convolve(data[i, :, 2], np.ones(2), "valid")
            valid = np.select(
                [valid == 2, valid == 1, valid == 0], [1, 1, 1], default=0
            )

            res[i, :, :2] = diff
            res[i, :, 2] = valid

            res[i, res[i, :, 2] == 0] = 0

        return np.float32(res), data[:, -1, :2]

    def extract_data(self, filename):
        """Load csv and extract the features required for PursuitNet.

        Args:
            filename: Filename of the csv to load.

        Returns:
            Feature dictionary required for PursuitNet.
        """
        df = pd.read_csv(filename)
        argo_id = int(Path(filename).stem)
        city = df["SOURCE"].values[0]
        agt_ts = np.sort(np.unique(df["TIMESTAMP"].values))
        mapping = {ts: i for i, ts in enumerate(agt_ts)}

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1),
            df.SPEED.to_numpy().reshape(-1, 1),
            df.ACCELERATION.to_numpy().reshape(-1, 1),
            np.zeros((df.shape[0], 1))  # Valid flag
        ), axis=1)

        steps = np.asarray([mapping[x] for x in df["TIMESTAMP"].values], np.int64)

        objs = df.groupby(["TRACK_ID", "OBJECT_TYPE"]).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agnt_key = keys.pop(obj_type.index("MICE"))
        av_key = keys.pop(obj_type.index("BAIT"))
        keys = [agnt_key, av_key] + keys

        res_trajs = []
        sp_trajs = []
        acc_trajs = []

        for key in keys:
            idcs = objs[key]
            tt = trajs[idcs]
            ts = steps[idcs]

            rt = np.zeros((20, 3))  # 2 for position, 1 for valid flag
            sp = np.zeros((20, 1))  # Speed
            acc = np.zeros((20, 1))  # Acceleration

            if 7 not in ts:
                continue

            rt[ts, :2] = tt[:, :2]  # Position features
            rt[ts, 2] = 1.0  # Valid flag
            sp[ts] = tt[:, 2].reshape(-1, 1)  # Speed (reshaping to (20, 1))
            acc[ts] = tt[:, 3].reshape(-1, 1)  # Acceleration (reshaping to (20, 1))

            res_trajs.append(rt)
            sp_trajs.append(sp)
            acc_trajs.append(acc)

        res_trajs = np.asarray(res_trajs, np.float32)
        sp_trajs = np.asarray(sp_trajs, np.float32)
        acc_trajs = np.asarray(acc_trajs, np.float32)

        assert res_trajs.ndim == 3 and sp_trajs.ndim == 3 and acc_trajs.ndim == 3
        
        sp_trajs = normalize_data(sp_trajs)
        acc_trajs = normalize_data(acc_trajs)
        res_gt = res_trajs[:, 8:].copy()
        sp_gt = sp_trajs[:, 8:].copy()
        acc_gt = acc_trajs[:, 8:].copy()

        origin = res_trajs[0, 7, :2].copy()
        rotation = np.eye(2, dtype=np.float32)
        theta = 0
        if self.align_image_with_target_x:
            pre = res_trajs[0, 7, :2] - res_trajs[0, 6, :2]
            theta = np.arctan2(pre[1], pre[0])
            rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]], np.float32)

        res_trajs[:, :, :2] = np.dot(res_trajs[:, :, :2] - origin, rotation)
        res_trajs[np.where(res_trajs[:, :, 2] == 0)] = 0

        res_fut_trajs = res_trajs[:, 8:].copy()
        sp_fut_trajs = sp_trajs[:, 8:].copy()
        acc_fut_trajs = acc_trajs[:, 8:].copy()

        res_trajs = res_trajs[:, :8].copy()
        sp_trajs = sp_trajs[:, :8].copy()
        acc_trajs = acc_trajs[:, :8].copy()

        sample = dict()
        sample["argo_id"] = argo_id
        sample["city"] = city
        sample["past_trajs"] = res_trajs
        sample["speed"] = sp_trajs
        #print("speed", sp_trajs.shape)
        sample["acceleration"] = acc_trajs
        sample["fut_trajs"] = res_fut_trajs
        sample["sp_fut_trajs"] = sp_fut_trajs
        sample["acc_fut_trajs"] = acc_fut_trajs
        sample["gt"] = res_gt[:, :, :2]
        sample["sp_gt"] = sp_gt
        sample["acc_gt"] = acc_gt
        sample["displ"], sample["centers"] = self.get_displ(sample["past_trajs"])
        sample["origin"] = origin
        sample["rotation"] = np.linalg.inv(rotation)

        return sample
