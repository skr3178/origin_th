"""Section 0 (data) — dataset resolution, in-RAM dataset, normalization stats.

The dataset is loaded from a permanent local copy committed under 10x/data/.
It is downloaded (once) only if no local copy and no robomimic cache is found.
"""
import os
import glob
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import robomimic

from . import config  # noqa: F401  (ensures MUJOCO_GL is set early)
from .config import OBS_KEYS, OBS_DIM, DATA_DIR

_FNAME = "low_dim_v141.hdf5"


def resolve_dataset_path():
    """Locate the lift-ph low-dim HDF5.

    Resolution order: (1) repo-local copy under 10x/data, (2) robomimic's cached
    datasets dir, (3) download as a last resort (first run only — permanent after).
    """
    candidates = [
        os.path.join(DATA_DIR, "lift", "ph", _FNAME),
        os.path.join(os.path.dirname(robomimic.__file__), "..",
                     "datasets", "lift", "ph", _FNAME),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print("No local dataset found; downloading lift-ph (first run only)...")
        subprocess.run(
            ["python", "-m", "robomimic.scripts.download_datasets",
             "--tasks", "lift", "--dataset_types", "ph", "--hdf5_types", "low_dim"],
            check=True,
        )
        cache = os.path.join(os.path.dirname(robomimic.__file__), "..",
                             "datasets", "lift", "ph", "low_dim*.hdf5")
        path = sorted(glob.glob(cache))[0]
    return os.path.abspath(path)


DATASET_PATH = resolve_dataset_path()


def _flatten_obs_batch(obs_group):
    return np.concatenate([obs_group[k][:] for k in OBS_KEYS], axis=-1).astype(np.float32)


def read_mask(split, hdf5_path=DATASET_PATH):
    """Return the list of demo keys (e.g. ['demo_3', ...]) for a named split in
    the file's `mask/` group. Available: train, valid, 20_percent[_train/_valid],
    50_percent[_train/_valid]."""
    with h5py.File(hdf5_path, "r") as f:
        return [k.decode() if isinstance(k, bytes) else str(k) for k in f["mask"][split][:]]


class LiftPHDataset(Dataset):
    """Loads (a subset of) the dataset into RAM once. ~2 MB total; caching is free
    and ~100x faster than reopening the HDF5 file on every __getitem__.

    demo_keys: optional list of demo names (e.g. from read_mask) to restrict to a
    split. None loads every demo, in numeric order."""
    def __init__(self, hdf5_path=DATASET_PATH, demo_keys=None):
        with h5py.File(hdf5_path, "r") as f:
            if demo_keys is None:
                demo_keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
            o, a, r, no, d = [], [], [], [], []
            for k in demo_keys:
                demo = f["data"][k]
                o.append(_flatten_obs_batch(demo["obs"]))
                a.append(demo["actions"][:].astype(np.float32))
                r.append(demo["rewards"][:].astype(np.float32))
                no.append(_flatten_obs_batch(demo["next_obs"]))
                d.append(demo["dones"][:].astype(np.float32))
        self.obs      = torch.from_numpy(np.concatenate(o,  axis=0))
        self.actions  = torch.from_numpy(np.concatenate(a,  axis=0))
        self.rewards  = torch.from_numpy(np.concatenate(r,  axis=0))
        self.next_obs = torch.from_numpy(np.concatenate(no, axis=0))
        self.dones    = torch.from_numpy(np.concatenate(d,  axis=0))

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return {
            "obs":      self.obs[idx],
            "action":   self.actions[idx],
            "reward":   self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done":     self.dones[idx],
        }


def load_obs_stats(dataset=None):
    """Return (obs_mean, obs_std) over the full dataset. Used as fixed
    normalization buffers inside the policies."""
    full = dataset if dataset is not None else LiftPHDataset(DATASET_PATH)
    obs_mean = full.obs.mean(dim=0)
    obs_std  = full.obs.std(dim=0) + 1e-6
    assert obs_mean.shape[0] == OBS_DIM
    return obs_mean, obs_std


if __name__ == "__main__":
    _full = LiftPHDataset(DATASET_PATH)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Dataset: {len(_full)} transitions (cached in RAM)")
    print(f"Action range: [{_full.actions.min():.3f}, {_full.actions.max():.3f}]")
    print(f"Reward nonzero fraction: {(_full.rewards != 0).float().mean().item():.4f}")
