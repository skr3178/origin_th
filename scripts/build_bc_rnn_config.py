#!/usr/bin/env python
"""Build a paper-faithful BC-RNN config for Lift-PH low-dim and dump it to JSON.

Uses the installed (site-packages) robomimic config_factory and sets the BC-RNN
hyperparameters exactly as Mandlekar et al. 2021 / robomimic's
generate_paper_configs prescribes for lift/ph/low_dim:
  rnn.enabled=True, rnn.horizon=10, rnn.hidden_dim=400, gmm.enabled=True,
  actor_layer_dims=(), lr=1e-4, seq_length=10, batch_size=100.

(We set these by hand rather than calling the reference checkout's helper, to
avoid its newer transformers/huggingface import chain — the values are identical.)
We override: dataset path, output dir, epoch budget (capped — Lift saturates
early), rollout count (30, matching our eval), and the train/valid filter keys.
Run:  python scripts/build_bc_rnn_config.py
"""
import os
from robomimic.config import config_factory

REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "data", "lift", "ph", "low_dim_v141.hdf5")
OUTDIR  = os.path.join(REPO, "out", "bc_rnn")
os.makedirs(OUTDIR, exist_ok=True)

config = config_factory("bc")
with config.values_unlocked():
    # --- paper BC-RNN algo hyperparameters (lift/ph/low_dim) ---
    config.algo.rnn.enabled = True
    config.algo.rnn.horizon = 10
    config.algo.rnn.hidden_dim = 400
    config.algo.gmm.enabled = True
    config.algo.actor_layer_dims = ()                 # no MLP between RNN and output
    config.algo.optim_params.policy.learning_rate.initial = 1e-4

    # --- observation modalities (same 19-dim low-dim obs our BC uses) ---
    config.observation.modalities.obs.low_dim = [
        "object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    ]
    config.observation.modalities.obs.rgb = []

    # --- data / io ---
    config.train.data = DATASET
    config.train.output_dir = OUTDIR
    config.train.hdf5_cache_mode = "all"
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.batch_size = 100
    config.train.seq_length = 10
    config.train.seed = 42
    config.train.num_epochs = 600                     # let BC-RNN's GMM/RNN converge

    # --- experiment ---
    # Rollout DISABLED during training: robomimic's in-training rollout builds
    # EnvRobosuite, which hard-imports the legacy `mujoco_py` (absent here, and
    # unneeded with robosuite 1.4). Instead we save checkpoints and evaluate them
    # in our own robosuite eval harness (scripts/eval_bc_rnn.py), which drives
    # robosuite directly. Saving every 20 epochs lets us report best-over-training
    # (the paper's protocol) plus the final checkpoint.
    config.experiment.name = "bc_rnn_lift_ph"
    config.experiment.validate = True
    config.experiment.epoch_every_n_steps = 100       # gradient steps / epoch
    config.experiment.rollout.enabled = False
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 100

cfg_path = os.path.join(OUTDIR, "bc_rnn_lift_ph_config.json")
with open(cfg_path, "w") as f:
    f.write(config.dump())
print("dataset exists:", os.path.exists(DATASET))
print("rnn:", config.algo.rnn.enabled, "| hidden:", config.algo.rnn.hidden_dim,
      "| gmm:", config.algo.gmm.enabled, "| seq_len:", config.train.seq_length,
      "| epochs:", config.train.num_epochs)
print("wrote", cfg_path)
