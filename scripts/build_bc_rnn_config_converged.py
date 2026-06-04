#!/usr/bin/env python
"""BC-RNN config for the CONVERGED run (1000 epochs) — companion to the
undertrained epoch-100 reference.

Identical paper-faithful hyperparameters to build_bc_rnn_config.py (LSTM
hidden_dim=400, GMM head, seq_length=10, lr=1e-4, seed 42), with two changes:
  - num_epochs 600 -> 1000 (Lift-PH saturates well before the paper's 2000;
    1000 is plenty to reach the published ~100% and makes the "harness
    reproduces the paper" claim airtight).
  - output_dir -> out/bc_rnn_converged, so the new run's checkpoints live
    SEPARATELY from the existing undertrained epoch-100 reference.
Run:  python scripts/build_bc_rnn_config_converged.py
"""
import os
from robomimic.config import config_factory

REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "data", "lift", "ph", "low_dim_v141.hdf5")
OUTDIR  = os.path.join(REPO, "out", "bc_rnn_converged")
os.makedirs(OUTDIR, exist_ok=True)

config = config_factory("bc")
with config.values_unlocked():
    # --- paper BC-RNN algo hyperparameters (lift/ph/low_dim) ---
    config.algo.rnn.enabled = True
    config.algo.rnn.horizon = 10
    config.algo.rnn.hidden_dim = 400
    config.algo.gmm.enabled = True
    config.algo.actor_layer_dims = ()
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
    config.train.num_epochs = 1000                    # converged (paper uses 2000; Lift saturates earlier)

    # --- experiment ---
    config.experiment.name = "bc_rnn_lift_ph_converged"
    config.experiment.validate = True
    config.experiment.epoch_every_n_steps = 100       # gradient steps / epoch
    config.experiment.rollout.enabled = False         # eval in our robosuite harness instead
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 100       # 10 checkpoints -> best-over-training

cfg_path = os.path.join(OUTDIR, "bc_rnn_lift_ph_converged_config.json")
with open(cfg_path, "w") as f:
    f.write(config.dump())
print("dataset exists:", os.path.exists(DATASET))
print("rnn:", config.algo.rnn.enabled, "| hidden:", config.algo.rnn.hidden_dim,
      "| gmm:", config.algo.gmm.enabled, "| seq_len:", config.train.seq_length,
      "| epochs:", config.train.num_epochs)
print("wrote", cfg_path)
