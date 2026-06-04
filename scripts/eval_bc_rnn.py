#!/usr/bin/env python
"""Evaluate the trained BC-RNN reference checkpoints in OUR robosuite harness.

robomimic's in-training rollout needs the legacy mujoco_py (absent), so we
disabled it and instead load each saved checkpoint with robomimic's
FileUtils.policy_from_checkpoint (a RolloutPolicy that manages RNN state) and
roll it out in the same robosuite Lift env our residual eval uses — 30 rollouts,
seed 42, identical protocol. Reports per-checkpoint success and the
best-over-training number (the paper's selection protocol), for an apples-to-
apples comparison against our BC (0.867) and residual (IQL 0.922).

Saves out/bc_rnn_eval.json.  Run:  python scripts/eval_bc_rnn.py
"""
import os, sys, glob, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from residual_lift.section4_eval import make_env, OBS_KEYS  # raw robosuite env

CKPT_DIR = os.path.join(REPO, "out", "bc_rnn", "bc_rnn_lift_ph")
N_ROLL, SEED, HORIZON = 30, 42, 400


def obs_for_policy(obs_dict):
    """Build the obs dict the BC-RNN policy expects (alias object-state->object)."""
    d = dict(obs_dict)
    if "object" not in d and "object-state" in d:
        d["object"] = d["object-state"]
    return {k: np.asarray(d[k], dtype=np.float32) for k in OBS_KEYS}


def eval_ckpt(ckpt_path, device):
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)
    env = make_env()
    np.random.seed(SEED); torch.manual_seed(SEED)
    successes = []
    for _ in range(N_ROLL):
        policy.start_episode()
        obs = env.reset()
        success = False
        for _t in range(HORIZON):
            act = policy(ob=obs_for_policy(obs))
            obs, _, done, _ = env.step(act)
            if env._check_success():
                success = True; break
            if done:
                break
        successes.append(success)
    env.close() if hasattr(env, "close") else None
    return float(np.mean(successes))


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "**", "*.pth"), recursive=True))
    if not ckpts:
        sys.exit(f"No checkpoints under {CKPT_DIR}. Train first (scripts/build_bc_rnn_config.py + train).")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"found {len(ckpts)} checkpoints")
    results = {}
    for c in ckpts:
        sr = eval_ckpt(c, device)
        results[os.path.relpath(c, CKPT_DIR)] = sr
        print(f"{os.path.basename(c):30s} success {sr:.3f}")
    best = max(results.values())
    final_key = sorted(results)[-1]
    payload = {"n_rollouts": N_ROLL, "seed": SEED, "horizon": HORIZON,
               "per_checkpoint": results,
               "best_over_training": best,
               "final_checkpoint_success": results[final_key],
               "bc_baseline": 0.867, "residual_iql": 0.922}
    out = os.path.join(REPO, "out", "bc_rnn_eval.json")
    json.dump(payload, open(out, "w"), indent=2)
    print(f"\nBest-over-training: {best:.3f}   final: {results[final_key]:.3f}")
    print(f"(BC 0.867 · residual IQL 0.922)\nsaved {out}")


if __name__ == "__main__":
    main()
