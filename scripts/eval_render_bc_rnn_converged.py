#!/usr/bin/env python
"""Evaluate the CONVERGED BC-RNN run (1000 epochs) in our robosuite harness and
render a rollout from the best checkpoint.

Mirrors scripts/eval_bc_rnn.py (30 rollouts, seed 42, identical protocol) but
points at out/bc_rnn_converged and additionally renders a rollout video from the
best-over-training checkpoint — the converged counterpart to the undertrained
out/rollout_bc_rnn_undertrained.mp4.

Run:  python scripts/eval_render_bc_rnn_converged.py
"""
import os, sys, glob, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import imageio

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from residual_lift.section4_eval import make_env, OBS_KEYS

CKPT_GLOB = os.path.join(REPO, "out", "bc_rnn_converged", "**", "models", "*.pth")
N_ROLL, SEED, HORIZON = 30, 42, 400


def obs_for_policy(obs_dict):
    d = dict(obs_dict)
    if "object" not in d and "object-state" in d:
        d["object"] = d["object-state"]
    return {k: np.asarray(d[k], dtype=np.float32) for k in OBS_KEYS}


def eval_ckpt(policy, env, render=False):
    """One pass of N_ROLL rollouts; optionally capture frames of the FIRST success."""
    np.random.seed(SEED); torch.manual_seed(SEED)
    successes, best_frames = [], None
    for _ in range(N_ROLL):
        policy.start_episode()
        obs = env.reset()
        success, frames = False, []
        for _t in range(HORIZON):
            act = policy(ob=obs_for_policy(obs))
            obs, _, done, _ = env.step(act)
            if render and best_frames is None:
                frames.append(np.flipud(env.sim.render(height=256, width=256, camera_name="agentview")))
            if env._check_success():
                success = True; break
            if done:
                break
        successes.append(success)
        if render and success and best_frames is None:
            best_frames = frames  # first successful rollout's frames
    return float(np.mean(successes)), best_frames


def main():
    ckpts = sorted(glob.glob(CKPT_GLOB, recursive=True),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    if not ckpts:
        sys.exit(f"No checkpoints under {CKPT_GLOB}. Train first "
                 f"(scripts/build_bc_rnn_config_converged.py + robomimic.scripts.train).")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    env = make_env()
    print(f"found {len(ckpts)} checkpoints")

    results = {}
    for c in ckpts:
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=c, device=device, verbose=False)
        sr, _ = eval_ckpt(policy, env, render=False)
        results[os.path.relpath(c, REPO)] = sr
        print(f"{os.path.basename(c):28s} success {sr:.3f}")

    best_path = max(results, key=results.get)
    best_sr = results[best_path]

    # Render a rollout from the best checkpoint.
    print(f"\nrendering best checkpoint ({best_sr:.3f}): {best_path}")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=os.path.join(REPO, best_path),
                                                 device=device, verbose=False)
    _, frames = eval_ckpt(policy, env, render=True)
    video = os.path.join(REPO, "out", "rollout_bc_rnn_converged.mp4")
    if frames:
        imageio.mimsave(video, frames, fps=20)
        print(f"saved {video}  ({len(frames)} frames)")
    else:
        video = None
        print("no successful rollout to render (unexpected for a converged Lift BC-RNN)")

    payload = {"n_rollouts": N_ROLL, "seed": SEED, "horizon": HORIZON,
               "num_epochs": 1000, "epoch_every_n_steps": 100,
               "per_checkpoint": results,
               "best_over_training": best_sr,
               "best_checkpoint": best_path,
               "video": os.path.relpath(video, REPO) if video else None,
               "bc_baseline": 0.867, "residual_iql": 0.922,
               "undertrained_epoch100": 0.73}
    out = os.path.join(REPO, "out", "bc_rnn_converged_eval.json")
    json.dump(payload, open(out, "w"), indent=2)
    print(f"\nBest-over-training: {best_sr:.3f}   (undertrained epoch-100 was 0.73)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
