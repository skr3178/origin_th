#!/usr/bin/env python
"""Train the two algorithms whose checkpoints were never saved (AWAC, IQL) and
render a rollout video for each, plus a render-only clip for the BC-RNN reference.

Why this exists: the algo_comparison run used save=False, so AWAC/IQL weights were
discarded. We retrain each at seed 42 (deterministic, ~30s on GPU), save the
checkpoint, and render a rollout under the SAME protocol as rollout_{bc,residual}.mp4
(30 rollouts, eval seed 42 = identical cube starts, first success saved). BC-RNN is
render-only from its saved (undertrained) epoch-100 checkpoint — labeled as such.

Run:  python scripts/render_extra_rollouts.py
"""
import os, sys, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from residual_lift.config import DEVICE, OUT_DIR, BC_CKPT
from residual_lift.section1_bc import load_bc
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import evaluate, make_env, OBS_KEYS
from residual_lift.algorithms import train_residual_awac, train_residual_iql

SEED, N_ROLL = 42, 30


def train_render(name, fn, bc, shield):
    """Train one residual algorithm (seed 42), save its checkpoint, render a rollout."""
    print(f"\n=== {name}: train (seed {SEED}) + render ===")
    res, hist = fn(bc, steps=10000, delta_bound=0.005, seed=SEED, log_every=2000)
    ckpt = os.path.join(OUT_DIR, f"{name.lower()}.pt")
    torch.save(res.state_dict(), ckpt)
    video = os.path.join(OUT_DIR, f"rollout_{name.lower()}.mp4")
    r = evaluate(res, shield=shield, n_rollouts=N_ROLL, seed=SEED,
                 save_video=video, label=name)
    print(f"{name}: success {r['success_rate']:.3f}  clip_rate {r['shield_clip_rate']:.3f}  "
          f"delta_mag {hist['delta_mag'][-1]:.4f}")
    print(f"  saved {ckpt}\n  saved {video}")
    return {"success_rate": r["success_rate"], "shield_clip_rate": r["shield_clip_rate"],
            "delta_mag_final": float(hist["delta_mag"][-1]),
            "checkpoint": os.path.relpath(ckpt, REPO), "video": os.path.relpath(video, REPO)}


def render_bc_rnn():
    """Render-only from the saved (undertrained) BC-RNN epoch-100 checkpoint.

    BC-RNN uses robomimic's RolloutPolicy interface (manages LSTM hidden state via
    start_episode), so it can't reuse our run_rollout. We capture the first rollout
    (seed 42, deterministic) regardless of outcome — an honest clip of an
    undertrained model — and report whether it succeeded.
    """
    import glob
    import imageio
    import robomimic.utils.file_utils as FileUtils

    ckpts = glob.glob(os.path.join(OUT_DIR, "bc_rnn", "bc_rnn_lift_ph", "**",
                                   "model_epoch_100.pth"), recursive=True)
    if not ckpts:
        print("\n[BC-RNN] epoch-100 checkpoint not found — skipping render.")
        return None
    ckpt = sorted(ckpts)[0]
    print(f"\n=== BC-RNN: render-only (undertrained epoch-100) ===\n  {ckpt}")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=DEVICE, verbose=False)

    def obs_for_policy(od):
        d = dict(od)
        if "object" not in d and "object-state" in d:
            d["object"] = d["object-state"]
        return {k: np.asarray(d[k], dtype=np.float32) for k in OBS_KEYS}

    env = make_env()
    np.random.seed(SEED); torch.manual_seed(SEED)  # same start as the other seed-42 clips
    policy.start_episode()
    obs = env.reset()
    frames, success = [], False
    for _t in range(400):
        act = policy(ob=obs_for_policy(obs))
        obs, _, done, _ = env.step(act)
        frames.append(np.flipud(env.sim.render(height=256, width=256, camera_name="agentview")))
        if env._check_success():
            success = True; break
        if done:
            break
    video = os.path.join(OUT_DIR, "rollout_bc_rnn_undertrained.mp4")
    imageio.mimsave(video, frames, fps=20)
    print(f"  rollout #0 outcome: {'SUCCESS' if success else 'FAILURE (stuck/no lift)'}  "
          f"[{len(frames)} frames]")
    print(f"  saved {video}  (labeled undertrained — epoch 100 of a run that needs ~2000)")
    return {"first_rollout_success": success, "frames": len(frames),
            "checkpoint": os.path.relpath(ckpt, REPO), "video": os.path.relpath(video, REPO),
            "note": "undertrained epoch-100 reference; not a converged BC-RNN"}


def main():
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")
    bc = load_bc()
    shield = build_shield(verbose=False)

    out = {"seed": SEED, "n_rollouts": N_ROLL}
    out["AWAC"] = train_render("AWAC", train_residual_awac, bc, shield)
    out["IQL"]  = train_render("IQL", train_residual_iql, bc, shield)
    out["BC_RNN"] = render_bc_rnn()

    jpath = os.path.join(OUT_DIR, "extra_rollouts.json")
    json.dump(out, open(jpath, "w"), indent=2)
    print(f"\nsaved manifest {jpath}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
