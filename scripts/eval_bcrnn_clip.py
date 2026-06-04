#!/usr/bin/env python
"""Clipping eval for a BC-RNN checkpoint — same shield, same Task-5 protocol.

BC-RNN is a stateful robomimic RolloutPolicy (manages its own LSTM hidden state),
so it can't reuse the residual run_rollout. But the *shield* is policy-agnostic —
it clips any 7-dim action to the demo-derived per-dim envelope (+ NaN guard). So
we wrap BC-RNN's raw action with the shield each step and measure how often it
fires, exactly as we report for the residuals. BC-RNN emits a *full* action (not a
residual), so this answers: does a converged BC-RNN ever leave the demo envelope?

Protocol: 30 rollouts, seed 42 (same cube starts), shield in the loop (env steps
the shielded action). Reports success, shield clip rate, and latency.

Usage:  python scripts/eval_bcrnn_clip.py [CKPT_PATH]
Saves:  out/bcrnn_clip_eval.json
"""
import os, sys, json, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from residual_lift.config import OUT_DIR
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import make_env, OBS_KEYS

DEFAULT_CKPT = os.path.join(
    OUT_DIR, "bc_rnn_converged", "bc_rnn_lift_ph_converged",
    "20260604160824", "models", "model_epoch_200.pth")
N_ROLL, SEED, HORIZON = 30, 42, 400


def obs_for_policy(obs_dict):
    d = dict(obs_dict)
    if "object" not in d and "object-state" in d:
        d["object"] = d["object-state"]
    return {k: np.asarray(d[k], dtype=np.float32) for k in OBS_KEYS}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    if not os.path.exists(ckpt):
        sys.exit(f"Missing checkpoint: {ckpt}")
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"Device: {device}\nCheckpoint: {ckpt}")

    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
    shield = build_shield(verbose=True)

    env = make_env()
    np.random.seed(SEED); torch.manual_seed(SEED)
    succ, steps, lat_p99 = [], [], []
    n_clip, n_step = 0, 0
    for _ in range(N_ROLL):
        policy.start_episode()
        obs = env.reset()
        prev, ok, lats = None, False, []
        for t in range(HORIZON):
            t0 = time.perf_counter()
            raw = np.asarray(policy(ob=obs_for_policy(obs)), dtype=np.float32)
            lats.append((time.perf_counter() - t0) * 1000)
            shielded = shield(raw, prev)
            n_step += 1
            if not np.allclose(shielded, raw):
                n_clip += 1
            prev = shielded
            obs, _, done, _ = env.step(shielded)
            if env._check_success():
                ok = True; break
            if done:
                break
        succ.append(ok); steps.append(t + 1); lat_p99.append(np.percentile(lats, 99))

    succ = np.array(succ)
    clip_rate = float(n_clip / n_step) if n_step else 0.0
    ok_steps = np.array(steps)[succ]
    out = {
        "checkpoint": os.path.relpath(ckpt, REPO),
        "n_rollouts": N_ROLL, "seed": SEED, "horizon": HORIZON,
        "success_rate": float(succ.mean()),
        "n_success": int(succ.sum()),
        "mean_steps_on_success": float(ok_steps.mean()) if ok_steps.size else 0.0,
        "p99_latency_ms": float(np.mean(lat_p99)),
        "shield_clip_rate": clip_rate,
        "n_clipped_steps": n_clip, "n_total_steps": n_step,
    }
    print("\n" + "=" * 50)
    print(f"BC-RNN (converged, epoch 200)")
    print(f"  success_rate      {out['success_rate']:.3f}  ({out['n_success']}/{N_ROLL})")
    print(f"  shield_clip_rate  {out['shield_clip_rate']:.4f}  "
          f"({n_clip}/{n_step} steps)")
    print(f"  mean_steps (succ) {out['mean_steps_on_success']:.1f}")
    print(f"  p99_latency_ms    {out['p99_latency_ms']:.3f}")
    jpath = os.path.join(OUT_DIR, "bcrnn_clip_eval.json")
    json.dump(out, open(jpath, "w"), indent=2)
    print(f"\nSaved: {jpath}")


if __name__ == "__main__":
    main()
