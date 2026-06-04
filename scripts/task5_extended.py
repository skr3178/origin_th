#!/usr/bin/env python
"""Task 5 (extended) — BC vs TD3+BC vs AWAC vs IQL, all shielded.

Standalone: loads four FROZEN checkpoints (no retraining) and runs the exact
Task-5 protocol — 30 rollouts, eval seed 42 (identical cube starts), shield in
the loop for every residual. Reuses make_env / flatten_obs_dict / build_shield
from the residual_lift package so the harness is bit-for-bit the same one Task 5
prints; this script only *widens the table* from 2 policies to 4 and adds a
paired (same-starts) comparison of each residual against BC.

Leaves section4_eval.py and plot_final_eval.py untouched.

Inputs (must already exist in out/):  bc.pt, residual.pt (=TD3+BC), awac.pt, iql.pt
Outputs:  out/task5_extended_comparison.png, out/task5_extended.json

Run:  python scripts/task5_extended.py
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import load_residual, RESIDUAL_CKPT
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import make_env, flatten_obs_dict

N, SEED, MAX_STEPS = 30, 42, 400


def eval_per_rollout(policy, shield=None):
    """Same harness as Task 5 (re-seed + rebuild env => same starts). Returns
    per-rollout success/steps/latency arrays plus the overall shield clip rate."""
    np.random.seed(SEED); torch.manual_seed(SEED)
    env = make_env()
    succ, steps, lat_p99 = [], [], []
    n_clip, n_step = 0, 0
    for _ in range(N):
        obs = flatten_obs_dict(env.reset())
        prev, ok, lats = None, False, []
        for t in range(MAX_STEPS):
            t0 = time.perf_counter()
            with torch.no_grad():
                a = policy.act(torch.from_numpy(obs).float().to(DEVICE)).cpu().numpy()
            lats.append((time.perf_counter() - t0) * 1000)
            if shield is not None:
                shielded = shield(a, prev)
                n_step += 1
                if not np.allclose(shielded, a):
                    n_clip += 1
                a = shielded
            prev = a
            od, _, done, _ = env.step(a)
            obs = flatten_obs_dict(od)
            if env._check_success():
                ok = True; break
            if done:
                break
        succ.append(ok); steps.append(t + 1); lat_p99.append(np.percentile(lats, 99))
    clip_rate = float(n_clip / n_step) if n_step else 0.0
    return np.array(succ), np.array(steps), np.array(lat_p99), clip_rate


def main():
    print(f"Device: {DEVICE}")
    bc = load_bc()
    shield = build_shield(verbose=False)

    # Four frozen policies. BC runs unshielded (baseline); residuals are shielded.
    awac_ckpt = os.path.join(OUT_DIR, "awac.pt")
    iql_ckpt  = os.path.join(OUT_DIR, "iql.pt")
    for p in (RESIDUAL_CKPT, awac_ckpt, iql_ckpt):
        if not os.path.exists(p):
            sys.exit(f"Missing checkpoint: {p}")

    policies = [
        ("BC",     bc,                                None),
        ("TD3+BC", load_residual(bc),                 shield),
        ("AWAC",   load_residual(bc, awac_ckpt),      shield),
        ("IQL",    load_residual(bc, iql_ckpt),       shield),
    ]

    results = {}
    for name, pol, sh in policies:
        print(f"Evaluating {name}...")
        s, steps, lat, clip = eval_per_rollout(pol, shield=sh)
        results[name] = {"succ": s, "steps": steps, "lat": lat, "clip": clip}

    bc_s = results["BC"]["succ"]

    # --- table ---
    print("\n" + "=" * 78)
    print(f"{'Policy':<10}{'success':>10}{'mean_steps':>12}{'p99_lat_ms':>12}"
          f"{'clip_rate':>11}{'flips_vs_BC':>14}")
    print("-" * 78)
    table = {}
    for name, _, _ in policies:
        r = results[name]
        s = r["succ"]
        ok_steps = r["steps"][s]
        mean_steps = float(ok_steps.mean()) if ok_steps.size else 0.0
        p99 = float(np.mean(r["lat"]))
        sr = float(s.mean())
        if name == "BC":
            flips = "—"
        else:
            fixed  = int((~bc_s & s).sum())   # residual succeeds where BC failed
            broke  = int((bc_s & ~s).sum())   # residual fails where BC succeeded
            flips = f"+{fixed}/-{broke}"
        clip = "—" if r["clip"] == 0.0 and name == "BC" else f"{r['clip']:.3f}"
        print(f"{name:<10}{sr:>10.3f}{mean_steps:>12.1f}{p99:>12.3f}{clip:>11}{flips:>14}")
        table[name] = {"success_rate": sr, "mean_steps_on_success": mean_steps,
                       "p99_latency_ms": p99, "shield_clip_rate": r["clip"],
                       "success_vector": s.astype(int).tolist()}
        if name != "BC":
            table[name]["flips_vs_bc"] = {"fixed": fixed, "broke": broke}

    # --- plot: 4 panels ---
    names = [n for n, _, _ in policies]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (a) success-rate bars
    a0 = ax[0, 0]
    srs = [results[n]["succ"].mean() for n in names]
    bars = a0.bar(names, srs, color=colors)
    for b, n in zip(bars, names):
        s = results[n]["succ"]
        a0.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{s.mean():.3f}\n({int(s.sum())}/{N})", ha="center", va="bottom", fontsize=9)
    a0.set_ylim(0, 1.08); a0.set_ylabel("success rate")
    a0.set_title("Success rate (30 rollouts, seed 42, shielded residuals)")

    # (b) per-rollout outcome matrix (rows = policies, same start per column)
    a1 = ax[0, 1]
    grid = np.vstack([results[n]["succ"].astype(int) for n in names])
    a1.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    a1.set_yticks(range(len(names))); a1.set_yticklabels(names)
    a1.set_xlabel("rollout index (same start down each column)")
    a1.set_title("Per-rollout outcome — green=success, red=fail")

    # (c) shield clip rate (BC has none)
    a2 = ax[1, 0]
    clip_names = names[1:]
    clip_vals = [results[n]["clip"] for n in clip_names]
    cbars = a2.bar(clip_names, clip_vals, color=colors[1:])
    for b, v in zip(cbars, clip_vals):
        a2.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("shield clip rate"); a2.set_title("How often the shield fires (BC: n/a)")

    # (d) per-rollout p99 latency
    a3 = ax[1, 1]
    a3.boxplot([results[n]["lat"] for n in names], labels=names)
    a3.axhline(50, ls="--", c="gray", lw=0.8, label="20 Hz budget = 50 ms")
    a3.set_ylabel("per-rollout p99 latency (ms)")
    a3.set_title("Inference latency"); a3.legend(fontsize=8)

    fig.suptitle("Task 5 (extended) — BC vs TD3+BC vs AWAC vs IQL  |  "
                 + "  ".join(f"{n} {results[n]['succ'].mean():.3f}" for n in names))
    path = os.path.join(OUT_DIR, "task5_extended_comparison.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

    jpath = os.path.join(OUT_DIR, "task5_extended.json")
    json.dump({"n_rollouts": N, "seed": SEED, "policies": table},
              open(jpath, "w"), indent=2)
    print(f"\nSaved: {path}\nSaved: {jpath}")


if __name__ == "__main__":
    main()
