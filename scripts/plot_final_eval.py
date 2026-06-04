#!/usr/bin/env python
"""Task 5 — graphs for the final eval (BC vs Residual+Shield, 30 rollouts, seed 42).

Because both policies are evaluated from the same seed (same cube starts), we can
compare them rollout-by-rollout and show exactly which rollouts the residual flips.
Produces out/final_eval_comparison.png (4 panels) and prints a flip table.

Run:  python scripts/plot_final_eval.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import load_residual
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import make_env, flatten_obs_dict

N, SEED, MAX_STEPS = 30, 42, 400


def eval_per_rollout(policy, shield=None):
    """Re-seed + rebuild env (same starts), return per-rollout success/steps/latency."""
    np.random.seed(SEED); torch.manual_seed(SEED)
    env = make_env()
    succ, steps, lat_p99 = [], [], []
    for _ in range(N):
        obs = flatten_obs_dict(env.reset())
        prev, ok, lats = None, False, []
        for t in range(MAX_STEPS):
            t0 = time.perf_counter()
            with torch.no_grad():
                a = policy.act(torch.from_numpy(obs).float().to(DEVICE)).cpu().numpy()
            lats.append((time.perf_counter() - t0) * 1000)
            if shield is not None:
                a = shield(a, prev)
            prev = a
            od, _, done, _ = env.step(a)
            obs = flatten_obs_dict(od)
            if env._check_success():
                ok = True; break
            if done:
                break
        succ.append(ok); steps.append(t + 1); lat_p99.append(np.percentile(lats, 99))
    return np.array(succ), np.array(steps), np.array(lat_p99)


def main():
    print(f"Device: {DEVICE}")
    bc = load_bc()
    residual = load_residual(bc)
    shield = build_shield(verbose=False)

    print("Evaluating BC...")
    bc_s, bc_steps, bc_lat = eval_per_rollout(bc)
    print("Evaluating Residual+Shield...")
    rs_s, rs_steps, rs_lat = eval_per_rollout(residual, shield=shield)

    # --- flip analysis (same starts) ---
    both = int((bc_s & rs_s).sum())
    only_bc = int((bc_s & ~rs_s).sum())
    only_rs = int((~bc_s & rs_s).sum())
    neither = int((~bc_s & ~rs_s).sum())
    print("\nPaired outcomes (same starts):")
    print(f"  both succeed : {both}")
    print(f"  BC only      : {only_bc}  (residual broke these)")
    print(f"  Residual only: {only_rs}  (residual fixed these)")
    print(f"  both fail    : {neither}")
    print(f"BC {bc_s.mean():.3f}  vs  Residual+Shield {rs_s.mean():.3f}  "
          f"(net {rs_s.mean()-bc_s.mean():+.3f})")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # (a) success-rate bar
    a0 = ax[0, 0]
    bars = a0.bar(["BC", "Residual+Shield"], [bc_s.mean(), rs_s.mean()],
                  color=["tab:blue", "tab:orange"])
    for b, s in zip(bars, [bc_s, rs_s]):
        a0.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{s.mean():.3f}\n({int(s.sum())}/{N})", ha="center", va="bottom")
    a0.set_ylim(0, 1.05); a0.set_ylabel("success rate")
    a0.set_title("Success rate (30 rollouts, seed 42)")

    # (b) per-rollout paired outcome matrix
    a1 = ax[0, 1]
    grid = np.vstack([bc_s.astype(int), rs_s.astype(int)])   # 2 x N
    a1.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    a1.set_yticks([0, 1]); a1.set_yticklabels(["BC", "Res+Shield"])
    a1.set_xlabel("rollout index (same start across both rows)")
    a1.set_title("Per-rollout outcome — green=success, red=fail")
    for i in range(N):                                       # mark flips
        if bc_s[i] != rs_s[i]:
            a1.text(i, -0.6, "▼", ha="center", va="bottom",
                    color=("tab:green" if rs_s[i] else "tab:red"), fontsize=8)

    # (c) episode length per rollout
    a2 = ax[1, 0]
    idx = np.arange(N); w = 0.4
    a2.bar(idx - w/2, bc_steps, w, label="BC", color="tab:blue")
    a2.bar(idx + w/2, rs_steps, w, label="Res+Shield", color="tab:orange")
    a2.axhline(MAX_STEPS, ls="--", c="gray", lw=0.8, label=f"horizon={MAX_STEPS}")
    a2.set_xlabel("rollout index"); a2.set_ylabel("episode length (steps)")
    a2.set_title("Episode length (full horizon = timed-out failure)"); a2.legend(fontsize=8)

    # (d) per-rollout p99 latency
    a3 = ax[1, 1]
    a3.boxplot([bc_lat, rs_lat], labels=["BC", "Res+Shield"])
    a3.set_ylabel("per-rollout p99 latency (ms)")
    a3.set_title("Inference latency (well under the 20 Hz = 50 ms budget)")

    fig.suptitle(f"Task 5 — BC vs Residual+Shield  |  BC {bc_s.mean():.3f}  "
                 f"Residual {rs_s.mean():.3f}  (both ~1 SE of each other)")
    path = os.path.join(OUT_DIR, "final_eval_comparison.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
