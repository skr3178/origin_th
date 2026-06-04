#!/usr/bin/env python
"""Task-4 ablation: shield margin sweep.

The shield clips each action dim to the train-split demo range expanded by
margin_frac * span (then clamped to ±1). This sweeps the margin and measures, for
the SHIPPED residual (out/residual.pt), how often the shield fires (clip rate) and
whether success changes — no training, just re-evaluation with different shields.

Expected: clip rate falls as the margin widens (looser box); success is flat,
because the bounded residual rarely leaves the demo box anyway. Confirms the 5%
choice is a safe non-intrusive default and quantifies the margin↔clip-rate trade-off
the writeup previously only argued.

Saves out/ablation_shield_margin.{png,json}.
Run:  python scripts/ablation_shield_margin.py
"""
import os, sys, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, BC_CKPT, RESIDUAL_CKPT, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import load_residual
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import evaluate

SEED, N_ROLL = 42, 30
MARGINS = [0.0, 0.05, 0.10, 0.20, 0.50]   # 0.05 is the shipped default


def main():
    for ck in (BC_CKPT, RESIDUAL_CKPT):
        if not os.path.exists(ck):
            sys.exit(f"Missing {ck}. Train first.")
    bc = load_bc()
    residual = load_residual(bc)

    results = {}
    for m in MARGINS:
        shield = build_shield(margin_frac=m, verbose=False)
        r = evaluate(residual, shield=shield, n_rollouts=N_ROLL, seed=SEED, label=f"margin{m}")
        results[f"{m:.2f}"] = {"margin_frac": m,
                               "success_rate": r["success_rate"],
                               "shield_clip_rate": r["shield_clip_rate"]}
        tag = "  <- shipped" if abs(m - 0.05) < 1e-9 else ""
        print(f"margin {m:.2f}: success {r['success_rate']:.3f}  "
              f"clip_rate {r['shield_clip_rate']:.3f}{tag}")

    payload = {"seed": SEED, "n_rollouts": N_ROLL, "shipped_margin": 0.05,
               "residual_ckpt": os.path.relpath(RESIDUAL_CKPT, REPO), "runs": results}
    jpath = os.path.join(OUT_DIR, "ablation_shield_margin.json")
    json.dump(payload, open(jpath, "w"), indent=2)

    ms   = [results[k]["margin_frac"] for k in results]
    clip = [results[k]["shield_clip_rate"] for k in results]
    succ = [results[k]["success_rate"] for k in results]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ms, clip, "o-", color="tab:red", label="shield clip rate")
    ax1.set_xlabel("margin_frac (fraction of per-dim demo span)")
    ax1.set_ylabel("shield clip rate", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red"); ax1.grid(alpha=0.3)
    ax1.axvline(0.05, ls="--", c="gray", lw=1, label="shipped (0.05)")
    ax2 = ax1.twinx()
    ax2.plot(ms, succ, "s--", color="tab:blue", label="success rate")
    ax2.axhline(0.867, ls=":", c="tab:green", lw=1)
    ax2.set_ylabel("success rate (30 rollouts)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue"); ax2.set_ylim(0, 1.02)
    ax1.set_title("Task-4 shield margin sweep (shipped residual, seed 42)")
    fig.tight_layout()
    ppath = os.path.join(OUT_DIR, "ablation_shield_margin.png")
    fig.savefig(ppath, dpi=130); plt.close(fig)

    print("\n================ SUMMARY (Task 4 — margin) ================")
    print(f"{'margin_frac':<12} {'success':>8} {'clip_rate':>10}")
    for k in results:
        r = results[k]
        print(f"{r['margin_frac']:<12.2f} {r['success_rate']:>8.3f} {r['shield_clip_rate']:>10.3f}")
    print(f"\nsaved: {jpath}\n        {ppath}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
