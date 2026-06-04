#!/usr/bin/env python
"""Decision #2 ablation: how the residual δ is bounded.

Three ways to keep δ in [-bound, +bound], everything else identical (frozen BC,
TD3+BC, bound 0.005, seed 42, 10k steps, per-dim shield, 30-rollout seed-42 eval):
  - tanh    : bound * tanh(f)      smooth, hard bound (shipped)
  - clip    : clamp(f, ±bound)     hard bound, ZERO gradient once saturated
  - softsign: bound * softsign(f)  smooth, softer/slower saturation

Why it matters here: δ settles at ~96% of the bound, i.e. near-saturated. At
saturation hard-clip has zero gradient (can't fine-tune), while tanh keeps a small
gradient. This checks whether that theoretical difference shows up in practice.

Saves out/ablation_activation.{png,json}.
Run:  python scripts/ablation_activation.py
"""
import os, sys, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, BC_CKPT, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import train_residual
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import evaluate

SEED, STEPS, BOUND, N_ROLL = 42, 10000, 0.005, 30

# (label, activation)
RUNS = [
    ("tanh x bound (shipped)", "tanh"),
    ("hard clip",              "clip"),
    ("softsign x bound",       "softsign"),
]


def main():
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")
    bc = load_bc()
    shield = build_shield(verbose=False)

    results = {}
    for label, act in RUNS:
        print(f"\n=== {label} ===")
        res, hist = train_residual(
            bc, steps=STEPS, delta_bound=BOUND, seed=SEED, save=False,
            activation=act, log_every=2000)
        r = evaluate(res, shield=shield, n_rollouts=N_ROLL, seed=SEED, label=label)
        results[label] = {
            "activation": act,
            "success_rate": r["success_rate"],
            "shield_clip_rate": r["shield_clip_rate"],
            "delta_mag_final": float(hist["delta_mag"][-1]),
        }
        print(f"{label}: success {r['success_rate']:.3f}  "
              f"delta_mag {hist['delta_mag'][-1]:.4f}  clip {r['shield_clip_rate']:.3f}")

    payload = {"seed": SEED, "steps": STEPS, "bound": BOUND, "n_rollouts": N_ROLL,
               "bc_success": 0.867, "runs": results}
    jpath = os.path.join(OUT_DIR, "ablation_activation.json")
    json.dump(payload, open(jpath, "w"), indent=2)

    labels = list(results.keys())
    succ = [results[k]["success_rate"] for k in labels]
    dmag = [results[k]["delta_mag_final"] for k in labels]
    colors = ["tab:gray", "tab:red", "tab:purple"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(range(len(labels)), succ, color=colors)
    ax1.axhline(0.867, ls="--", c="tab:green", lw=1, label="BC (0.867)")
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.set_ylabel("success rate (30 rollouts)"); ax1.set_ylim(0, 1.02)
    ax1.set_title("δ activation — success"); ax1.legend(); ax1.grid(alpha=0.3, axis="y")
    ax2.bar(range(len(labels)), dmag, color=colors)
    ax2.axhline(BOUND, ls="--", c="r", lw=0.8, label=f"bound={BOUND}")
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=12, ha="right")
    ax2.set_ylabel("settled delta_mag"); ax2.set_title("Residual magnitude")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    ppath = os.path.join(OUT_DIR, "ablation_activation.png")
    fig.savefig(ppath, dpi=130); plt.close(fig)

    print("\n================ SUMMARY (decision #2) ================")
    print(f"{'activation':<24} {'success':>8} {'delta_mag':>10} {'clip':>7}")
    for k in labels:
        r = results[k]
        print(f"{k:<24} {r['success_rate']:>8.3f} {r['delta_mag_final']:>10.4f} {r['shield_clip_rate']:>7.3f}")
    print(f"\nsaved: {jpath}\n        {ppath}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
