#!/usr/bin/env python
"""Decision #1 ablation: residual head delta(s) vs delta(s, a_BC(s)).

The only difference is whether the frozen BC action a_BC(s) is fed to the residual
MLP as extra input. Everything else (frozen BC, TD3+BC, bound 0.005, seed 42, 10k
steps, per-dim shield, 30-rollout seed-42 eval) is identical.

Argument under test: a_BC(s) is a deterministic function of s, so an MLP on s can
already recover it internally — feeding it explicitly should add no information.
This run checks that empirically (expected: statistically identical).

Saves out/ablation_architecture.{png,json}.
Run:  python scripts/ablation_architecture.py
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

# (label, condition_on_bc)
RUNS = [
    ("delta(s) (shipped)",      False),
    ("delta(s, a_BC(s))",       True),
]


def main():
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")
    bc = load_bc()
    shield = build_shield(verbose=False)

    results = {}
    for label, cond in RUNS:
        print(f"\n=== {label} ===")
        res, hist = train_residual(
            bc, steps=STEPS, delta_bound=BOUND, seed=SEED, save=False,
            condition_on_bc=cond, log_every=2000)
        n_params = sum(p.numel() for p in res.delta_net.parameters())
        r = evaluate(res, shield=shield, n_rollouts=N_ROLL, seed=SEED, label=label)
        results[label] = {
            "condition_on_bc": cond,
            "success_rate": r["success_rate"],
            "shield_clip_rate": r["shield_clip_rate"],
            "delta_mag_final": float(hist["delta_mag"][-1]),
            "delta_net_params": int(n_params),
        }
        print(f"{label}: success {r['success_rate']:.3f}  "
              f"delta_mag {hist['delta_mag'][-1]:.4f}  clip {r['shield_clip_rate']:.3f}  "
              f"params {n_params}")

    payload = {"seed": SEED, "steps": STEPS, "bound": BOUND, "n_rollouts": N_ROLL,
               "bc_success": 0.867, "runs": results}
    jpath = os.path.join(OUT_DIR, "ablation_architecture.json")
    json.dump(payload, open(jpath, "w"), indent=2)

    labels = list(results.keys())
    succ = [results[k]["success_rate"] for k in labels]
    dmag = [results[k]["delta_mag_final"] for k in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.bar(range(len(labels)), succ, color=["tab:gray", "tab:blue"])
    ax1.axhline(0.867, ls="--", c="tab:green", lw=1, label="BC (0.867)")
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, rotation=10, ha="right")
    ax1.set_ylabel("success rate (30 rollouts)"); ax1.set_ylim(0, 1.02)
    ax1.set_title("Residual input: delta(s) vs delta(s, a_BC)"); ax1.legend(); ax1.grid(alpha=0.3, axis="y")
    ax2.bar(range(len(labels)), dmag, color=["tab:gray", "tab:blue"])
    ax2.axhline(BOUND, ls="--", c="r", lw=0.8, label=f"bound={BOUND}")
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=10, ha="right")
    ax2.set_ylabel("settled delta_mag"); ax2.set_title("Residual magnitude")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    ppath = os.path.join(OUT_DIR, "ablation_architecture.png")
    fig.savefig(ppath, dpi=130); plt.close(fig)

    print("\n================ SUMMARY (decision #1) ================")
    print(f"{'head':<22} {'success':>8} {'delta_mag':>10} {'clip':>7} {'params':>8}")
    for k in labels:
        r = results[k]
        print(f"{k:<22} {r['success_rate']:>8.3f} {r['delta_mag_final']:>10.4f} "
              f"{r['shield_clip_rate']:>7.3f} {r['delta_net_params']:>8d}")
    print(f"\nsaved: {jpath}\n        {ppath}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
