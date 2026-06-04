#!/usr/bin/env python
"""Decision #5 ablation: sparse terminal reward vs dense reward shaping.

The only difference between runs is the reward fed to the critic; everything else
(frozen BC, TD3+BC, bound 0.005, seed 42, 10k steps, per-dim shield, 30-rollout
seed-42 eval) is identical. Shaped reward = sparse + coef*(-||gripper_to_cube_pos||),
i.e. the brief's `-|cube - eef|` dense bonus. We test two coefficients so the result
isn't an artifact of one λ.

Question: does telling the critic "get closer to the cube" help the residual on
all-expert data? Hypothesis (defended in the writeup): no — the ceiling is BC, and
non-potential shaping biases toward reaching over grasping, so it ties-or-hurts.

Saves out/ablation_reward_shaping.{png,json}.
Run:  python scripts/ablation_reward_shaping.py
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

# (label, reward_shaping, shape_coef)
RUNS = [
    ("sparse (shipped)", False, 0.0),
    ("shaped coef=0.1",  True,  0.1),
    ("shaped coef=1.0",  True,  1.0),
]


def main():
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")
    bc = load_bc()
    shield = build_shield(verbose=False)

    results = {}
    for label, shaping, coef in RUNS:
        print(f"\n=== {label} ===")
        res, hist = train_residual(
            bc, steps=STEPS, delta_bound=BOUND, seed=SEED, save=False,
            reward_shaping=shaping, shape_coef=coef, log_every=2000)
        r = evaluate(res, shield=shield, n_rollouts=N_ROLL, seed=SEED, label=label)
        results[label] = {
            "reward_shaping": shaping, "shape_coef": coef,
            "success_rate": r["success_rate"],
            "shield_clip_rate": r["shield_clip_rate"],
            "delta_mag_final": float(hist["delta_mag"][-1]),
        }
        print(f"{label}: success {r['success_rate']:.3f}  "
              f"delta_mag {hist['delta_mag'][-1]:.4f}  clip {r['shield_clip_rate']:.3f}")

    payload = {"seed": SEED, "steps": STEPS, "bound": BOUND, "n_rollouts": N_ROLL,
               "bc_success": 0.867, "runs": results}
    jpath = os.path.join(OUT_DIR, "ablation_reward_shaping.json")
    json.dump(payload, open(jpath, "w"), indent=2)

    # --- plot: success + delta_mag per reward variant ---
    labels = list(results.keys())
    succ = [results[k]["success_rate"] for k in labels]
    dmag = [results[k]["delta_mag_final"] for k in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(range(len(labels)), succ, color=["tab:gray", "tab:orange", "tab:red"])
    ax1.axhline(0.867, ls="--", c="tab:green", lw=1, label="BC (0.867)")
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, rotation=12, ha="right")
    ax1.set_ylabel("success rate (30 rollouts)"); ax1.set_ylim(0, 1.02)
    ax1.set_title("Reward shaping vs sparse — success"); ax1.legend(); ax1.grid(alpha=0.3, axis="y")
    ax2.bar(range(len(labels)), dmag, color=["tab:gray", "tab:orange", "tab:red"])
    ax2.axhline(BOUND, ls="--", c="r", lw=0.8, label=f"bound={BOUND}")
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=12, ha="right")
    ax2.set_ylabel("settled delta_mag"); ax2.set_title("Residual magnitude")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    ppath = os.path.join(OUT_DIR, "ablation_reward_shaping.png")
    fig.savefig(ppath, dpi=130); plt.close(fig)

    print("\n================ SUMMARY (decision #5) ================")
    print(f"{'reward':<20} {'success':>8} {'delta_mag':>10} {'clip':>7}")
    for k in labels:
        r = results[k]
        print(f"{k:<20} {r['success_rate']:>8.3f} {r['delta_mag_final']:>10.4f} {r['shield_clip_rate']:>7.3f}")
    print(f"\nsaved: {jpath}\n        {ppath}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
