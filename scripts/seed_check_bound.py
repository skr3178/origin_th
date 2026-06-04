#!/usr/bin/env python
"""Multi-seed robustness check: delta_bound 0.002 vs 0.005.

Turns the "0.002 vs 0.005 is within noise" claim from an inference into measured
evidence. Trains the residual at each bound across several seeds (all <=10k
steps), evaluates each on the same 30 rollouts (eval seed 42 -> identical start
states), and reports whether the two bounds' success distributions overlap / the
per-seed ranking flips.

Saves out/residual_bound_seedcheck.{png,json}.
Run:  python scripts/seed_check_bound.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, BC_CKPT, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import train_residual
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import evaluate

BOUNDS = [0.002, 0.005]
SEEDS  = [0, 1, 2]
STEPS  = 10000          # <=10k cap
N_ROLL = 30
EVAL_SEED = 42          # fixed -> all policies face identical start states


def main():
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")
    bc = load_bc()
    shield = build_shield(verbose=False)

    results = {f"{b}": [] for b in BOUNDS}
    for b in BOUNDS:
        for s in SEEDS:
            print(f"\n=== bound {b}  seed {s} ===")
            res, _ = train_residual(bc, steps=STEPS, delta_bound=b, seed=s,
                                    save=False, log_every=99999)
            r = evaluate(res, shield=shield, n_rollouts=N_ROLL, seed=EVAL_SEED,
                         label=f"b{b}_s{s}")
            sr = r["success_rate"]
            results[f"{b}"].append({"seed": s, "success_rate": sr,
                                    "shield_clip_rate": r["shield_clip_rate"]})
            print(f"bound {b} seed {s}: success {sr:.3f}")

    # --- summary stats ---
    summary = {}
    for b in BOUNDS:
        srs = [d["success_rate"] for d in results[f"{b}"]]
        summary[f"{b}"] = {"mean": float(np.mean(srs)), "std": float(np.std(srs, ddof=1)),
                           "min": float(min(srs)), "max": float(max(srs)), "per_seed": srs}

    # rank flip: does the better bound differ across seeds?
    wins = {f"{b}": 0 for b in BOUNDS}
    ties = 0
    for i, s in enumerate(SEEDS):
        a = results[f"{BOUNDS[0]}"][i]["success_rate"]
        c = results[f"{BOUNDS[1]}"][i]["success_rate"]
        if a > c: wins[f"{BOUNDS[0]}"] += 1
        elif c > a: wins[f"{BOUNDS[1]}"] += 1
        else: ties += 1

    payload = {"bounds": BOUNDS, "seeds": SEEDS, "steps": STEPS,
               "n_rollouts": N_ROLL, "eval_seed": EVAL_SEED,
               "per_run": results, "summary": summary,
               "per_seed_wins": wins, "ties": ties}
    jpath = os.path.join(OUT_DIR, "residual_bound_seedcheck.json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)

    # --- plot: per-seed dots + mean bar per bound ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(BOUNDS))
    for xi, b in zip(x, BOUNDS):
        srs = summary[f"{b}"]["per_seed"]
        ax.scatter([xi] * len(srs), srs, s=70, zorder=3,
                   label=f"bound {b}" if xi == 0 else None, color="tab:blue")
        ax.hlines(summary[f"{b}"]["mean"], xi - 0.18, xi + 0.18, color="black", lw=2)
    ax.axhline(0.867, ls="--", c="tab:green", lw=1, label="BC (26/30)")
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(mean {summary[str(b)]['mean']:.2f})" for b in BOUNDS])
    ax.set_ylabel("success rate (30 rollouts)")
    ax.set_title(f"Bound 0.002 vs 0.005 — {len(SEEDS)} seeds (eval seed {EVAL_SEED})")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, axis="y"); ax.legend(loc="lower right")
    fig.tight_layout()
    ppath = os.path.join(OUT_DIR, "residual_bound_seedcheck.png")
    fig.savefig(ppath, dpi=130); plt.close(fig)

    print("\n================ SUMMARY ================")
    for b in BOUNDS:
        sm = summary[f"{b}"]
        print(f"bound {b}: mean {sm['mean']:.3f}  std {sm['std']:.3f}  "
              f"range [{sm['min']:.2f}, {sm['max']:.2f}]  per-seed {sm['per_seed']}")
    print(f"per-seed wins: {wins}  ties: {ties}")
    print(f"saved: {jpath}\n        {ppath}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
