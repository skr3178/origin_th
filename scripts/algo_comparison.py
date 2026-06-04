#!/usr/bin/env python
"""Decision #4 cross-algorithm study: TD3+BC vs AWAC vs IQL (non-TD3+BC variants).

Same residual framing (frozen BC + bounded delta, bound=0.005), same eval, only
the offline-RL algorithm differs. Trains AWAC and IQL at 3 seeds each (10k steps),
evaluates each (30 rollouts, eval seed 42 = identical starts), and reuses the
TD3+BC 3-seed numbers already measured in residual_bound_seedcheck.json (bound
0.005). Reports success (mean/std + significance vs TD3+BC), delta_mag, and shield
clip rate — the metrics that actually separate algorithms on all-expert data.

Saves out/algo_comparison.{png,json} and a 4-panel diagnostics plot per variant.
Run:  python scripts/algo_comparison.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from residual_lift.config import DEVICE, BC_CKPT, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import evaluate
from residual_lift.algorithms import train_residual_awac, train_residual_iql

SEEDS = [0, 1, 2]
STEPS = 10000
BOUND = 0.005
N_ROLL = 30
EVAL_SEED = 42
BC_SR = 0.867

ALGOS = {"AWAC": train_residual_awac, "IQL": train_residual_iql}


def diag_plot(history, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (k, t) in zip(axes.ravel(), [("q_mean", "min-twin Q"), ("delta_mag", "mean |delta|"),
                                         ("critic_loss", "critic loss"), ("actor_loss", "actor loss")]):
        ax.plot(history["step"], history[k], marker=".", ms=3)
        ax.set_title(t); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    axes.ravel()[1].axhline(BOUND, ls="--", c="r", lw=0.8, label=f"bound={BOUND}")
    axes.ravel()[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")
    bc = load_bc()
    shield = build_shield(verbose=False)

    results = {}

    # Reuse TD3+BC @ bound 0.005 from the seed-check (no retrain).
    sc_path = os.path.join(OUT_DIR, "residual_bound_seedcheck.json")
    if os.path.exists(sc_path):
        sc = json.load(open(sc_path))
        td3 = [d["success_rate"] for d in sc["per_run"]["0.005"]]
        results["TD3+BC"] = {"success": td3, "delta_mag": [None] * len(td3),
                             "clip_rate": [d["shield_clip_rate"] for d in sc["per_run"]["0.005"]],
                             "source": "reused from residual_bound_seedcheck.json"}
        print(f"TD3+BC (reused): {td3}")

    for name, fn in ALGOS.items():
        succ, dmag, clip = [], [], []
        for s in SEEDS:
            print(f"\n=== {name} seed {s} ===")
            res, hist = fn(bc, steps=STEPS, delta_bound=BOUND, seed=s, log_every=2000)
            r = evaluate(res, shield=shield, n_rollouts=N_ROLL, seed=EVAL_SEED, label=f"{name}_s{s}")
            succ.append(r["success_rate"]); clip.append(r["shield_clip_rate"])
            dmag.append(hist["delta_mag"][-1])
            print(f"{name} seed {s}: success {r['success_rate']:.3f}  "
                  f"delta_mag {hist['delta_mag'][-1]:.4f}  clip {r['shield_clip_rate']:.3f}")
            if s == SEEDS[0]:
                diag_plot(hist, f"{name} residual — training diagnostics (seed 0)",
                          os.path.join(OUT_DIR, f"algo_diag_{name.lower().replace('+','')}.png"))
        results[name] = {"success": succ, "delta_mag": dmag, "clip_rate": clip}

    # --- summary + significance vs TD3+BC ---
    summary = {}
    base = results.get("TD3+BC", {}).get("success")
    for name, d in results.items():
        srs = d["success"]
        row = {"mean": float(np.mean(srs)), "std": float(np.std(srs, ddof=1)),
               "per_seed": srs}
        if base is not None and name != "TD3+BC":
            _, p = stats.ttest_ind(srs, base, equal_var=False)
            row["welch_p_vs_td3bc"] = float(p)
            row["significant_vs_td3bc"] = bool(p < 0.05)
        dm = [x for x in d.get("delta_mag", []) if x is not None]
        if dm: row["delta_mag_mean"] = float(np.mean(dm))
        row["clip_rate_mean"] = float(np.mean(d["clip_rate"]))
        summary[name] = row

    payload = {"seeds": SEEDS, "steps": STEPS, "bound": BOUND, "n_rollouts": N_ROLL,
               "eval_seed": EVAL_SEED, "bc_success": BC_SR,
               "per_run": results, "summary": summary}
    jpath = os.path.join(OUT_DIR, "algo_comparison.json")
    json.dump(payload, open(jpath, "w"), indent=2)

    # --- plot: per-seed dots + mean per algorithm ---
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for xi, nm in enumerate(names):
        srs = results[nm]["success"]
        ax.scatter([xi] * len(srs), srs, s=70, zorder=3, color="tab:blue")
        ax.hlines(summary[nm]["mean"], xi - 0.18, xi + 0.18, color="black", lw=2)
    ax.axhline(BC_SR, ls="--", c="tab:green", lw=1, label=f"BC ({BC_SR:.3f})")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n(mean {summary[n]['mean']:.2f})" for n in names])
    ax.set_ylabel("success rate (30 rollouts)"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Residual algorithm comparison — {len(SEEDS)} seeds, bound {BOUND}")
    ax.grid(alpha=0.3, axis="y"); ax.legend(loc="lower right")
    fig.tight_layout()
    ppath = os.path.join(OUT_DIR, "algo_comparison.png")
    fig.savefig(ppath, dpi=130); plt.close(fig)

    print("\n================ SUMMARY ================")
    for nm in names:
        r = summary[nm]
        extra = ""
        if "welch_p_vs_td3bc" in r:
            extra = f"  vs TD3+BC: Welch p={r['welch_p_vs_td3bc']:.3f} ({'NS' if not r['significant_vs_td3bc'] else 'SIG'})"
        dm = f"  delta_mag {r['delta_mag_mean']:.4f}" if "delta_mag_mean" in r else ""
        print(f"{nm:8s}: mean {r['mean']:.3f}  std {r['std']:.3f}  "
              f"clip {r['clip_rate_mean']:.3f}{dm}{extra}")
    print(f"\nsaved: {jpath}\n        {ppath}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
