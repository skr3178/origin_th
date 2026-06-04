#!/usr/bin/env python
"""Overlay AWAC vs IQL training diagnostics on shared axes (no run — parses logs).

Reads the per-step diagnostics that the cross-algorithm study already logged to
out/algo_comparison.log ([AWAC]/[IQL] step ... q_mean ... delta_mag ...
critic_loss ... actor_loss ...), groups them by seed (detected via step resets),
and overlays the two algorithms — mean across seeds (bold) + individual seeds
(faint) — in a 4-panel figure.

TD3+BC is omitted: its per-step history was never saved as data (only the
residual_training_curves.png image), and the seed-check run logged only endpoints.

Saves out/algo_diag_overlay.png.  Run:  python scripts/plot_algo_diag_overlay.py
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG  = os.path.join(REPO, "out", "algo_comparison.log")
OUT  = os.path.join(REPO, "out", "algo_diag_overlay.png")

LINE = re.compile(
    r"\[(AWAC|IQL)\]\s+step\s+(\d+)\s+q_mean\s+([-+.\d]+)\s+delta_mag\s+([.\d]+)\s+"
    r"critic_loss\s+([-.\d]+)\s+actor_loss\s+([-.\d]+)")
KEYS = ["q_mean", "delta_mag", "critic_loss", "actor_loss"]


def parse():
    """-> {algo: [ {step:[], q_mean:[], ...} per seed ]}.  New seed when step resets."""
    runs = {"AWAC": [], "IQL": []}
    for line in open(LOG):
        m = LINE.search(line)
        if not m:
            continue
        algo, step = m.group(1), int(m.group(2))
        vals = dict(step=step, q_mean=float(m.group(3)), delta_mag=float(m.group(4)),
                    critic_loss=float(m.group(5)), actor_loss=float(m.group(6)))
        seeds = runs[algo]
        if not seeds or step < seeds[-1]["step"][-1]:   # step reset -> new seed
            seeds.append({k: [] for k in ["step"] + KEYS})
        cur = seeds[-1]
        for k in ["step"] + KEYS:
            cur[k].append(vals[k])
    return runs


def main():
    runs = parse()
    colors = {"AWAC": "tab:orange", "IQL": "tab:blue"}
    titles = {"q_mean": "min-twin Q (q_mean)", "delta_mag": "mean |delta|",
              "critic_loss": "critic loss", "actor_loss": "actor loss"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, key in zip(axes.ravel(), KEYS):
        for algo, seeds in runs.items():
            if not seeds:
                continue
            steps = np.array(seeds[0]["step"])
            stack = np.array([s[key] for s in seeds if len(s[key]) == len(steps)])
            for s in stack:                                    # faint per-seed
                ax.plot(steps, s, color=colors[algo], alpha=0.25, lw=1)
            ax.plot(steps, stack.mean(0), color=colors[algo], lw=2.2,
                    marker="o", ms=4, label=f"{algo} (mean of {len(stack)} seeds)")
        ax.set_title(titles[key]); ax.set_xlabel("training step"); ax.grid(alpha=0.3)
        if key == "delta_mag":
            ax.axhline(0.005, ls="--", c="k", lw=0.8, label="bound=0.005")
        ax.legend(fontsize=8)
    fig.suptitle("Residual training diagnostics — AWAC vs IQL (3 seeds, bound 0.005, 10k steps)\n"
                 "logged every 2000 steps; TD3+BC history not saved as data (see residual_training_curves.png)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130); plt.close(fig)
    print(f"parsed seeds: AWAC={len(runs['AWAC'])}  IQL={len(runs['IQL'])}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
