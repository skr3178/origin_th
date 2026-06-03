#!/usr/bin/env python
"""Entrypoint: train the residual policy (Section 2) on the frozen BC.

Loads out/bc.pt, trains TD3+BC, saves out/residual.pt, and writes a 4-panel
training-curve plot (q_mean, delta_mag, critic_loss, actor_loss) to out/.
Run:  python scripts/train_residual.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, BC_CKPT, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import train_residual, DELTA_BOUND


def plot_history(history, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("q_mean", "min-twin Q (batch)"), ("delta_mag", "mean |delta|"),
              ("critic_loss", "critic loss"), ("actor_loss", "actor loss")]
    for ax, (key, title) in zip(axes.ravel(), panels):
        ax.plot(history["step"], history[key], marker=".", ms=3)
        ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    axes.ravel()[1].axhline(DELTA_BOUND, ls="--", c="r", lw=0.8, label=f"bound={DELTA_BOUND}")
    axes.ravel()[1].legend()
    fig.suptitle("Residual TD3+BC training diagnostics")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")
    bc_policy = load_bc()
    residual, history = train_residual(bc_policy)
    out_plot = os.path.join(OUT_DIR, "residual_training_curves.png")
    plot_history(history, out_plot)
    print(f"Training curves saved: {out_plot}")
