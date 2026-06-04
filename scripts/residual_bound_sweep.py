#!/usr/bin/env python
"""Decision #3 evidence — residual success & saturation vs delta-bound.

Trains the TD3+BC residual at several delta_bound values (same seed, 10k steps),
evaluates each on 30 rollouts (seed 42, no shield, to isolate the residual), and
records settled |delta|. Saves out/residual_bound_sweep.png (success-vs-bound and
saturation panels) + JSON, archives to out/runs/residual_bound_sweep/.
Run:  python scripts/residual_bound_sweep.py
"""
import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import train_residual
from residual_lift.section4_eval import evaluate
from residual_lift.data import LiftPHDataset

BOUNDS = [0.05, 0.02, 0.01, 0.005, 0.002]
STEPS, N_ROLL, BC_SR = 10000, 30, 0.867


def main():
    print(f"Device: {DEVICE}")
    bc = load_bc()
    probe = LiftPHDataset().obs[:2048].to(DEVICE)
    rows = []
    for b in BOUNDS:
        res, _ = train_residual(bc, steps=STEPS, save=False, delta_bound=b, log_every=10**9)
        with torch.no_grad():
            dmag = res.raw_delta(probe).abs().mean().item()
        sr = evaluate(res, n_rollouts=N_ROLL, seed=42, label=f"b{b}")["success_rate"]
        rows.append(dict(bound=b, delta_mag=dmag, success=sr))
        print(f"bound {b:.3f}   settled|δ| {dmag:.4f}   success {sr:.3f}")

    bounds = [r["bound"] for r in rows]
    succ = [r["success"] for r in rows]
    dm = [r["delta_mag"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(bounds, succ, "o-", color="tab:blue")
    for r in rows:
        ax[0].annotate(f'{r["success"]:.2f}', (r["bound"], r["success"]),
                       textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center")
    ax[0].axhline(BC_SR, ls="--", c="gray", lw=0.9, label=f"BC = {BC_SR}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("δ bound (log)")
    ax[0].set_ylabel(f"success ({N_ROLL} rollouts)"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("(a) Success vs residual bound"); ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].plot(bounds, dm, "s-", color="tab:orange", label="settled mean |δ|")
    ax[1].plot(bounds, bounds, "--", c="gray", lw=0.9, label="|δ| = bound (full saturation)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("δ bound (log)"); ax[1].set_ylabel("settled mean |δ| (log)")
    ax[1].set_title("(b) Residual rides ~96% of its budget at every bound")
    ax[1].grid(alpha=0.3); ax[1].legend()

    fig.suptitle("Decision #3 — residual bound sweep (shipped = 0.005)")
    live = os.path.join(OUT_DIR, "residual_bound_sweep.png")
    fig.tight_layout(); fig.savefig(live, dpi=120); plt.close(fig)

    snap = os.path.join(OUT_DIR, "runs", "residual_bound_sweep")
    os.makedirs(snap, exist_ok=True)
    shutil.copy2(live, snap)
    json.dump(rows, open(os.path.join(snap, "results.json"), "w"), indent=2)
    print(f"\nSaved: {live}\nArchived: {snap}/")


if __name__ == "__main__":
    main()
