#!/usr/bin/env python
"""Re-run the Section-1 BC training (seed 42 → identical to the saved run),
capture per-epoch train/val MSE, and plot the loss curve.

The training loop in residual_lift.section1_bc prints the history but doesn't
persist it, and out/bc.pt holds weights only. Because training is fully seeded,
re-running here reproduces the exact curve behind out/bc.pt. We do NOT overwrite
the checkpoint — this only produces the plot + a history json.

Run:  python scripts/plot_bc_loss.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, BC_CKPT
from residual_lift.data import LiftPHDataset, load_obs_stats, read_mask
from residual_lift.section1_bc import (
    BCPolicy, _epoch_mse, BC_EPOCHS, BC_LR, BC_BS, BC_PATIENCE,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def run():
    # Mirror train_bc() exactly so the curve matches the saved checkpoint.
    torch.manual_seed(42); np.random.seed(42)

    train_keys, valid_keys = read_mask("train"), read_mask("valid")
    train_ds = LiftPHDataset(demo_keys=train_keys)
    valid_ds = LiftPHDataset(demo_keys=valid_keys)
    obs_mean, obs_std = load_obs_stats(train_ds)
    bc = BCPolicy(obs_mean, obs_std).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BC_BS, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BC_BS, shuffle=False)
    opt = torch.optim.Adam(bc.parameters(), lr=BC_LR)

    hist = {"epoch": [], "train_mse": [], "val_mse": []}
    best_val, best_epoch, since = float("inf"), -1, 0
    for epoch in range(BC_EPOCHS):
        bc.train()
        run_, n = 0.0, 0
        for batch in train_loader:
            obs = batch["obs"].to(DEVICE); act = batch["action"].to(DEVICE)
            loss = F.mse_loss(bc(obs), act)
            opt.zero_grad(); loss.backward(); opt.step()
            run_ += loss.item() * obs.shape[0]; n += obs.shape[0]
        train_mse = run_ / n
        val_mse = _epoch_mse(bc, valid_loader)
        hist["epoch"].append(epoch); hist["train_mse"].append(train_mse)
        hist["val_mse"].append(val_mse)

        improved = val_mse < best_val - 1e-6
        if improved:
            best_val, best_epoch, since = val_mse, epoch, 0
        else:
            since += 1
        if since >= BC_PATIENCE:
            print(f"early stop at epoch {epoch} (best epoch {best_epoch}, val {best_val:.5f})")
            break

    return hist, best_epoch, best_val


def plot(hist, best_epoch, best_val):
    ep = hist["epoch"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ep, hist["train_mse"], "-o", ms=3, label="train MSE", color="#1f77b4")
    ax.plot(ep, hist["val_mse"],   "-o", ms=3, label="val MSE",   color="#d62728")
    ax.axvline(best_epoch, ls="--", color="gray", lw=1)
    ax.scatter([best_epoch], [best_val], s=80, facecolors="none",
               edgecolors="black", zorder=5,
               label=f"best (epoch {best_epoch}, val {best_val:.4f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (action)")
    ax.set_title("BC training loss — lift-ph (seed 42, early stop on val MSE)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "bc_loss_curve.png")
    fig.savefig(out_png, dpi=130)
    with open(os.path.join(OUT_DIR, "bc_loss_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"wrote {out_png}")
    print(f"wrote {os.path.join(OUT_DIR, 'bc_loss_history.json')}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    hist, best_epoch, best_val = run()
    plot(hist, best_epoch, best_val)
