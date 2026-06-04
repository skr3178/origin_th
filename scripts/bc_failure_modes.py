#!/usr/bin/env python
"""Generate more BC failure modes by sweeping training duration.

Trains BC at several fixed epoch budgets (underfit -> early-stop -> overfit), with
NO early stopping, then evaluates each on N rollouts and classifies every rollout
by manipulation phase (reach -> grasp -> lift). Shows how the failure-mode mix
shifts with training, and surfaces more/rarer failures than the shipped BC.

Outputs out/bc_failure_modes.png (stacked bars) + a per-variant table.
Run:  python scripts/bc_failure_modes.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, OUT_DIR
from residual_lift.data import LiftPHDataset, load_obs_stats, read_mask
from residual_lift.section1_bc import BCPolicy
from residual_lift.section4_eval import make_env, flatten_obs_dict

EPOCH_VARIANTS = [2, 5, 20, 80]     # underfit -> ~best -> overfit -> heavy overfit
N_ROLLOUTS = 50
SEED = 42
MAX_STEPS = 400
REACH_DIST, GRASP_OPEN, LIFT_RISE = 0.030, 0.045, 0.040

CATS = ["success", "fail:never_reached", "fail:reached_no_grasp",
        "fail:grasp_no_lift", "fail:lifted_then_dropped"]
COLORS = {"success": "tab:green", "fail:never_reached": "tab:red",
          "fail:reached_no_grasp": "tab:orange", "fail:grasp_no_lift": "gold",
          "fail:lifted_then_dropped": "tab:purple"}


def train_bc_fixed(epochs, train_ds, valid_ds, obs_mean, obs_std):
    """Train BC for a FIXED number of epochs (no early stop). Returns (policy, val_mse)."""
    torch.manual_seed(42); np.random.seed(42)
    pol = BCPolicy(obs_mean, obs_std).to(DEVICE)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    tl = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    vl = DataLoader(valid_ds, batch_size=256, shuffle=False)
    for _ in range(epochs):
        pol.train()
        for b in tl:
            o, a = b["obs"].to(DEVICE), b["action"].to(DEVICE)
            loss = F.mse_loss(pol(o), a)
            opt.zero_grad(); loss.backward(); opt.step()
    pol.eval()
    with torch.no_grad():
        tot = n = 0
        for b in vl:
            o, a = b["obs"].to(DEVICE), b["action"].to(DEVICE)
            tot += F.mse_loss(pol(o), a).item() * o.shape[0]; n += o.shape[0]
    return pol, tot / n


def classify(success, cube_z, dist, grip):
    if success:
        return "success"
    reached = dist.min() < REACH_DIST
    grasped = reached and grip[dist < REACH_DIST].min() < GRASP_OPEN
    lifted = (cube_z.max() - cube_z[0]) > LIFT_RISE
    if not reached:
        return "fail:never_reached"
    if not grasped:
        return "fail:reached_no_grasp"
    if not lifted:
        return "fail:grasp_no_lift"
    return "fail:lifted_then_dropped"


def eval_modes(policy):
    np.random.seed(SEED); torch.manual_seed(SEED)
    env = make_env()
    cats = []
    for _ in range(N_ROLLOUTS):
        obs = flatten_obs_dict(env.reset())
        cz, dd, gg, ok = [], [], [], False
        for t in range(MAX_STEPS):
            with torch.no_grad():
                a = policy.act(torch.from_numpy(obs).float().to(DEVICE)).cpu().numpy()
            od, _, done, _ = env.step(a)
            cz.append(float(od["cube_pos"][2]))
            dd.append(float(np.linalg.norm(od["gripper_to_cube_pos"])))
            gg.append(float(np.sum(np.abs(od["robot0_gripper_qpos"]))))
            obs = flatten_obs_dict(od)
            if env._check_success():
                ok = True; break
            if done:
                break
        cats.append(classify(ok, np.array(cz), np.array(dd), np.array(gg)))
    return Counter(cats)


def main():
    print(f"Device: {DEVICE}  |  {N_ROLLOUTS} rollouts/variant")
    train_ds = LiftPHDataset(demo_keys=read_mask("train"))
    valid_ds = LiftPHDataset(demo_keys=read_mask("valid"))
    obs_mean, obs_std = load_obs_stats(train_ds)

    results = {}
    for ep in EPOCH_VARIANTS:
        pol, vmse = train_bc_fixed(ep, train_ds, valid_ds, obs_mean, obs_std)
        counts = eval_modes(pol)
        sr = counts["success"] / N_ROLLOUTS
        results[ep] = (counts, vmse, sr)
        print(f"\nepochs {ep:3d}  val_mse {vmse:.5f}  success {sr:.2f}")
        for c in CATS:
            if counts[c]:
                print(f"    {c:<28} {counts[c]}")

    # stacked bar chart of outcome mix per variant
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [str(e) for e in EPOCH_VARIANTS]
    bottoms = np.zeros(len(EPOCH_VARIANTS))
    for c in CATS:
        vals = np.array([results[e][0][c] for e in EPOCH_VARIANTS])
        ax.bar(xs, vals, bottom=bottoms, label=c, color=COLORS[c])
        bottoms += vals
    for i, e in enumerate(EPOCH_VARIANTS):
        _, vmse, sr = results[e]
        ax.text(i, N_ROLLOUTS + 0.5, f"succ {sr:.0%}\nvmse {vmse:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("BC training epochs (no early stop)")
    ax.set_ylabel(f"rollouts (of {N_ROLLOUTS})")
    ax.set_ylim(0, N_ROLLOUTS + 6)
    ax.set_title("BC failure modes vs training duration "
                 "(underfit -> early-stop -> overfit)")
    ax.legend(fontsize=8, loc="lower left")
    path = os.path.join(OUT_DIR, "bc_failure_modes.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
