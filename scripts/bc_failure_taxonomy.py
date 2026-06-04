#!/usr/bin/env python
"""Task 2 — BC failure taxonomy over many rollouts.

Runs a large rollout sweep (150 each) on (i) the shipped early-stopped BC and
(ii) a deliberately over-trained (lesser-quality) BC, to surface a bigger failure
sample, then categorizes every failure by manipulation phase:

  never_reached | reached_no_grasp | grasp_no_lift | lifted_then_dropped

Produces out/bc_failure_taxonomy.png (4 panels: failure-mode bars for each model,
a failure phase-space scatter, and failure timing) and archives to
out/runs/bc_failure_taxonomy/.  Run:  python scripts/bc_failure_taxonomy.py
"""
import os
import sys
import shutil
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from residual_lift.config import DEVICE, OUT_DIR
from residual_lift.data import LiftPHDataset, load_obs_stats, read_mask
from residual_lift.section1_bc import BCPolicy, load_bc
from residual_lift.section4_eval import make_env, flatten_obs_dict

N_ROLLOUTS, SEED, MAX_STEPS = 150, 42, 400
REACH_DIST, GRASP_OPEN, LIFT_RISE = 0.030, 0.045, 0.040
FAIL_CATS = ["fail:never_reached", "fail:reached_no_grasp",
             "fail:grasp_no_lift", "fail:lifted_then_dropped"]
CCOL = {"fail:never_reached": "tab:red", "fail:reached_no_grasp": "tab:orange",
        "fail:grasp_no_lift": "gold", "fail:lifted_then_dropped": "tab:purple"}


def train_overfit_bc(epochs=80):
    tr = LiftPHDataset(demo_keys=read_mask("train"))
    obs_mean, obs_std = load_obs_stats(tr)
    torch.manual_seed(42); np.random.seed(42)
    pol = BCPolicy(obs_mean, obs_std).to(DEVICE)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    tl = DataLoader(tr, batch_size=256, shuffle=True, drop_last=True)
    for _ in range(epochs):
        pol.train()
        for b in tl:
            loss = F.mse_loss(pol(b["obs"].to(DEVICE)), b["action"].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    pol.eval()
    return pol


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


def sweep(policy):
    """Return list of per-rollout dicts (cat, min_dist, max_lift, end_step)."""
    np.random.seed(SEED); torch.manual_seed(SEED)
    env = make_env()
    recs = []
    for _ in range(N_ROLLOUTS):
        obs = flatten_obs_dict(env.reset())
        cz, dd, gg, ok, t = [], [], [], False, 0
        for t in range(MAX_STEPS):
            with torch.no_grad():
                a = policy.act(torch.from_numpy(obs).float().to(DEVICE)).cpu().numpy()
            od, _, done, _ = env.step(a)
            cz.append(float(od["cube_pos"][2]))
            dd.append(float(np.linalg.norm(od["gripper_to_cube_pos"])))
            gg.append(float(np.sum(np.abs(od["robot0_gripper_qpos"]))))
            obs = flatten_obs_dict(od)
            if env._check_success(): ok = True; break
            if done: break
        cz, dd, gg = np.array(cz), np.array(dd), np.array(gg)
        recs.append(dict(cat=classify(ok, cz, dd, gg), min_dist=float(dd.min()),
                         max_lift=float(cz.max() - cz[0]), end=t + 1, ok=ok))
    return recs


def fail_bar(ax, recs, title):
    c = Counter(r["cat"] for r in recs)
    sr = sum(r["ok"] for r in recs) / len(recs)
    counts = [c[k] for k in FAIL_CATS]
    ax.bar([k.replace("fail:", "") for k in FAIL_CATS], counts,
           color=[CCOL[k] for k in FAIL_CATS])
    for i, v in enumerate(counts):
        if v: ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# failed rollouts (count)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # integer counts only
    ax.set_ylim(0, max(counts) + 1)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.set_title(f"{title}\nsuccess {sr:.2f}  |  {sum(counts)} failures / {len(recs)}")


def main():
    print(f"Device: {DEVICE} | {N_ROLLOUTS} rollouts/model")
    snap = os.path.join(OUT_DIR, "runs", "bc_failure_taxonomy")
    os.makedirs(snap, exist_ok=True)
    cache = os.path.join(snap, "recs.json")
    if os.path.exists(cache):                       # re-plot without re-sweeping
        print(f"loading cached rollouts: {cache}")
        d = json.load(open(cache)); rs, ro = d["shipped"], d["overfit"]
    else:
        shipped = load_bc()
        print("sweeping shipped BC..."); rs = sweep(shipped)
        overfit = train_overfit_bc(80)
        print("sweeping overfit BC..."); ro = sweep(overfit)
        json.dump({"shipped": rs, "overfit": ro}, open(cache, "w"))

    for name, recs in [("shipped", rs), ("overfit", ro)]:
        c = Counter(r["cat"] for r in recs)
        print(f"\n{name} BC: success {sum(r['ok'] for r in recs)}/{len(recs)}")
        for k in FAIL_CATS:
            if c[k]: print(f"   {k:<28} {c[k]}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fail_bar(ax[0, 0], rs, "(a) Shipped BC — failure modes")
    fail_bar(ax[0, 1], ro, "(b) Overfit BC (80 ep) — failure modes")

    # (c) phase-space: min gripper->cube distance vs max cube lift, per failure
    axc = ax[1, 0]
    for r in rs:
        if r["ok"]: continue
        axc.scatter(r["min_dist"], r["max_lift"], color=CCOL[r["cat"]], s=40, alpha=0.7)
    axc.axvline(REACH_DIST, ls="--", c="gray", lw=0.8)
    axc.text(REACH_DIST, axc.get_ylim()[1], " reach thresh", fontsize=7, va="top")
    axc.set_xlabel("min gripper→cube distance (m)"); axc.set_ylabel("max cube lift (m)")
    axc.set_title("(c) Shipped-BC failures in phase space")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=CCOL[k],
               label=k.replace("fail:", "")) for k in FAIL_CATS]
    axc.legend(handles=handles, fontsize=7)

    # (d) failure timing: end step (most failures time out at the horizon)
    axd = ax[1, 1]
    fend = [r["end"] for r in rs if not r["ok"]]
    axd.hist(fend, bins=20, color="tab:red", alpha=0.7)
    axd.axvline(MAX_STEPS, ls="--", c="gray", lw=0.8, label=f"horizon={MAX_STEPS}")
    axd.set_xlabel("episode end step"); axd.set_ylabel("# failed rollouts")
    axd.set_title("(d) When shipped-BC failures end (timeout vs early)"); axd.legend(fontsize=8)

    fig.suptitle(f"Task 2 — BC failure taxonomy ({N_ROLLOUTS} rollouts/model)")
    live = os.path.join(OUT_DIR, "bc_failure_taxonomy.png")
    fig.tight_layout(); fig.savefig(live, dpi=120); plt.close(fig)

    shutil.copy2(live, snap)
    print(f"\nSaved: {live}\nArchived: {snap}/")


if __name__ == "__main__":
    main()
