#!/usr/bin/env python
"""Decision #8 evidence — residual diagnostics & success vs training steps.

ONE continuous TD3+BC run to 40k steps (shipped bound 0.005), logging the four
required diagnostics every 500 steps and evaluating rollout success at
{5k,10k,20k,40k}. Shows the policy converges early (delta_mag flat by ~2k, success
flat) while q_mean keeps slowly drifting — i.e. training longer is pointless.

Saves out/residual_step_ablation.png + JSON, archives to out/runs/residual_step_ablation/.
Run:  python scripts/residual_step_ablation.py
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
from torch.utils.data import DataLoader

from residual_lift.config import DEVICE, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import (
    ResidualPolicy, QCritic, DELTA_BOUND, GAMMA, TAU, ACTOR_LR, CRITIC_LR,
    POLICY_DELAY, ALPHA, POLICY_NOISE, NOISE_CLIP, BATCH_SIZE)
from residual_lift.section4_eval import evaluate
from residual_lift.data import LiftPHDataset

MAX_STEPS = 40000
CKPTS = [5000, 10000, 20000, 40000]
BC_SR = 0.867


def _save_rng():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state())


def _restore_rng(s):
    torch.set_rng_state(s[0]); torch.cuda.set_rng_state_all(s[1]); np.random.set_state(s[2])


def main():
    print(f"Device: {DEVICE}")
    bc = load_bc()
    torch.manual_seed(42); np.random.seed(42)
    om, os_ = bc.obs_mean, bc.obs_std

    res = ResidualPolicy(bc).to(DEVICE)
    res_t = ResidualPolicy(bc).to(DEVICE); res_t.load_state_dict(res.state_dict())
    q = QCritic(om, os_).to(DEVICE)
    qt = QCritic(om, os_).to(DEVICE); qt.load_state_dict(q.state_dict())
    aopt = torch.optim.Adam(res.delta_net.parameters(), lr=ACTOR_LR)
    copt = torch.optim.Adam(q.parameters(), lr=CRITIC_LR)

    loader = DataLoader(LiftPHDataset(), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    it = iter(loader)
    def nb():
        nonlocal it
        try: return next(it)
        except StopIteration:
            it = iter(loader); return next(it)

    hist = {k: [] for k in ["step", "q_mean", "delta_mag", "critic_loss", "actor_loss"]}
    ckpt_succ = {}
    last_actor = float("nan")

    for step in range(MAX_STEPS + 1):
        b = nb()
        obs = b["obs"].to(DEVICE); act = b["action"].to(DEVICE)
        rew = b["reward"].to(DEVICE).unsqueeze(-1)
        nobs = b["next_obs"].to(DEVICE); done = b["done"].to(DEVICE).unsqueeze(-1)
        with torch.no_grad():
            abc = bc(nobs)
            noise = (torch.randn_like(act) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            dn = (res_t.raw_delta(nobs) + noise).clamp(-DELTA_BOUND, DELTA_BOUND)
            anext = torch.clamp(abc + dn, -1, 1)
            q1t, q2t = qt(nobs, anext); y = rew + GAMMA * (1 - done) * torch.min(q1t, q2t)
        q1, q2 = q(obs, act); closs = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        copt.zero_grad(); closs.backward(); copt.step()

        if step % POLICY_DELAY == 0:
            ae = res(obs); q1pi = q.q1_only(obs, ae)
            lam = ALPHA / q1pi.abs().mean().detach()
            aloss = -lam * q1pi.mean() + ((ae - act) ** 2).mean()
            aopt.zero_grad(); aloss.backward(); aopt.step(); last_actor = aloss.item()
            with torch.no_grad():
                for p, tp in zip(q.parameters(), qt.parameters()): tp.mul_(1 - TAU).add_(TAU * p)
                for p, tp in zip(res.delta_net.parameters(), res_t.delta_net.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * p)

        if step % 500 == 0:
            with torch.no_grad():
                qm = torch.min(q1, q2).mean().item()
                dmag = res.raw_delta(obs).abs().mean().item()
            hist["step"].append(step); hist["q_mean"].append(qm); hist["delta_mag"].append(dmag)
            hist["critic_loss"].append(float(closs)); hist["actor_loss"].append(last_actor)

        if step in CKPTS:
            rng = _save_rng(); res.eval()
            sr = evaluate(res, n_rollouts=30, seed=42, label=f"s{step}")["success_rate"]
            res.train(); _restore_rng(rng)
            ckpt_succ[step] = sr
            print(f"[ckpt {step:5d}]  success {sr:.3f}  q_mean {hist['q_mean'][-1]:+.3f}  "
                  f"delta_mag {hist['delta_mag'][-1]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a0 = ax[0]
    a0.plot(hist["step"], hist["q_mean"], color="tab:blue", label="q_mean")
    a0.set_xlabel("training step"); a0.set_ylabel("q_mean", color="tab:blue")
    a0.tick_params(axis="y", labelcolor="tab:blue"); a0.grid(alpha=0.3)
    a0b = a0.twinx()
    a0b.plot(hist["step"], hist["delta_mag"], color="tab:orange", label="delta_mag")
    a0b.axhline(DELTA_BOUND, ls="--", c="gray", lw=0.8)
    a0b.set_ylabel("mean |δ|", color="tab:orange"); a0b.tick_params(axis="y", labelcolor="tab:orange")
    for c in CKPTS: a0.axvline(c, ls=":", c="gray", lw=0.6)
    a0.set_title("(a) q_mean keeps drifting; delta_mag flat by ~2k")

    a1 = ax[1]; xs = sorted(ckpt_succ)
    a1.plot(xs, [ckpt_succ[s] for s in xs], "o-", color="tab:green")
    for s in xs:
        a1.annotate(f"{ckpt_succ[s]:.2f}", (s, ckpt_succ[s]),
                    textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center")
    a1.axhline(BC_SR, ls="--", c="gray", lw=0.9, label=f"BC = {BC_SR}")
    a1.set_xlabel("training step"); a1.set_ylabel("success (30 rollouts)"); a1.set_ylim(0, 1.05)
    a1.set_title("(b) Rollout success flat across step counts"); a1.grid(alpha=0.3); a1.legend()

    fig.suptitle("Decision #8 — training-step ablation: policy converges early, q_mean drifts")
    live = os.path.join(OUT_DIR, "residual_step_ablation.png")
    fig.tight_layout(); fig.savefig(live, dpi=120); plt.close(fig)

    snap = os.path.join(OUT_DIR, "runs", "residual_step_ablation")
    os.makedirs(snap, exist_ok=True)
    shutil.copy2(live, snap)
    json.dump({"hist": hist, "ckpt_succ": ckpt_succ},
              open(os.path.join(snap, "results.json"), "w"), indent=2)
    print(f"\nSaved: {live}\nArchived: {snap}/")


if __name__ == "__main__":
    main()
