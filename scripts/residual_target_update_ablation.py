#!/usr/bin/env python
"""Decision #7 ablation — soft Polyak vs hard periodic-copy target update.

Trains the residual identically (same seed, init, bound, all TD3+BC machinery)
except for HOW the target critic/actor are updated:

    soft   (shipped)  tp <- (1-tau) tp + tau p   every actor step  (tau=0.005)
    hard@C            tp <- p                     every C actor steps (periodic copy)

Soft (Polyak) tracking was introduced by DDPG (Lillicrap et al. 2016) precisely to
replace DQN's hard periodic copy (Mnih et al. 2015): a slowly-tracking target makes
the bootstrap target change smoothly, improving critic stability. Here we show the
hard copy makes q_mean / critic_loss jump at each copy, while soft stays smooth.

Saves out/residual_target_update_ablation.png + archives JSON to
out/runs/residual_target_update_ablation/.  Run: python scripts/residual_target_update_ablation.py
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

from residual_lift.config import DEVICE, BC_CKPT, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import (
    ResidualPolicy, QCritic, DELTA_BOUND, GAMMA, TAU, ACTOR_LR, CRITIC_LR,
    POLICY_DELAY, ALPHA, POLICY_NOISE, NOISE_CLIP, BATCH_SIZE)
from residual_lift.section4_eval import evaluate
from residual_lift.data import LiftPHDataset

STEPS    = 10000
LOG_EVERY = 100          # finer than the usual 500, to resolve hard-copy jumps
# Soft tau=0.005 has an EMA time-constant ~1/tau = 200 target updates. A "matched"
# hard copy fires every ~200 actor steps; a slow copy every 1000 exaggerates jumps.
HARD_PERIODS = [250, 1000]
N_ROLLOUTS = 30


def _save_rng():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state())


def _restore_rng(s):
    torch.set_rng_state(s[0]); torch.cuda.set_rng_state_all(s[1]); np.random.set_state(s[2])


def train(bc, mode, hard_period=None, seed=42):
    """Train the residual with a given target-update mode. mode in {'soft','hard'}.
    Returns (residual, history). Identical to section2.train_residual except the
    target update rule (and GPU-resident replay for speed)."""
    torch.manual_seed(seed); np.random.seed(seed)
    om, os_ = bc.obs_mean, bc.obs_std

    res   = ResidualPolicy(bc).to(DEVICE)
    res_t = ResidualPolicy(bc).to(DEVICE); res_t.load_state_dict(res.state_dict())
    q     = QCritic(om, os_).to(DEVICE)
    qt    = QCritic(om, os_).to(DEVICE); qt.load_state_dict(q.state_dict())
    aopt  = torch.optim.Adam(res.delta_net.parameters(), lr=ACTOR_LR)
    copt  = torch.optim.Adam(q.parameters(), lr=CRITIC_LR)

    ds = LiftPHDataset()
    obs_all  = ds.obs.to(DEVICE);      act_all  = ds.actions.to(DEVICE)
    rew_all  = ds.rewards.to(DEVICE).unsqueeze(-1)
    nobs_all = ds.next_obs.to(DEVICE); done_all = ds.dones.to(DEVICE).unsqueeze(-1)
    n = obs_all.shape[0]

    hist = {k: [] for k in ["step", "q_mean", "delta_mag", "critic_loss", "actor_loss"]}
    last_actor = float("nan")
    n_actor_updates = 0
    tag = "soft" if mode == "soft" else f"hard@{hard_period}"

    for step in range(STEPS + 1):
        idx  = torch.randint(0, n, (BATCH_SIZE,), device=DEVICE)
        obs, act = obs_all[idx], act_all[idx]
        rew, nobs, done = rew_all[idx], nobs_all[idx], done_all[idx]

        with torch.no_grad():
            abc   = bc(nobs)
            noise = (torch.randn_like(act) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            dn    = (res_t.raw_delta(nobs) + noise).clamp(-DELTA_BOUND, DELTA_BOUND)
            anext = torch.clamp(abc + dn, -1, 1)
            q1t, q2t = qt(nobs, anext)
            y = rew + GAMMA * (1 - done) * torch.min(q1t, q2t)

        q1, q2 = q(obs, act)
        closs = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        copt.zero_grad(); closs.backward(); copt.step()

        if step % POLICY_DELAY == 0:
            ae = res(obs); q1pi = q.q1_only(obs, ae)
            lam = ALPHA / q1pi.abs().mean().detach()
            aloss = -lam * q1pi.mean() + ((ae - act) ** 2).mean()
            aopt.zero_grad(); aloss.backward(); aopt.step(); last_actor = aloss.item()
            n_actor_updates += 1

            with torch.no_grad():
                if mode == "soft":
                    for p, tp in zip(q.parameters(), qt.parameters()):
                        tp.mul_(1 - TAU).add_(TAU * p)
                    for p, tp in zip(res.delta_net.parameters(), res_t.delta_net.parameters()):
                        tp.mul_(1 - TAU).add_(TAU * p)
                else:  # hard: copy every `hard_period` actor updates
                    if n_actor_updates % hard_period == 0:
                        qt.load_state_dict(q.state_dict())
                        res_t.delta_net.load_state_dict(res.delta_net.state_dict())

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                qm   = torch.min(q1, q2).mean().item()
                dmag = res.raw_delta(obs).abs().mean().item()
            hist["step"].append(step); hist["q_mean"].append(qm); hist["delta_mag"].append(dmag)
            hist["critic_loss"].append(float(closs)); hist["actor_loss"].append(last_actor)

    res.eval()
    return res, hist, tag


def jumpiness(series):
    """Mean abs step-to-step change — a scalar 'how jumpy' metric."""
    a = np.asarray(series, dtype=np.float64)
    return float(np.abs(np.diff(a)).mean())


def main():
    print(f"Device: {DEVICE}")
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")
    bc = load_bc()

    runs = {}
    print("\n=== soft Polyak (shipped, tau=0.005) ===")
    res_soft, h_soft, _ = train(bc, "soft")
    runs["soft"] = h_soft
    for C in HARD_PERIODS:
        print(f"\n=== hard copy every {C} actor-updates ===")
        _, h, _ = train(bc, "hard", hard_period=C)
        runs[f"hard@{C}"] = h

    # Behavioral check: rollout success for soft vs the matched hard copy.
    print("\n=== rollout success (30 rollouts, seed 42) ===")
    rng = _save_rng()
    sr_soft = evaluate(res_soft, n_rollouts=N_ROLLOUTS, seed=42, label="soft")["success_rate"]
    _restore_rng(rng)
    res_hard_matched, _, _ = train(bc, "hard", hard_period=HARD_PERIODS[0])
    rng = _save_rng()
    sr_hard = evaluate(res_hard_matched, n_rollouts=N_ROLLOUTS, seed=42, label=f"hard@{HARD_PERIODS[0]}")["success_rate"]
    _restore_rng(rng)

    # ---- stability metrics ----
    print("\n--- stability (mean |Δ| step-to-step; lower = smoother) ---")
    stab = {}
    for name, h in runs.items():
        stab[name] = {"q_mean_jump": jumpiness(h["q_mean"]),
                      "critic_loss_jump": jumpiness(h["critic_loss"])}
        print(f"{name:10s}  q_mean_jump {stab[name]['q_mean_jump']:.4f}  "
              f"critic_loss_jump {stab[name]['critic_loss_jump']:.4f}")
    print(f"\nrollout success:  soft {sr_soft:.3f}   hard@{HARD_PERIODS[0]} {sr_hard:.3f}")

    # ---- plot ----
    colors = {"soft": "tab:green", f"hard@{HARD_PERIODS[0]}": "tab:orange",
              f"hard@{HARD_PERIODS[1]}": "tab:red"}
    fig, (axq, axc) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, h in runs.items():
        axq.plot(h["step"], h["q_mean"], label=name, c=colors.get(name), lw=1.3)
        axc.plot(h["step"], h["critic_loss"], label=name, c=colors.get(name), lw=1.3)
    axq.set_title("q_mean — soft tracks smoothly; hard copies jump")
    axq.set_xlabel("step"); axq.grid(alpha=0.3); axq.legend()
    axc.set_title("critic_loss — spikes at each hard copy")
    axc.set_xlabel("step"); axc.set_yscale("log"); axc.grid(alpha=0.3); axc.legend()
    fig.suptitle(f"Decision #7 — target update: soft Polyak vs hard copy  "
                 f"(success soft {sr_soft:.2f} vs hard {sr_hard:.2f})")
    live = os.path.join(OUT_DIR, "residual_target_update_ablation.png")
    fig.tight_layout(); fig.savefig(live, dpi=120); plt.close(fig)

    snap = os.path.join(OUT_DIR, "runs", "residual_target_update_ablation")
    os.makedirs(snap, exist_ok=True)
    shutil.copy2(live, snap)
    json.dump({"runs": runs, "stability": stab,
               "success": {"soft": sr_soft, f"hard@{HARD_PERIODS[0]}": sr_hard}},
              open(os.path.join(snap, "results.json"), "w"), indent=2)
    print(f"\nSaved: {live}\nArchived: {snap}/")


if __name__ == "__main__":
    main()
