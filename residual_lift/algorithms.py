"""Alternative residual-learning algorithms (decision #4 cross-algorithm study).

Same residual framing as Section 2 — a frozen BC backbone plus a bounded
correction delta(s) (tanh x bound), trained offline on lift-ph — but with
*non-TD3+BC* objectives, so the comparison isolates the algorithm choice:

  - AWAC  (Advantage-Weighted Actor-Critic): the actor is advantage-weighted
          regression toward the demo action; collapses to BC when advantages
          vanish (which they do on all-expert data — the ideal behaviour here).
  - IQL   (Implicit Q-Learning): learns a value net V via expectile regression
          and bootstraps Q from V(s') — so it NEVER queries Q at policy/OOD
          actions, the most offline-safe choice on narrow expert data.

Both reuse ResidualPolicy + QCritic from section2_residual, so only the learning
rule differs. Both log the four required diagnostics every `log_every` steps:
q_mean, delta_mag, critic_loss, actor_loss.

References / provenance:
  - IQL  is shipped in robomimic (`robomimic/algo/iql.py`, `config/iql_config.py`).
         Our value loss (expectile regression), V-bootstrapped Q target, and
         advantage-weighted actor mirror that reference; we adopt its defaults
         (vf_quantile=0.7 here, robomimic ships 0.9; adv.beta). Paper: Kostrikov
         et al. 2021 (arXiv:2110.06169).
  - AWAC has no standalone robomimic algo, but its defining advantage-weighted
         regression actor IS robomimic's IQL policy-extraction step
         (`actor_loss = (-log_prob * exp(beta*adv)).mean()`); it differs only in
         the critic (standard twin-Q TD, no expectile V). Paper: Nair et al. 2020
         (arXiv:2006.09359). Full impls: CORL, d3rlpy, CleanRL.
Both are adapted to emit a *bounded residual* delta(s) rather than a full action.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import OBS_DIM, DEVICE
from .data import LiftPHDataset
from .section2_residual import (
    ResidualPolicy, QCritic,
    GAMMA, TAU, ACTOR_LR, CRITIC_LR, POLICY_DELAY, BATCH_SIZE,
)


class VNet(nn.Module):
    """State-value V(s) for IQL. Same obs-normalization as the critic."""
    def __init__(self, obs_mean, obs_std, obs_dim=OBS_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("obs_mean", obs_mean.clone())
        self.register_buffer("obs_std",  obs_std.clone())

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)


def _gpu_replay(seed):
    """Load the dataset onto the GPU once (matches section2's fast path)."""
    ds = LiftPHDataset()
    return (ds.obs.to(DEVICE), ds.actions.to(DEVICE),
            ds.rewards.to(DEVICE).unsqueeze(-1), ds.next_obs.to(DEVICE),
            ds.dones.to(DEVICE).unsqueeze(-1), ds.obs.shape[0])


def _hist():
    return {k: [] for k in ["step", "q_mean", "delta_mag", "critic_loss", "actor_loss"]}


def _log(history, step, q_mean, delta_mag, critic_loss, actor_loss, tag):
    history["step"].append(step)
    history["q_mean"].append(float(q_mean))
    history["delta_mag"].append(float(delta_mag))
    history["critic_loss"].append(float(critic_loss))
    history["actor_loss"].append(float(actor_loss))
    print(f"[{tag}] step {step:5d}  q_mean {q_mean:+.3f}  delta_mag {delta_mag:.4f}  "
          f"critic_loss {float(critic_loss):.4f}  actor_loss {float(actor_loss):.4f}")


def train_residual_awac(bc_policy, steps=10000, delta_bound=0.005, seed=42,
                        beta=3.0, weight_clip=20.0, log_every=500):
    """AWAC residual. Critic = standard twin-Q TD (same as TD3+BC's critic); the
    *actor* is advantage-weighted regression toward the demo action:
        w = exp(beta * (Q(s,a_demo) - Q(s, a_exec))),  loss = w * ||a_exec - a_demo||^2.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    om, os_ = bc_policy.obs_mean, bc_policy.obs_std
    res  = ResidualPolicy(bc_policy, delta_bound=delta_bound).to(DEVICE)
    rest = ResidualPolicy(bc_policy, delta_bound=delta_bound).to(DEVICE)
    rest.load_state_dict(res.state_dict())
    q  = QCritic(om, os_).to(DEVICE)
    qt = QCritic(om, os_).to(DEVICE); qt.load_state_dict(q.state_dict())
    a_opt = torch.optim.Adam(res.delta_net.parameters(), lr=ACTOR_LR)
    c_opt = torch.optim.Adam(q.parameters(), lr=CRITIC_LR)

    obs_all, act_all, rew_all, nobs_all, done_all, n = _gpu_replay(seed)
    pnoise, nclip = 0.2 * delta_bound, 0.5 * delta_bound
    history, last_actor = _hist(), float("nan")

    for step in tqdm(range(steps), desc=f"AWAC seed{seed}"):
        idx = torch.randint(0, n, (BATCH_SIZE,), device=DEVICE)
        obs, action, reward, nobs, done = (obs_all[idx], act_all[idx], rew_all[idx],
                                           nobs_all[idx], done_all[idx])
        # --- critic: twin-Q TD with bounded policy action in the target ---
        with torch.no_grad():
            a_bc_n = bc_policy(nobs)
            noise  = (torch.randn_like(action) * pnoise).clamp(-nclip, nclip)
            dn     = (rest.raw_delta(nobs) + noise).clamp(-delta_bound, delta_bound)
            a_next = torch.clamp(a_bc_n + dn, -1.0, 1.0)
            q1t, q2t = qt(nobs, a_next)
            y = reward + GAMMA * (1.0 - done) * torch.min(q1t, q2t)
        q1, q2 = q(obs, action)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        c_opt.zero_grad(); c_loss.backward(); c_opt.step()

        # --- actor: advantage-weighted regression toward a_demo ---
        if step % POLICY_DELAY == 0:
            with torch.no_grad():
                a_cur = res(obs)
                q_demo = torch.min(*q(obs, action))
                q_pi   = torch.min(*q(obs, a_cur))
                w = torch.exp(beta * (q_demo - q_pi)).clamp(max=weight_clip)
            a_exec = res(obs)
            per = ((a_exec - action) ** 2).mean(dim=-1, keepdim=True)
            a_loss = (w * per).mean()
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()
            last_actor = a_loss.item()
            with torch.no_grad():
                for p, tp in zip(q.parameters(), qt.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * p)
                for p, tp in zip(res.delta_net.parameters(), rest.delta_net.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * p)

        if step % log_every == 0:
            with torch.no_grad():
                _log(history, step, torch.min(q1, q2).mean().item(),
                     res.raw_delta(obs).abs().mean().item(), c_loss, last_actor, "AWAC")
    res.eval()
    return res, history


def train_residual_iql(bc_policy, steps=10000, delta_bound=0.005, seed=42,
                       expectile=0.7, beta=3.0, weight_clip=100.0, log_every=500):
    """IQL residual. Learns V via expectile regression toward Q_target(s,a_demo),
    bootstraps Q from V(s') (no OOD action query), and extracts the residual via
    advantage-weighted regression with A = Q_target(s,a_demo) - V(s).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    om, os_ = bc_policy.obs_mean, bc_policy.obs_std
    res = ResidualPolicy(bc_policy, delta_bound=delta_bound).to(DEVICE)
    q   = QCritic(om, os_).to(DEVICE)
    qt  = QCritic(om, os_).to(DEVICE); qt.load_state_dict(q.state_dict())
    v   = VNet(om, os_).to(DEVICE)
    a_opt = torch.optim.Adam(res.delta_net.parameters(), lr=ACTOR_LR)
    c_opt = torch.optim.Adam(q.parameters(), lr=CRITIC_LR)
    v_opt = torch.optim.Adam(v.parameters(), lr=CRITIC_LR)

    obs_all, act_all, rew_all, nobs_all, done_all, n = _gpu_replay(seed)
    history, last_actor = _hist(), float("nan")

    for step in tqdm(range(steps), desc=f"IQL seed{seed}"):
        idx = torch.randint(0, n, (BATCH_SIZE,), device=DEVICE)
        obs, action, reward, nobs, done = (obs_all[idx], act_all[idx], rew_all[idx],
                                           nobs_all[idx], done_all[idx])
        # --- V: expectile regression toward min target-Q on the DEMO action ---
        with torch.no_grad():
            q_target = torch.min(*qt(obs, action))
        v_val = v(obs)
        diff = q_target - v_val
        w_exp = torch.where(diff > 0, expectile, 1.0 - expectile)
        v_loss = (w_exp * diff.pow(2)).mean()
        v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        # --- Q: TD toward r + gamma (1-done) V(s')  — no OOD action queried ---
        with torch.no_grad():
            y = reward + GAMMA * (1.0 - done) * v(nobs)
        q1, q2 = q(obs, action)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        c_opt.zero_grad(); q_loss.backward(); c_opt.step()
        c_loss = q_loss + v_loss

        # --- actor: advantage-weighted regression toward a_demo ---
        if step % POLICY_DELAY == 0:
            with torch.no_grad():
                adv = torch.min(*qt(obs, action)) - v(obs)
                w = torch.exp(beta * adv).clamp(max=weight_clip)
            a_exec = res(obs)
            per = ((a_exec - action) ** 2).mean(dim=-1, keepdim=True)
            a_loss = (w * per).mean()
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()
            last_actor = a_loss.item()
            with torch.no_grad():
                for p, tp in zip(q.parameters(), qt.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * p)

        if step % log_every == 0:
            with torch.no_grad():
                _log(history, step, torch.min(q1, q2).mean().item(),
                     res.raw_delta(obs).abs().mean().item(), c_loss, last_actor, "IQL")
    res.eval()
    return res, history
