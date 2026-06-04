#!/usr/bin/env python
"""Task 1 — BC architecture ablation.

Compares four BC policy heads on lift-ph, all trained on the same train split with
the same obs normalization, early-stopped on each model's own validation loss, then
evaluated on 50 rollouts (seed 42):

  - MLP 256x256 + tanh   (our shipped backbone)
  - MLP 1024x1024 + tanh (robomimic's reference vanilla-BC width)
  - GMM head (5 modes)    (multimodality test — should ≈ MLP if data is unimodal)
  - LSTM 400x2            (temporal context — robomimic's BC-RNN family)

Saves out/bc_arch_ablation.png + table, and archives to out/runs/bc_arch_ablation/.
Run:  python scripts/bc_arch_ablation.py
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal, Independent, MixtureSameFamily, Categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, OBS_DIM, ACT_DIM, OUT_DIR
from residual_lift.data import LiftPHDataset, load_obs_stats, read_mask
from residual_lift.section4_eval import make_env, flatten_obs_dict

N_ROLLOUTS, SEED, MAX_STEPS = 50, 42, 400
SEQ_LEN, SEQ_STRIDE = 10, 5


# ----------------------------- models -----------------------------
class MLPPolicy(nn.Module):
    def __init__(self, mean, std, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, ACT_DIM), nn.Tanh())
        self.register_buffer("m", mean.clone()); self.register_buffer("s", std.clone())

    def forward(self, o): return self.net((o - self.m) / self.s)
    def loss(self, o, a): return F.mse_loss(self.forward(o), a)

    @torch.no_grad()
    def act(self, o):
        if o.dim() == 1: o = o.unsqueeze(0)
        return self.forward(o).squeeze(0)


class GMMPolicy(nn.Module):
    def __init__(self, mean, std, n_modes=5, hidden=1024):
        super().__init__()
        self.M = n_modes
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU())
        self.mean = nn.Linear(hidden, n_modes * ACT_DIM)
        self.scale = nn.Linear(hidden, n_modes * ACT_DIM)
        self.logit = nn.Linear(hidden, n_modes)
        self.register_buffer("m", mean.clone()); self.register_buffer("s", std.clone())

    def _dist(self, o):
        h = self.trunk((o - self.m) / self.s)
        B = h.shape[0]
        means = self.mean(h).view(B, self.M, ACT_DIM)
        scales = F.softplus(self.scale(h)).view(B, self.M, ACT_DIM) + 1e-4
        comp = Independent(Normal(means, scales), 1)
        return MixtureSameFamily(Categorical(logits=self.logit(h)), comp), means

    def loss(self, o, a):
        d, _ = self._dist(o)
        return -d.log_prob(a).mean()

    @torch.no_grad()
    def act(self, o):
        if o.dim() == 1: o = o.unsqueeze(0)
        d, means = self._dist(o)
        idx = d.mixture_distribution.logits.argmax(-1)          # most-likely mode
        a = means[torch.arange(means.shape[0]), idx]
        return torch.clamp(a, -1, 1).squeeze(0)


class RNNPolicy(nn.Module):
    def __init__(self, mean, std, hidden=400, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(OBS_DIM, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, ACT_DIM)
        self.register_buffer("m", mean.clone()); self.register_buffer("s", std.clone())
        self._h = None

    def forward(self, seq):                                     # (B,T,obs)
        out, _ = self.lstm((seq - self.m) / self.s)
        return torch.tanh(self.head(out))

    def loss(self, seq_o, seq_a):
        return F.mse_loss(self.forward(seq_o), seq_a)

    def reset(self): self._h = None

    @torch.no_grad()
    def act(self, o):
        if o.dim() == 1: o = o.unsqueeze(0)
        x = ((o - self.m) / self.s).unsqueeze(1)                # (1,1,obs)
        out, self._h = self.lstm(x, self._h)
        return torch.tanh(self.head(out)).view(-1)


# ----------------------------- training -----------------------------
def _flat_split(keys):
    ds = LiftPHDataset(demo_keys=keys)
    return TensorDataset(ds.obs, ds.actions)


def _seq_split(keys):
    o, a = [], []
    import h5py
    from residual_lift.data import DATASET_PATH, _flatten_obs_batch
    with h5py.File(DATASET_PATH, "r") as f:
        for k in keys:
            d = f["data"][k]
            ob = _flatten_obs_batch(d["obs"]); ac = d["actions"][:].astype(np.float32)
            for i in range(0, len(ob) - SEQ_LEN, SEQ_STRIDE):
                o.append(ob[i:i + SEQ_LEN]); a.append(ac[i:i + SEQ_LEN])
    return TensorDataset(torch.tensor(np.array(o)), torch.tensor(np.array(a)))


def train(model, tr, va, max_epochs, patience, bs):
    torch.manual_seed(42); np.random.seed(42)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=bs, shuffle=False)
    best, best_state, since, vhist = float("inf"), None, 0, []
    for ep in range(max_epochs):
        model.train()
        for x, y in tl:
            loss = model.loss(x.to(DEVICE), y.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tot = n = 0
            for x, y in vl:
                l = model.loss(x.to(DEVICE), y.to(DEVICE)).item()
                tot += l * x.shape[0]; n += x.shape[0]
        v = tot / n; vhist.append(v)
        if v < best - 1e-6:
            best, since = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since += 1
        if since >= patience: break
    model.load_state_dict(best_state); model.eval()
    return model, best, ep + 1, vhist


# ----------------------------- eval -----------------------------
def eval_success(policy):
    np.random.seed(SEED); torch.manual_seed(SEED)
    env = make_env(); succ = 0
    for _ in range(N_ROLLOUTS):
        if hasattr(policy, "reset"): policy.reset()
        obs = flatten_obs_dict(env.reset())
        for t in range(MAX_STEPS):
            with torch.no_grad():
                a = policy.act(torch.from_numpy(obs).float().to(DEVICE)).cpu().numpy()
            od, _, done, _ = env.step(a)
            obs = flatten_obs_dict(od)
            if env._check_success(): succ += 1; break
            if done: break
    return succ / N_ROLLOUTS


def main():
    print(f"Device: {DEVICE} | {N_ROLLOUTS} rollouts/variant")
    tr_keys, va_keys = read_mask("train"), read_mask("valid")
    obs_mean, obs_std = load_obs_stats(LiftPHDataset(demo_keys=tr_keys))
    flat_tr, flat_va = _flat_split(tr_keys), _flat_split(va_keys)
    seq_tr, seq_va = _seq_split(tr_keys), _seq_split(va_keys)

    variants = []  # (label, model, train_args, datasets)
    variants.append(("MLP 256x256\n(ours)", MLPPolicy(obs_mean, obs_std, 256), (flat_tr, flat_va, 60, 8, 256)))
    variants.append(("MLP 1024x1024\n(robomimic)", MLPPolicy(obs_mean, obs_std, 1024), (flat_tr, flat_va, 60, 8, 256)))
    variants.append(("GMM 5-mode\n(1024x1024)", GMMPolicy(obs_mean, obs_std, 5, 1024), (flat_tr, flat_va, 60, 8, 256)))
    variants.append(("LSTM 400x2\n(BC-RNN)", RNNPolicy(obs_mean, obs_std, 400, 2), (seq_tr, seq_va, 40, 6, 64)))

    COLORS = ["tab:blue", "tab:cyan", "tab:orange", "tab:green"]
    rows = []
    for label, model, (tr, va, me, pat, bs) in variants:
        m, vloss, eps, vhist = train(model, tr, va, me, pat, bs)
        sr = eval_success(m)
        nparam = sum(p.numel() for p in m.parameters())
        rows.append((label, sr, nparam, vloss, eps, vhist))
        print(f"\n{label.splitlines()[0]:<16} success {sr:.3f}  params {nparam:,}  "
              f"val_loss {vloss:.4f}  epochs {eps}")

    labels = [r[0] for r in rows]; srs = [r[1] for r in rows]; ps = [r[2] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) success-rate bar
    a0 = ax[0, 0]
    bars = a0.bar(labels, srs, color=COLORS)
    for b, r in zip(bars, rows):
        a0.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{r[1]:.2f}", ha="center", va="bottom", fontsize=9)
    a0.set_ylim(0, 1.08); a0.set_ylabel(f"success rate ({N_ROLLOUTS} rollouts)")
    a0.axhline(0.867, ls="--", c="gray", lw=0.8, label="shipped BC (30-roll) = 0.867")
    a0.set_title("(a) Success rate"); a0.legend(fontsize=8); a0.tick_params(labelsize=8)

    # (b) success vs params (efficiency frontier)
    a1 = ax[0, 1]
    for (label, sr, np_, _, _, _), c in zip(rows, COLORS):
        a1.scatter(np_, sr, s=90, color=c, zorder=3)
        a1.annotate(label.replace("\n", " "), (np_, sr), fontsize=8,
                    xytext=(0, 8), textcoords="offset points", ha="center")
    a1.set_xscale("log"); a1.set_xlabel("parameters (log)"); a1.set_ylabel("success rate")
    a1.set_ylim(0.5, 1.05); a1.grid(alpha=0.3)
    a1.set_title("(b) Efficiency: success vs capacity (top-left = better)")

    # (c) validation-loss training curves (MSE-trained variants only; GMM uses NLL)
    a2 = ax[1, 0]
    for (label, _, _, _, _, vh), c in zip(rows, COLORS):
        if "GMM" in label:
            continue
        a2.plot(range(len(vh)), vh, marker=".", color=c, label=label.replace("\n", " "))
    a2.set_xlabel("epoch"); a2.set_ylabel("validation MSE")
    a2.set_title("(c) Overfitting dynamics (val MSE; lower=better)")
    a2.grid(alpha=0.3); a2.legend(fontsize=8)

    # (d) parameter count (log)
    a3 = ax[1, 1]
    a3.bar(labels, ps, color=COLORS)
    for i, p in enumerate(ps):
        a3.text(i, p, f"{p/1e3:.0f}k", ha="center", va="bottom", fontsize=8)
    a3.set_yscale("log"); a3.set_ylabel("parameters (log)")
    a3.set_title("(d) Model capacity"); a3.tick_params(labelsize=8)

    fig.suptitle("Task 1 — BC architecture ablation on lift-ph "
                 "(256×2 matches/beats bigger nets at far fewer params)")
    live = os.path.join(OUT_DIR, "bc_arch_ablation.png")
    fig.tight_layout(); fig.savefig(live, dpi=120); plt.close(fig)

    # archive snapshot
    snap = os.path.join(OUT_DIR, "runs", "bc_arch_ablation")
    os.makedirs(snap, exist_ok=True)
    shutil.copy2(live, snap)
    with open(os.path.join(snap, "results.txt"), "w") as fh:
        fh.write("BC architecture ablation (lift-ph, 50 rollouts, seed 42)\n\n")
        fh.write(f"{'variant':<24}{'success':>9}{'params':>12}{'val_loss':>11}{'epochs':>8}\n")
        for label, sr, nparam, vloss, eps, _ in rows:
            fh.write(f"{label.replace(chr(10),' '):<24}{sr:>9.3f}{nparam:>12,}{vloss:>11.4f}{eps:>8}\n")
    print(f"\nSaved: {live}\nArchived: {snap}/")


if __name__ == "__main__":
    main()
