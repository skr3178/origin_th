"""Section 1 — BC training  [TODO].

Train a BC policy on the lift-ph dataset. This becomes your *frozen backbone* —
Section 2's residual sits on top of it.

A suggested 3-layer 256-hidden MLP is given. You may modify it (depth, width,
activation, normalization scheme); defend any change in your writeup.

Decisions to defend:
  1. Architecture — keep the suggested 3-layer 256-hidden MLP, or change it?
  2. Loss function — MSE on continuous actions, or something else?
  3. Training duration — how many epochs? (imitation training overfits; longer
     is not always better.)
  4. Stopping criterion — fixed epochs, loss plateau, validation split, or
     rollout success rate?

Aim for ~75-90% rollout success (the final eval checks this). Outside that range
usually means a training-loop bug worth investigating.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import OBS_DIM, ACT_DIM, DEVICE, BC_CKPT
from .data import LiftPHDataset, load_obs_stats, read_mask

# Hyperparameters (defend in writeup).
BC_EPOCHS   = 60      # upper bound; early stopping ends well before this
BC_LR       = 1e-3
BC_BS       = 256
BC_PATIENCE = 8       # stop if val MSE hasn't improved for this many epochs


class BCPolicy(nn.Module):
    """SUGGESTED 3-layer MLP architecture.

    - hidden=256, two ReLU hidden layers
    - tanh output (action space is [-1, +1] per dim)
    - observation normalization via registered buffers (mean/std from data)

    You may modify this. Defend any change in your writeup.
    """
    def __init__(self, obs_mean, obs_std, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.register_buffer("obs_mean", obs_mean.clone())
        self.register_buffer("obs_std",  obs_std.clone())

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.forward(obs).squeeze(0)


@torch.no_grad()
def _epoch_mse(policy, loader):
    """Mean MSE over a loader (no grad). Used for the validation curve."""
    policy.eval()
    tot, n = 0.0, 0
    for batch in loader:
        obs = batch["obs"].to(DEVICE)
        act = batch["action"].to(DEVICE)
        bs  = obs.shape[0]
        tot += F.mse_loss(policy(obs), act, reduction="mean").item() * bs
        n   += bs
    return tot / n


def train_bc():
    """Train BC, freeze it, save to BC_CKPT, and return the frozen policy.

    Decisions (defend in writeup):
      (1) architecture — keep the suggested 3-layer/256/ReLU/tanh MLP. tanh matches
          the [-1,1] action space; 73k params is ample for a 19->7 low-dim map.
      (2) loss — MSE on continuous actions (Gaussian-noise MLE; standard for BC).
      (3) duration — up to BC_EPOCHS, but governed by (4).
      (4) stopping — early stopping on held-out validation MSE (the file's own
          train/valid demo split, so no train/val leakage), patience=BC_PATIENCE.
          We keep the best-val weights, not the last epoch's.
    """
    torch.manual_seed(42); np.random.seed(42)

    # Demo-level train/valid split from the dataset's own mask (180 / 20 demos).
    train_keys = read_mask("train")
    valid_keys = read_mask("valid")
    train_ds = LiftPHDataset(demo_keys=train_keys)
    valid_ds = LiftPHDataset(demo_keys=valid_keys)
    print(f"BC split: {len(train_ds)} train / {len(valid_ds)} valid transitions "
          f"({len(train_keys)}/{len(valid_keys)} demos)")

    # Normalize using TRAIN-split stats only (no leakage from valid).
    obs_mean, obs_std = load_obs_stats(train_ds)
    bc_policy = BCPolicy(obs_mean, obs_std).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BC_BS, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BC_BS, shuffle=False)
    opt = torch.optim.Adam(bc_policy.parameters(), lr=BC_LR)

    best_val, best_state, best_epoch, since_improve = float("inf"), None, -1, 0
    for epoch in range(BC_EPOCHS):
        bc_policy.train()
        run, n = 0.0, 0
        for batch in train_loader:
            obs = batch["obs"].to(DEVICE)
            act = batch["action"].to(DEVICE)
            loss = F.mse_loss(bc_policy(obs), act)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * obs.shape[0]; n += obs.shape[0]
        train_mse = run / n
        val_mse = _epoch_mse(bc_policy, valid_loader)

        improved = val_mse < best_val - 1e-6
        if improved:
            best_val, best_epoch, since_improve = val_mse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in bc_policy.state_dict().items()}
        else:
            since_improve += 1

        if epoch % 5 == 0 or improved or since_improve >= BC_PATIENCE:
            print(f"epoch {epoch:3d}  train_mse {train_mse:.5f}  val_mse {val_mse:.5f}"
                  f"{'  *best' if improved else ''}")
        if since_improve >= BC_PATIENCE:
            print(f"early stop: no val improvement for {BC_PATIENCE} epochs.")
            break

    # Restore best-val weights, freeze, save.
    bc_policy.load_state_dict(best_state)
    bc_policy.eval()
    for p in bc_policy.parameters():
        p.requires_grad_(False)
    torch.save(bc_policy.state_dict(), BC_CKPT)
    print(f"BC frozen at best epoch {best_epoch} (val_mse {best_val:.5f}). Saved to {BC_CKPT}")
    return bc_policy


def load_bc(path=BC_CKPT):
    """Reconstruct a frozen BCPolicy from a checkpoint (used by Sections 2 & 4)."""
    obs_mean, obs_std = load_obs_stats()
    bc_policy = BCPolicy(obs_mean, obs_std).to(DEVICE)
    bc_policy.load_state_dict(torch.load(path, map_location=DEVICE))
    bc_policy.eval()
    for p in bc_policy.parameters():
        p.requires_grad_(False)
    return bc_policy
