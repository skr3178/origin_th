"""Section 2 — Residual policy (TD3+BC).

Train a residual correction on top of the *frozen* BC. The executed action is

    a_executed(s) = clip(a_BC(s) + delta_theta(s), -1, +1)

delta_theta is a small MLP trained with an offline-RL objective (TD3+BC) on the
lift-ph dataset. The eight design decisions (defended in notes.md):

  1. Architecture     delta(s) only — state-conditioned, no a_BC input (a_BC is a
                      deterministic function of s, so it adds no information).
  2. Activation       tanh x delta_bound — smooth, hard-bounded residual.
  3. Bound magnitude  delta_bound = 0.05 (see notes; conservative on the gripper dim).
  4. Algorithm        TD3+BC: twin critics + delayed actor + target smoothing, with
                      a BC-anchor term keeping the executed action near demo actions.
  5. Reward           sparse terminal as given. NOTE: in this dataset done == reward
                      (both fire on the lift-success step), so y = r + g(1-done)Q'
                      correctly cuts the bootstrap at success.
  6. Clip delta in    YES. The target action must be the *executable* action
     target Q         (residual bounded, sum clipped to [-1,1]). Not clipping trains
                      Q on actions the policy can never take -> OOD overestimation,
                      delta saturates. (Isolated by the ablation flag below.)
  7. Target update    soft Polyak (tau = 0.005) for both critic and target actor.
  8. Step count       10k steps; watch q_mean (should stay bounded) and delta_mag
                      (should settle below delta_bound, not saturate).

Required diagnostics, printed every 500 steps: q_mean, delta_mag, critic_loss, actor_loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import OBS_DIM, ACT_DIM, DEVICE, RESIDUAL_CKPT
from .data import LiftPHDataset

# --- training hyperparameters (defend in notes) ---
# Decision #3: the scaffold's 0.05 is FAR too large for this precision task — at 0.05
# the residual saturates its budget and destroys BC's grasp (success 0.0-0.15). A bound
# sweep (see notes.md) shows degradation is monotonic in the bound:
#   0.05 -> 0.0-0.15 | 0.02 -> 0.75 | 0.01 -> 0.85 (=BC) | 0.005 -> 0.90 (best).
# We ship 0.005: tiny corrections that stay close to BC (delta_mag ~0.0048).
DELTA_BOUND = 0.005     # residual magnitude cap per action dim
TRAIN_STEPS = 10000
BATCH_SIZE  = 256
GAMMA       = 0.99
TAU         = 0.005     # Polyak coefficient
ACTOR_LR    = 1e-4
CRITIC_LR   = 3e-4
POLICY_DELAY = 2        # actor + target updates every N critic steps
ALPHA       = 2.5       # TD3+BC Q-vs-BC trade-off (Fujimoto & Gu, 2021)
# Target-smoothing noise, scaled to the residual's own range (not the full action range).
POLICY_NOISE = 0.2 * DELTA_BOUND
NOISE_CLIP   = 0.5 * DELTA_BOUND


class ResidualPolicy(nn.Module):
    """Frozen BC + small bounded correction head delta(s).

    condition_on_bc (decision-#1 ablation): if True, the head is delta(s, a_BC(s)) —
    the frozen BC action is concatenated to the (normalized) obs as extra input.
    Default False = delta(s). Since a_BC(s) is a deterministic function of s, the MLP
    can already recover it internally, so this should add no information (we verify).
    """
    def __init__(self, bc_policy, act_dim=ACT_DIM, delta_bound=DELTA_BOUND, hidden=128,
                 condition_on_bc=False, activation="tanh"):
        super().__init__()
        self.bc = bc_policy           # MUST stay frozen
        self.delta_bound = delta_bound
        self.condition_on_bc = condition_on_bc
        # decision-#2 ablation: how δ is bounded.
        #   "tanh"     -> bound * tanh(f)      smooth, hard per-dim bound (shipped)
        #   "clip"     -> clamp(f, ±bound)     hard bound, ZERO gradient when saturated
        #   "softsign" -> bound * softsign(f)  smooth, softer/slower saturation
        assert activation in ("tanh", "clip", "softsign"), activation
        self.activation = activation
        in_dim = OBS_DIM + (act_dim if condition_on_bc else 0)
        self.delta_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        # Normalize obs the same way BC does (shared, fixed stats).
        self.register_buffer("obs_mean", bc_policy.obs_mean.clone())
        self.register_buffer("obs_std",  bc_policy.obs_std.clone())

    def raw_delta(self, obs):
        """The bounded residual delta(s) in [-delta_bound, +delta_bound]."""
        z = (obs - self.obs_mean) / self.obs_std
        if self.condition_on_bc:
            z = torch.cat([z, self.bc(obs).detach()], dim=-1)  # a_BC(s) already in [-1,1]
        f = self.delta_net(z)
        if self.activation == "clip":
            return torch.clamp(f, -self.delta_bound, self.delta_bound)
        if self.activation == "softsign":
            return self.delta_bound * F.softsign(f)
        return self.delta_bound * torch.tanh(f)

    def forward(self, obs):
        """Executed action: clip(a_BC(s) + delta(s), -1, +1). a_BC is frozen/detached."""
        a_bc = self.bc(obs).detach()
        return torch.clamp(a_bc + self.raw_delta(obs), -1.0, 1.0)

    @torch.no_grad()
    def act(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.forward(obs).squeeze(0)


class QCritic(nn.Module):
    """Twin Q(s, a). Returns (q1, q2); q1_only(...) for the actor objective."""
    def __init__(self, obs_mean, obs_std, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        def mlp():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),            nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1, self.q2 = mlp(), mlp()
        self.register_buffer("obs_mean", obs_mean.clone())
        self.register_buffer("obs_std",  obs_std.clone())

    def _in(self, obs, act):
        return torch.cat([(obs - self.obs_mean) / self.obs_std, act], dim=-1)

    def forward(self, obs, act):
        x = self._in(obs, act)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, act):
        return self.q1(self._in(obs, act))


def train_residual(bc_policy, steps=TRAIN_STEPS, ablate_clip_in_target=False, log_every=500,
                   alpha=ALPHA, delta_bound=DELTA_BOUND, save=True, seed=42,
                   reward_shaping=False, shape_coef=1.0, condition_on_bc=False,
                   activation="tanh"):
    """Train the residual with TD3+BC on top of a frozen `bc_policy`.

    alpha controls the TD3+BC Q-vs-BC trade-off: larger -> trust the offline Q
    more (more aggressive residual); smaller -> stay closer to the demo/BC action.
    ablate_clip_in_target=True disables decision #6 (clipping the target action to
    the executable set) to demonstrate the resulting Q overestimation.
    seed sets the training RNG (default 42, the shipped run); vary it for multi-seed
    robustness checks.
    Returns (residual, history) and saves to RESIDUAL_CKPT (unless ablating or save=False).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    obs_mean, obs_std = bc_policy.obs_mean, bc_policy.obs_std
    bound = delta_bound

    residual        = ResidualPolicy(bc_policy, delta_bound=bound, condition_on_bc=condition_on_bc,
                                     activation=activation).to(DEVICE)
    residual_target = ResidualPolicy(bc_policy, delta_bound=bound, condition_on_bc=condition_on_bc,
                                     activation=activation).to(DEVICE)
    residual_target.load_state_dict(residual.state_dict())
    q_critic        = QCritic(obs_mean, obs_std).to(DEVICE)
    target_q_critic = QCritic(obs_mean, obs_std).to(DEVICE)
    target_q_critic.load_state_dict(q_critic.state_dict())

    actor_opt  = torch.optim.Adam(residual.delta_net.parameters(), lr=ACTOR_LR)
    critic_opt = torch.optim.Adam(q_critic.parameters(),           lr=CRITIC_LR)

    # GPU-resident replay: the whole dataset (~0.7 MB) lives on the GPU once, and
    # batches are drawn with a single randint index — no DataLoader, no per-step
    # CPU->GPU copy. Removes the host-side batch overhead (~1.6 ms/step here).
    _ds = LiftPHDataset()
    obs_all      = _ds.obs.to(DEVICE)
    action_all   = _ds.actions.to(DEVICE)
    reward_all   = _ds.rewards.to(DEVICE).unsqueeze(-1)
    next_obs_all = _ds.next_obs.to(DEVICE)
    done_all     = _ds.dones.to(DEVICE).unsqueeze(-1)
    n_trans = obs_all.shape[0]

    # Decision #5 ablation: dense reward shaping. The brief's example is `-|cube - eef|`;
    # `gripper_to_cube_pos` (obs dims 7:10, part of the `object` key) IS that vector, so the
    # shaped reward adds a dense distance bonus `shape_coef * (-||gripper_to_cube_pos(s')||)`
    # at every step on top of the sparse terminal reward. This is non-potential-based shaping,
    # so it deliberately changes the optimal policy (the point of the ablation). Default OFF.
    if reward_shaping:
        dist_next = next_obs_all[:, 7:10].norm(dim=-1, keepdim=True)   # ||gripper->cube|| at s'
        reward_all = reward_all + shape_coef * (-dist_next)
        print(f"[reward_shaping] coef={shape_coef}  mean dense term "
              f"{(shape_coef * -dist_next).mean().item():+.4f}  (sparse reward kept)")

    history = {k: [] for k in ["step", "q_mean", "delta_mag", "critic_loss", "actor_loss"]}
    last_actor_loss = float("nan")
    clip_in_target = not ablate_clip_in_target
    tag = "ABLATION(no-clip)" if ablate_clip_in_target else "TD3+BC"

    for step in tqdm(range(steps), desc=f"residual {tag}"):
        idx      = torch.randint(0, n_trans, (BATCH_SIZE,), device=DEVICE)
        obs      = obs_all[idx]
        action   = action_all[idx]                    # demo action a_demo
        reward   = reward_all[idx]
        next_obs = next_obs_all[idx]
        done     = done_all[idx]

        # --- Critic update ---
        with torch.no_grad():
            a_bc_next  = bc_policy(next_obs)
            noise      = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            delta_next = residual_target.raw_delta(next_obs) + noise
            if clip_in_target:  # decision #6 (correct)
                delta_next = delta_next.clamp(-bound, bound)
                a_next = torch.clamp(a_bc_next + delta_next, -1.0, 1.0)
            else:               # ablation: target action may be unrealizable
                a_next = a_bc_next + delta_next
            q1_t, q2_t = target_q_critic(next_obs, a_next)
            q_t = torch.min(q1_t, q2_t)
            y = reward + GAMMA * (1.0 - done) * q_t

        q1, q2 = q_critic(obs, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

        # --- Actor update (delayed) ---
        if step % POLICY_DELAY == 0:
            a_exec = residual(obs)
            q1_pi  = q_critic.q1_only(obs, a_exec)
            lam    = alpha / q1_pi.abs().mean().detach()      # TD3+BC normalization
            bc_anchor = F.mse_loss(a_exec, action)            # stay near demo actions
            actor_loss = -lam * q1_pi.mean() + bc_anchor
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
            last_actor_loss = actor_loss.item()

            # --- Polyak target updates (decision #7) ---
            with torch.no_grad():
                for p, tp in zip(q_critic.parameters(), target_q_critic.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * p)
                for p, tp in zip(residual.delta_net.parameters(), residual_target.delta_net.parameters()):
                    tp.mul_(1 - TAU).add_(TAU * p)

        # --- Required diagnostics ---
        if step % log_every == 0:
            with torch.no_grad():
                q_mean    = torch.min(q1, q2).mean().item()
                delta_mag = residual.raw_delta(obs).abs().mean().item()
            history["step"].append(step)
            history["q_mean"].append(q_mean)
            history["delta_mag"].append(delta_mag)
            history["critic_loss"].append(float(critic_loss))
            history["actor_loss"].append(float(last_actor_loss))
            print(f"step {step:5d}  q_mean {q_mean:+.3f}  delta_mag {delta_mag:.4f}  "
                  f"critic_loss {float(critic_loss):.4f}  actor_loss {last_actor_loss:.4f}")

    residual.eval()
    if save and not ablate_clip_in_target and not reward_shaping:
        torch.save(residual.state_dict(), RESIDUAL_CKPT)
        print(f"Residual trained. Saved to {RESIDUAL_CKPT}")
    return residual, history


def load_residual(bc_policy, path=RESIDUAL_CKPT):
    """Reconstruct a ResidualPolicy from a checkpoint (used by Section 4)."""
    residual = ResidualPolicy(bc_policy, delta_bound=DELTA_BOUND).to(DEVICE)
    residual.load_state_dict(torch.load(path, map_location=DEVICE))
    residual.eval()
    return residual
