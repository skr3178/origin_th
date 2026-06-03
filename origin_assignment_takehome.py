#!/usr/bin/env python
# coding: utf-8

# # Origin — AI Research Engineer Assignment (Take-Home)
# ### Residual Policy Learning on robomimic lift-ph
# 
# ---
# 
# **Before you start:** Runtime → Change runtime type → T4 GPU.
# 
# Plan for **4-5 hours of focused work** (hard cap 6). The notebook itself runs end-to-end in ~5-7 minutes on a T4; the rest of the time is thinking, sweeping, reading diagnostics, and writing up your decisions.
# 
# **See `docs/candidate_assignment.md` (sent with this notebook) for the full brief, evaluation rubric, FAQ, and submission instructions.** Read it before starting.
# 
# **Three placeholders to fill in:**
# 1. **Section 1** — BC training (architecture suggested; you write the training loop).
# 2. **Section 2** — Residual policy on top of frozen BC.
# 3. **Section 3** — Safety shield wrapping the executed action.
# 
# **Final cell is completed** — runs both policies, prints the comparison, saves a rollout video.
# 
# **Final deliverables:**
# 1. This notebook, run end-to-end with cell outputs preserved.
# 2. A 2-4 page writeup defending each design decision.
# 3. The rollout video saved by the final cell.
# 
# A 30-minute review call follows submission.

# ---
# ## Section 0 — Setup  `[PREFILLED]`
# 
# Install pinned versions, set the MuJoCo backend, download the dataset, and load it into RAM. Do not reorder the cells in this section.

# In[ ]:


# ~2 min on fresh Colab. Ignore dependency warnings.
# robosuite==1.4.1 is pinned to match the dataset's collection version. 1.4.1
# already uses DeepMind's modern `mujoco` binding (not the deprecated mujoco_py),
# so install is identical to 1.5.x. Pinning 1.5.x instead would force a dataset
# conversion step due to a sign flip in `gripper_to_cube_pos`.
get_ipython().system('pip install -q robomimic==0.3.0 robosuite==1.4.1 torch h5py imageio imageio-ffmpeg tqdm')
print("Installed.")


# In[ ]:


import os
# CRITICAL: these must be set before importing mujoco or robosuite.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch, time, glob, subprocess, h5py
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import imageio
import robomimic

assert torch.cuda.is_available(), "No GPU! Runtime > Change runtime type > T4 GPU."
print(f"GPU: {torch.cuda.get_device_name(0)}")
DEVICE = "cuda"


# In[ ]:


# Download lift-ph low-dim dataset (~22MB, ~30s on first run; skipped if cached).
# robomimic's download script always overwrites, so we gate it on file existence.
_GLOB = os.path.join(os.path.dirname(robomimic.__file__), "..",
                     "datasets", "lift", "ph", "low_dim*.hdf5")
if not glob.glob(_GLOB):
    print("Downloading lift-ph dataset...")
    subprocess.run(
        ["python", "-m", "robomimic.scripts.download_datasets",
         "--tasks", "lift", "--dataset_types", "ph", "--hdf5_types", "low_dim"],
        check=True,
    )
DATASET_PATH = sorted(glob.glob(_GLOB))[0]
print(f"Dataset: {DATASET_PATH}")


# In[ ]:


# Dataset class + observation normalization stats. Plumbing — leave as-is.

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
# OBS_DIM = object(10: cube_pos[3] + cube_quat[4] + gripper_to_cube_pos[3])
#         + eef_pos(3) + eef_quat(4) + gripper_qpos(2 — Panda has 2 finger joints) = 19
OBS_DIM = 19
ACT_DIM = 7


def _flatten_obs_batch(obs_group):
    return np.concatenate([obs_group[k][:] for k in OBS_KEYS], axis=-1).astype(np.float32)


class LiftPHDataset(Dataset):
    """Loads the entire dataset into RAM once. ~2 MB total; caching is free and
    ~100x faster than reopening the HDF5 file on every __getitem__."""
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            o, a, r, no, d = [], [], [], [], []
            for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
                demo = f["data"][k]
                o.append(_flatten_obs_batch(demo["obs"]))
                a.append(demo["actions"][:].astype(np.float32))
                r.append(demo["rewards"][:].astype(np.float32))
                no.append(_flatten_obs_batch(demo["next_obs"]))
                d.append(demo["dones"][:].astype(np.float32))
        self.obs      = torch.from_numpy(np.concatenate(o,  axis=0))
        self.actions  = torch.from_numpy(np.concatenate(a,  axis=0))
        self.rewards  = torch.from_numpy(np.concatenate(r,  axis=0))
        self.next_obs = torch.from_numpy(np.concatenate(no, axis=0))
        self.dones    = torch.from_numpy(np.concatenate(d,  axis=0))

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return {
            "obs":      self.obs[idx],
            "action":   self.actions[idx],
            "reward":   self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done":     self.dones[idx],
        }


_full = LiftPHDataset(DATASET_PATH)
OBS_MEAN = _full.obs.mean(dim=0)
OBS_STD  = _full.obs.std(dim=0) + 1e-6
assert OBS_MEAN.shape[0] == OBS_DIM
print(f"Dataset: {len(_full)} transitions (cached in RAM)")
print(f"Action range: [{_full.actions.min():.3f}, {_full.actions.max():.3f}]")
print(f"Reward nonzero fraction: {(_full.rewards != 0).float().mean().item():.4f}")


# ---
# ## Section 1 — BC training  `[TODO]`
# 
# Train a BC policy on the lift-ph dataset. This becomes your *frozen backbone* — Section 2's residual sits on top of it.
# 
# We give a **suggested 3-layer MLP architecture** below. It's a known-good baseline for low-dim continuous control on this dataset. You may modify it (depth, width, activation, normalization scheme); defend any change in your writeup.
# 
# **Decisions to defend:**
# 1. Architecture — keep the suggested 3-layer 256-hidden MLP, or change it?
# 2. Loss function — MSE on continuous actions, or something else?
# 3. Training duration — how many epochs? *Be careful*: imitation training has an overfitting curve; longer is not always better.
# 4. Stopping criterion — fixed epochs, loss plateau, validation split, or rollout success rate?
# 
# **Aim for ~75-90% rollout success** (the final eval cell will check this). If you land outside that range, your training loop probably has a bug worth investigating.

# In[ ]:


class BCPolicy(nn.Module):
    """SUGGESTED 3-layer MLP architecture.

    - hidden=256, two ReLU hidden layers
    - tanh output (action space is [-1, +1] per dim)
    - observation normalization via registered buffers (mean/std from data)

    You may modify this. Defend any change in your writeup.
    """
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.register_buffer("obs_mean", OBS_MEAN.clone())
        self.register_buffer("obs_std",  OBS_STD.clone())

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.forward(obs).squeeze(0)


# Suggested hyperparameters. You may modify; defend any change.
BC_EPOCHS = 15        # think carefully — too few = underfit; too many = overfit on imitation data
BC_LR     = 1e-3
BC_BS     = 256

torch.manual_seed(42); np.random.seed(42)
bc_policy = BCPolicy().to(DEVICE)

# TODO: train BC.
#   - Adam optimizer with BC_LR
#   - DataLoader(LiftPHDataset(DATASET_PATH), batch_size=BC_BS, shuffle=True, drop_last=True)
#   - MSE loss against demo actions (or your alternative — defend it)
#   - Print per-epoch loss (every few epochs and at the end)
#   - After training: bc_policy.eval(); freeze all params (requires_grad=False)
#   - Save to /content/bc.pt
#
# Replace the NotImplementedError below with your training loop.

raise NotImplementedError("Implement BC training and save the frozen model to /content/bc.pt")

print("BC trained and frozen.")


# ---
# ## Section 2 — Residual policy  `[TODO]`
# 
# Train a residual correction on top of the **frozen** BC. The executed action is
# 
# ```
# a_executed(s) = clip(a_BC(s) + δ_θ(s), -1, +1)
# ```
# 
# with `δ_θ` a small MLP you train using an offline-RL-style objective on the same dataset.
# 
# **Eight design decisions to defend in your writeup:**
# 
# 1. **Architecture** — δ(s) only, or δ(s, a_BC(s))?
# 2. **Activation** on δ — `tanh × bound`, hard clip, or soft constraint?
# 3. **Bound magnitude** — inspect the action distribution before picking. Is 0.05 right for this data?
# 4. **Algorithm** — TD3+BC, IQL, AWAC, BCQ, pure distillation, something else? **Run at least one ablation** and explain why your choice won.
# 5. **Reward** — sparse terminal as given, or shaped (e.g., `−|cube − eef|`)?
# 6. **Clip δ inside the target Q computation** — yes or no? *There is a right answer.* What happens if you get this wrong?
# 7. **Critic target update** — soft Polyak or hard?
# 8. **Training step count** — what's the convergence behavior? When do you stop?
# 
# **Required diagnostics during training (don't remove these prints):**
# 
# - `q_mean` (min of twin critics on training batch)
# - `delta_mag` (mean abs of `residual.raw_delta(obs)`)
# - `critic_loss`
# - `actor_loss`
# 
# Print every 500 steps. We'll look at the trajectory in the review call.
# 
# **Reasonable target on a T4 in ~10 min:** 5,000–10,000 gradient steps, batch size 256.

# In[ ]:


# TODO: implement ResidualPolicy and QCritic, then train.
# Skeleton below shows expected signatures and a minimal training-loop scaffold.

class ResidualPolicy(nn.Module):
    """Frozen BC + small correction head."""
    def __init__(self, bc_policy, act_dim=ACT_DIM, delta_bound=0.05, hidden=128):
        super().__init__()
        self.bc = bc_policy           # MUST stay frozen
        self.delta_bound = delta_bound
        # TODO: define delta_net
        self.delta_net = None
        self.register_buffer("obs_mean", bc_policy.obs_mean.clone())
        self.register_buffer("obs_std",  bc_policy.obs_std.clone())

    def forward(self, obs):
        """Return executed action (post-bound, clipped to [-1, +1])."""
        # TODO
        raise NotImplementedError

    def raw_delta(self, obs):
        """Return the bounded residual (for diagnostic logging)."""
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.forward(obs).squeeze(0)


class QCritic(nn.Module):
    """Q(s, a). Twin Q recommended."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        # TODO

    def forward(self, obs, act):
        # TODO
        raise NotImplementedError


# --- training-loop scaffolding (feel free to restructure) ---
DELTA_BOUND = 0.05      # TODO: defend in writeup
TRAIN_STEPS = 5000
GAMMA       = 0.99
TAU         = 0.005     # Polyak coefficient
ACTOR_LR    = 1e-4
CRITIC_LR   = 3e-4

torch.manual_seed(42); np.random.seed(42)
residual        = ResidualPolicy(bc_policy, delta_bound=DELTA_BOUND).to(DEVICE)
q_critic        = QCritic().to(DEVICE)
target_q_critic = QCritic().to(DEVICE)
target_q_critic.load_state_dict(q_critic.state_dict())

actor_opt  = torch.optim.Adam(residual.delta_net.parameters(), lr=ACTOR_LR)
critic_opt = torch.optim.Adam(q_critic.parameters(),           lr=CRITIC_LR)

batch_loader = DataLoader(LiftPHDataset(DATASET_PATH), batch_size=256,
                          shuffle=True, num_workers=0, drop_last=True)
batch_iter = iter(batch_loader)

def next_batch():
    global batch_iter
    try:
        return next(batch_iter)
    except StopIteration:
        batch_iter = iter(batch_loader)
        return next(batch_iter)


for step in tqdm(range(TRAIN_STEPS), desc="residual training"):
    batch    = next_batch()
    obs      = batch["obs"].to(DEVICE)
    action   = batch["action"].to(DEVICE)
    reward   = batch["reward"].to(DEVICE).unsqueeze(-1)
    next_obs = batch["next_obs"].to(DEVICE)
    done     = batch["done"].to(DEVICE).unsqueeze(-1)

    # --- Critic update ---
    # TODO
    critic_loss = torch.tensor(0.0)   # placeholder so the print at step % 500 works

    # --- Actor update ---
    # TODO
    actor_loss = torch.tensor(0.0)    # placeholder

    # --- Polyak target update ---
    # TODO

    if step % 500 == 0:
        # Required diagnostics: do NOT remove — we look at the trajectory in review.
        with torch.no_grad():
            q_mean    = 0.0    # TODO: replace with min(q1, q2).mean().item()
            delta_mag = 0.0    # TODO: replace with residual.raw_delta(obs).abs().mean().item()
        print(f"step {step:5d}  q_mean {q_mean:+.3f}  delta_mag {delta_mag:.4f}  "
              f"critic_loss {float(critic_loss):.4f}  actor_loss {float(actor_loss):.4f}")

residual.eval()
torch.save(residual.state_dict(), "/content/residual.pt")
print("Residual trained.")


# ---
# ## Section 3 — Safety shield  `[TODO]`
# 
# A minimal shield: **per-dimension clipping to bounds learned from the training data, with a margin you choose.**
# 
# (Production version would also do rate-limiting and NaN handling. Clip-only is fine for this assignment.)
# 
# **Decisions to defend:**
# 1. How did you pick the margin? (Numbers from the data, not vibes.)
# 2. Why per-dimension instead of a global L2 norm?
# 
# Bonus: report the clip rate during a residual rollout (how often does the shield actually trigger?).

# In[ ]:


class SafetyShield:
    """Clip-only shield. Per-dimension bounds from data with margin."""
    def __init__(self, act_low, act_high):
        self.low  = np.asarray(act_low,  dtype=np.float32)
        self.high = np.asarray(act_high, dtype=np.float32)

    def __call__(self, action, prev_action=None):
        # TODO: clip per-dim
        raise NotImplementedError


# TODO: compute bounds from the training data with a margin you pick.
# Defend the margin in your writeup.
act_low  = None  # TODO
act_high = None  # TODO

shield = SafetyShield(act_low, act_high)
print(f"shield bounds low:  {act_low}")
print(f"shield bounds high: {act_high}")


# ---
# ## Section 4 — Final evaluation: BC vs your residual  `[PREFILLED]`
# 
# Defines the rollout / eval helpers, runs BC and Residual+Shield each on 30 rollouts (seed 42 → same starts), saves a rollout video for both, prints the comparison.
# 
# If anything in this cell errors out, the bug is in your Section 1/2/3 code — fix it there, then re-run.

# In[ ]:


# Eval helpers + BC vs Residual+Shield comparison.
import robosuite
from robosuite.controllers import load_controller_config


def make_env():
    return robosuite.make(
        env_name="Lift", robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, reward_shaping=False,
        horizon=400, ignore_done=False, camera_names="agentview",
    )


def flatten_obs_dict(obs_dict):
    # Dataset stores `cube_pos + cube_quat + gripper_to_cube_pos` under "object";
    # the live env exposes the same vector under "object-state". Alias.
    if "object" not in obs_dict and "object-state" in obs_dict:
        obs_dict = {**obs_dict, "object": obs_dict["object-state"]}
    return np.concatenate([obs_dict[k] for k in OBS_KEYS], axis=-1).astype(np.float32)


def run_rollout(env, policy, shield=None, max_steps=400, render=False):
    obs = flatten_obs_dict(env.reset())
    prev_action = None
    success = False
    latencies, frames = [], []
    for t in range(max_steps):
        obs_t = torch.from_numpy(obs).float().to(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            action = policy.act(obs_t).cpu().numpy()
        latencies.append((time.perf_counter() - t0) * 1000)
        if shield is not None:
            action = shield(action, prev_action)
        prev_action = action
        next_obs_dict, _, done, _ = env.step(action)
        obs = flatten_obs_dict(next_obs_dict)
        if render:
            frames.append(np.flipud(env.sim.render(height=256, width=256, camera_name="agentview")))
        if env._check_success():
            success = True; break
        if done:
            break
    return success, t + 1, latencies, frames


def evaluate(policy, shield=None, n_rollouts=30, seed=42, save_video=None, label=""):
    np.random.seed(seed); torch.manual_seed(seed)
    env = make_env()
    successes, steps_list, all_lat = [], [], []
    saved = False
    pbar = tqdm(range(n_rollouts), desc=f"eval {label}")
    for _ in pbar:
        render = save_video is not None and not saved
        ok, steps, lat, frames = run_rollout(env, policy, shield=shield, render=render)
        successes.append(ok); steps_list.append(steps); all_lat.extend(lat)
        if render and ok and save_video:
            imageio.mimsave(save_video, frames, fps=20); saved = True
        pbar.set_postfix(success_rate=f"{np.mean(successes):.2f}")
    sr = float(np.mean(successes))
    ok_steps = [s for s, o in zip(steps_list, successes) if o]
    return {
        "success_rate":          sr,
        "n":                     n_rollouts,
        "mean_steps_on_success": float(np.mean(ok_steps)) if ok_steps else 0.0,
        "p99_latency_ms":        float(np.percentile(all_lat, 99)),
    }


# --- Run both evals ---
print("Evaluating BC...")
bc_results = evaluate(bc_policy, n_rollouts=30, seed=42,
                      save_video="/content/rollout_bc.mp4", label="BC")

print("\nEvaluating Residual + Shield...")
residual_results = evaluate(residual, shield=shield, n_rollouts=30, seed=42,
                             save_video="/content/rollout_residual.mp4", label="residual")

# --- Comparison ---
print("\n" + "=" * 66)
print(f"{'Metric':<28} {'BC':>10} {'Residual':>10}  {'Delta':>10}")
print("-" * 66)
for k in ["success_rate", "mean_steps_on_success", "p99_latency_ms"]:
    a, b = bc_results[k], residual_results[k]
    print(f"{k:<28} {a:>10.3f} {b:>10.3f}  {b-a:>+10.3f}")

def pcount(m): return sum(p.numel() for p in m.parameters())
print(f"\nBC params:       {pcount(bc_policy):,}")
print(f"Residual params: {pcount(residual):,}")

# --- Inline video preview ---
from IPython.display import Video
Video("/content/rollout_residual.mp4", embed=True, width=400)

