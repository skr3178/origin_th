"""Section 4 — Final evaluation: BC vs Residual+Shield  [PREFILLED].

Rollout / eval helpers, runs a policy on N rollouts (seed 42 -> same starts),
saves a rollout video, and prints the BC-vs-Residual comparison table.

If anything here errors out, the bug is in your Section 1/2/3 code — fix it
there, then re-run.
"""
import time

import numpy as np
import torch
import imageio

from . import config  # noqa: F401  (sets MUJOCO_GL before robosuite import)
from .config import OBS_KEYS, DEVICE, BC_VIDEO, RESIDUAL_VIDEO

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
    n_steps, n_clipped = 0, 0   # bonus: how often the shield alters the action
    for t in range(max_steps):
        obs_t = torch.from_numpy(obs).float().to(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            action = policy.act(obs_t).cpu().numpy()
        latencies.append((time.perf_counter() - t0) * 1000)
        if shield is not None:
            shielded = shield(action, prev_action)
            n_steps += 1
            if not np.allclose(shielded, action):
                n_clipped += 1
            action = shielded
        prev_action = action
        next_obs_dict, _, done, _ = env.step(action)
        obs = flatten_obs_dict(next_obs_dict)
        if render:
            frames.append(np.flipud(env.sim.render(height=256, width=256, camera_name="agentview")))
        if env._check_success():
            success = True; break
        if done:
            break
    return success, t + 1, latencies, frames, (n_clipped, n_steps)


def evaluate(policy, shield=None, n_rollouts=30, seed=42, save_video=None, label=""):
    from tqdm.auto import tqdm
    np.random.seed(seed); torch.manual_seed(seed)
    env = make_env()
    successes, steps_list, all_lat = [], [], []
    clip_events, clip_steps = 0, 0
    saved = False
    pbar = tqdm(range(n_rollouts), desc=f"eval {label}")
    for _ in pbar:
        render = save_video is not None and not saved
        ok, steps, lat, frames, (nc, ns) = run_rollout(env, policy, shield=shield, render=render)
        successes.append(ok); steps_list.append(steps); all_lat.extend(lat)
        clip_events += nc; clip_steps += ns
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
        "shield_clip_rate":      float(clip_events / clip_steps) if clip_steps else 0.0,
    }


def compare(bc_policy, residual, shield, n_rollouts=30, seed=42):
    """Run both evals, save videos, and print the comparison table."""
    print("Evaluating BC...")
    bc_results = evaluate(bc_policy, n_rollouts=n_rollouts, seed=seed,
                          save_video=BC_VIDEO, label="BC")

    print("\nEvaluating Residual + Shield...")
    residual_results = evaluate(residual, shield=shield, n_rollouts=n_rollouts, seed=seed,
                                save_video=RESIDUAL_VIDEO, label="residual")

    print("\n" + "=" * 66)
    print(f"{'Metric':<28} {'BC':>10} {'Residual':>10}  {'Delta':>10}")
    print("-" * 66)
    for k in ["success_rate", "mean_steps_on_success", "p99_latency_ms"]:
        a, b = bc_results[k], residual_results[k]
        print(f"{k:<28} {a:>10.3f} {b:>10.3f}  {b-a:>+10.3f}")
    print(f"{'shield_clip_rate':<28} {'-':>10} {residual_results['shield_clip_rate']:>10.3f}")

    def pcount(m):
        return sum(p.numel() for p in m.parameters())
    print(f"\nBC params:       {pcount(bc_policy):,}")
    print(f"Residual params: {pcount(residual):,}")
    print(f"\nVideos saved:\n  {BC_VIDEO}\n  {RESIDUAL_VIDEO}")
    return bc_results, residual_results
