#!/usr/bin/env python
"""Task 2 — BC failure-mode diagnostics.

Replays the frozen BC for 30 rollouts (seed 42, same starts as final eval) and
records per-step traces (cube height, gripper->cube distance, gripper opening).
For each rollout it classifies the outcome into a coarse manipulation phase:

    reach  -> did the gripper get to the cube?
    grasp  -> did it close on the cube while there?
    lift   -> did the cube rise enough to count as success?

Saves two plots to out/ and prints a per-phase failure summary that motivates
the residual design (Task 3).

Run:  python scripts/diagnose_bc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section4_eval import make_env, flatten_obs_dict

N_ROLLOUTS = 30
SEED       = 42
MAX_STEPS  = 400

# Phase thresholds (metres). Cube half-size ~0.02; Lift success is cube lifted
# ~0.04 above the table. These are diagnostic buckets, not the env's success test.
REACH_DIST = 0.030   # gripper considered "at" the cube below this distance
GRASP_OPEN = 0.045   # summed finger qpos below this == closed on something
LIFT_RISE  = 0.040   # cube z rise above its rest height that counts as a lift


def rollout_trace(env, policy):
    """One BC rollout. Returns success flag, end step, and per-step arrays."""
    obs_dict = env.reset()
    obs = flatten_obs_dict(obs_dict)
    cube_z, dist, grip = [], [], []
    success = False
    t = 0
    for t in range(MAX_STEPS):
        with torch.no_grad():
            action = policy.act(torch.from_numpy(obs).float().to(DEVICE)).cpu().numpy()
        obs_dict, _, done, _ = env.step(action)
        cube_z.append(float(obs_dict["cube_pos"][2]))
        dist.append(float(np.linalg.norm(obs_dict["gripper_to_cube_pos"])))
        grip.append(float(np.sum(np.abs(obs_dict["robot0_gripper_qpos"]))))
        obs = flatten_obs_dict(obs_dict)
        if env._check_success():
            success = True; t += 1; break
        if done:
            t += 1; break
    return success, t, np.array(cube_z), np.array(dist), np.array(grip)


def classify(success, cube_z, dist, grip):
    """Coarse failure phase for a rollout."""
    if success:
        return "success"
    reached = dist.min() < REACH_DIST
    grasped = reached and grip[dist < REACH_DIST].min() < GRASP_OPEN
    lifted  = (cube_z.max() - cube_z[0]) > LIFT_RISE
    if not reached:
        return "fail:never_reached"
    if not grasped:
        return "fail:reached_no_grasp"
    if not lifted:
        return "fail:grasp_no_lift"
    return "fail:lifted_then_dropped"


def main():
    print(f"Device: {DEVICE}")
    bc = load_bc()
    np.random.seed(SEED); torch.manual_seed(SEED)
    env = make_env()

    records = []
    traces = []
    for i in range(N_ROLLOUTS):
        success, end, cube_z, dist, grip = rollout_trace(env, bc)
        phase = classify(success, cube_z, dist, grip)
        records.append(dict(i=i, success=success, end=end, phase=phase,
                            min_dist=float(dist.min()),
                            max_lift=float(cube_z.max() - cube_z[0])))
        traces.append((success, cube_z))
        print(f"  rollout {i:2d}  {'OK ' if success else 'FAIL'}  end={end:3d}  "
              f"min_dist={dist.min():.3f}  max_lift={cube_z.max()-cube_z[0]:+.3f}  {phase}")

    # --- summary ---
    sr = np.mean([r["success"] for r in records])
    from collections import Counter
    phases = Counter(r["phase"] for r in records)
    print("\n" + "=" * 56)
    print(f"BC success rate: {sr:.3f}  ({sum(r['success'] for r in records)}/{N_ROLLOUTS})")
    print("Outcome breakdown:")
    for p, c in sorted(phases.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {p:<26} {c}")
    fail_steps = [r["end"] for r in records if not r["success"]]
    if fail_steps:
        print(f"Failures run to step (mean / max): {np.mean(fail_steps):.0f} / {max(fail_steps)} "
              f"(horizon {MAX_STEPS}) -> they time out, not crash.")

    # --- plot 1: per-rollout outcome + end step ---
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"success": "tab:green"}
    for r in records:
        c = colors.get(r["phase"], "tab:red")
        ax.bar(r["i"], r["end"], color=c)
    ax.set_xlabel("rollout index"); ax.set_ylabel("episode length (steps)")
    ax.set_title("BC rollouts: green=success, red=failure (bar height = steps survived)")
    ax.axhline(MAX_STEPS, ls="--", c="gray", lw=0.8, label=f"horizon={MAX_STEPS}")
    ax.legend()
    p1 = os.path.join(OUT_DIR, "bc_diag_outcomes.png")
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig)

    # --- plot 2: cube-height (lift) traces, failed vs success ---
    fig, ax = plt.subplots(figsize=(9, 4))
    for success, cube_z in traces:
        lift = cube_z - cube_z[0]
        ax.plot(lift, color=("tab:green" if success else "tab:red"),
                alpha=0.5, lw=1.0)
    ax.axhline(LIFT_RISE, ls="--", c="k", lw=0.8, label=f"lift threshold {LIFT_RISE} m")
    ax.set_xlabel("step"); ax.set_ylabel("cube height above rest (m)")
    ax.set_title("Cube lift over time — green=success, red=failure")
    ax.legend()
    p2 = os.path.join(OUT_DIR, "bc_diag_lift_traces.png")
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig)

    print(f"\nPlots saved:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
