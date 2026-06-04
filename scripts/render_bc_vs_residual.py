#!/usr/bin/env python
"""Side-by-side BC vs Residual+Shield on *flip* rollouts — a video that actually
shows a difference.

A single successful rollout looks identical for BC, the residual, and a dataset
demo (the residual is a sub-perceptual ±0.005 nudge). The only way to *see* the
residual do something is a **paired, same-start rollout where the two diverge**:
same cube, one policy succeeds and the other fails.

This renders, from the frozen checkpoints (no retraining):
  Row A — a start where the residual FIXES a BC failure  (BC fail, Res success)
  Row B — a start where the residual BREAKS a BC success  (BC success, Res fail)
Showing both directions is deliberate: the residual fixed 1 / broke 3 overall
(net wash), so a one-sided clip would misrepresent the result.

Protocol matches Task 5: 30 rollouts, seed 42 (identical cube starts), shield in
the loop for the residual. Output: out/rollout_bc_vs_residual.mp4

Run:  python scripts/render_bc_vs_residual.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import cv2
import imageio

from residual_lift.config import DEVICE, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import load_residual
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import make_env, flatten_obs_dict

N, SEED, MAX_STEPS = 30, 42, 400
RES, FPS = 256, 20
GREEN, RED = (40, 180, 70), (210, 50, 50)   # RGB


def success_vector(policy, shield):
    """Per-rollout success over the seed-42 protocol (no frames)."""
    np.random.seed(SEED); torch.manual_seed(SEED)
    env = make_env()
    succ = []
    for _ in range(N):
        obs = flatten_obs_dict(env.reset())
        prev, ok = None, False
        for t in range(MAX_STEPS):
            with torch.no_grad():
                a = policy.act(torch.from_numpy(obs).float().to(DEVICE)).cpu().numpy()
            if shield is not None:
                a = shield(a, prev)
            prev = a
            od, _, done, _ = env.step(a)
            obs = flatten_obs_dict(od)
            if env._check_success():
                ok = True; break
            if done:
                break
        succ.append(ok)
    return np.array(succ)


def render_at(policy, shield, k):
    """Replay to rollout k (identical starts) and capture that rollout's frames."""
    np.random.seed(SEED); torch.manual_seed(SEED)
    env = make_env()
    frames, success = [], False
    for i in range(k + 1):
        obs = flatten_obs_dict(env.reset())
        prev, ok, fr = None, False, []
        for t in range(MAX_STEPS):
            with torch.no_grad():
                a = policy.act(torch.from_numpy(obs).float().to(DEVICE)).cpu().numpy()
            if shield is not None:
                a = shield(a, prev)
            prev = a
            od, _, done, _ = env.step(a)
            obs = flatten_obs_dict(od)
            if i == k:
                fr.append(np.flipud(env.sim.render(height=RES, width=RES,
                                                    camera_name="agentview")).copy())
            if env._check_success():
                ok = True; break
            if done:
                break
        if i == k:
            frames, success = fr, ok
    return frames, success


def tile(frame, title, color):
    """256x256 frame -> bordered tile with a title bar."""
    bordered = cv2.copyMakeBorder(frame, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=color)
    w = bordered.shape[1]
    bar = np.full((30, w, 3), 20, np.uint8)
    cv2.putText(bar, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, bordered])


def status(ok, ended):
    return ("SUCCESS" if ok else "FAIL (timeout)") if ended else "running..."


def build_row(label, k, bc, residual, shield):
    fb, bc_ok = render_at(bc, None, k)
    fr, rs_ok = render_at(residual, shield, k)
    end_b, end_r = len(fb) - 1, len(fr) - 1
    n = max(len(fb), len(fr))
    fb += [fb[-1]] * (n - len(fb)); fr += [fr[-1]] * (n - len(fr))
    bc_col = GREEN if bc_ok else RED
    rs_col = GREEN if rs_ok else RED
    out = []
    for t in range(n):
        lt = tile(fb[t], f"BC  [{status(bc_ok, t >= end_b)}]", bc_col)
        rt = tile(fr[t], f"Residual+Shield  [{status(rs_ok, t >= end_r)}]", rs_col)
        gap = np.full((lt.shape[0], 8, 3), 20, np.uint8)
        row = np.hstack([lt, gap, rt])
        cap = np.full((26, row.shape[1], 3), 35, np.uint8)
        cv2.putText(cap, f"start #{k}: {label}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 230, 120), 1, cv2.LINE_AA)
        out.append(np.vstack([cap, row]))
    return out, (bc_ok, rs_ok)


def main():
    bc = load_bc()
    residual = load_residual(bc)
    shield = build_shield(verbose=False)

    print("Computing per-rollout outcomes (seed 42)...")
    bc_s = success_vector(bc, None)
    rs_s = success_vector(residual, shield)
    fixed = np.where((bc_s == 0) & (rs_s == 1))[0]
    broke = np.where((bc_s == 1) & (rs_s == 0))[0]
    print(f"  BC {bc_s.mean():.3f}  Residual {rs_s.mean():.3f}")
    print(f"  residual-FIXED starts: {fixed.tolist()}")
    print(f"  residual-BROKE starts: {broke.tolist()}")

    rows_spec = []
    if fixed.size:
        rows_spec.append(("residual FIXES a BC failure", int(fixed[0])))
    if broke.size:
        rows_spec.append(("residual BREAKS a BC success", int(broke[0])))
    if not rows_spec:
        sys.exit("No flip rollouts in this run — BC and residual agreed on all 30 starts.")

    bands = []
    for label, k in rows_spec:
        print(f"Rendering start #{k}: {label} ...")
        band, _ = build_row(label, k, bc, residual, shield)
        bands.append(band)

    n = max(len(b) for b in bands)
    bands = [b + [b[-1]] * (n - len(b)) for b in bands]
    W = max(b[0].shape[1] for b in bands)
    sep = np.full((6, W, 3), 0, np.uint8)
    frames = []
    for t in range(n):
        stacked = []
        for b in bands:
            f = b[t]
            if f.shape[1] != W:
                f = cv2.copyMakeBorder(f, 0, 0, 0, W - f.shape[1], cv2.BORDER_CONSTANT, value=20)
            stacked.append(f); stacked.append(sep)
        frames.append(np.vstack(stacked[:-1]))

    out = os.path.join(OUT_DIR, "rollout_bc_vs_residual.mp4")
    imageio.mimsave(out, frames, fps=FPS)
    print(f"\nSaved: {out}  ({len(frames)} frames, ~{len(frames)/FPS:.1f}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
