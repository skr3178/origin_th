# Plan — Origin take-home: residual policy on robomimic lift-ph

## Context

This is the Origin AI-Research-Engineer take-home (`take_home_Robot_learning_assignment.pdf`):
miniaturized version of Origin's stack — **frozen BC backbone → small residual policy → safety shield**
— on robomimic **lift-ph**. Setup is done: the `origin10x` conda env works (CUDA on RTX 3060,
offscreen EGL render verified), the dataset is a permanent local copy at
`data/lift/ph/low_dim_v141.hdf5`, and the notebook is ported to a modular package `residual_lift/`.

**Task 1 (BC) is already complete**: implemented in `section1_bc.py` (3-layer/256 MLP, MSE, early-stop on
val MSE), trained, frozen to `out/bc.pt`, **86.7% rollout success** (in the 75–90% target band).

Remaining: Tasks 2–5. Per user decisions this run is **lean, scripts-only, TD3+BC primary, notes+plots
(not a full writeup)**. The rubric weights Diagnostics 25% + Defense 25% and explicitly does NOT grade
success rate — so the required diagnostic prints, one ablation, and decision rationale matter more than peak %.

## Approach (what gets built)

### Task 2 — BC failure diagnostics  → NEW `scripts/diagnose_bc.py`
- Replay the 30 BC rollouts (seed 42), record per-rollout: success, failure step, and a coarse
  phase label (reached cube? grasped? lifted?) using cube height + `gripper_to_cube_pos` from the obs.
- Save 1–2 plots to `out/`: failure timeline (when episodes end) and cube-height trace for failed vs
  successful rollouts. Output a short stdout summary feeding Task 3.
- Reuse `section4_eval.make_env / flatten_obs_dict / run_rollout` (extend `run_rollout` to optionally
  return the per-step obs trace).

### Task 3 — Residual (TD3+BC)  → implement in `residual_lift/section2_residual.py`
Replace the `[TODO]` bodies of `ResidualPolicy`, `QCritic`, `train_residual`:
- **ResidualPolicy**: `delta_net` = small MLP (2×128 ReLU) on normalized obs → `tanh × delta_bound`.
  `forward = clip(bc(obs).detach() + raw_delta(obs), -1, 1)`; `raw_delta` returns the bounded δ.
- **QCritic**: twin Q (two MLP heads, 2×256) over `concat(obs, action)`.
- **train_residual (TD3+BC)**, the 8 defended decisions baked in + commented:
  1. arch: δ(s) only (state-conditioned). 2. activation: `tanh × bound`. 3. bound: `delta_bound=0.05`
  (note in notes.md that gripper dim 6 has std≈0.9 so 0.05 is conservative there). 4. algorithm: TD3+BC.
  5. reward: sparse terminal as given (note `done ≡ reward` in this data → terminal bootstrap cut is correct).
  6. **clip δ inside target Q = YES** (target action must be the executable, bounded+clipped action;
  not clipping trains Q on unrealizable actions → overestimation). 7. target update: soft Polyak τ=0.005.
  8. steps: TRAIN_STEPS=10000, bs=256, γ=0.99, actor_lr=1e-4, critic_lr=3e-4, policy_delay=2,
  target-smoothing noise σ=0.1/clip 0.2.
  - Critic trained on dataset actions; actor maximizes `Q1(s, a_exec)` with TD3+BC normalization
    `λ = α/|Q|.mean().detach()` (α=2.5) **plus** a BC-anchor `MSE(a_exec, a_demo)` so the residual stays
    near demonstrated actions.
  - **Required diagnostics every 500 steps**: `q_mean` (min of twins), `delta_mag` (`raw_delta` abs mean),
    `critic_loss`, `actor_loss`. Also append them to a history dict for plotting.
- Add an `ablate_clip_in_target: bool` arg to `train_residual` for the single required ablation.

### Task 3 ablation  → NEW `scripts/ablation_residual.py`
Run train_residual with clip-in-target ON vs OFF; overlay `q_mean` and `delta_mag` curves to `out/`.
Demonstrates decision #6's "right answer" (OFF → Q overestimates, δ saturates at the bound).

### Task 4 — Safety shield  → implement in `residual_lift/section3_shield.py`
- `build_shield()`: per-dim `act_low/high` = train-split demo-action min/max expanded by a small margin
  (e.g. 5% of per-dim range — defended as: don't clip valid in-distribution actions, only catch outliers).
- `SafetyShield.__call__`: `np.clip(action, low, high)`.
- Bonus clip-rate: extend `section4_eval` to optionally count the fraction of steps the shield alters the action.

### Task 5 — Final eval  (already prefilled in `section4_eval.compare`)
Run `scripts/run_eval.py`: BC vs residual+shield, 30 rollouts seed 42, saves `out/rollout_{bc,residual}.mp4`,
prints comparison table with success rate, mean steps, p99 latency (+ clip rate).

### Notes + plots  → NEW `notes.md`
Bullet rationale per decision (Tasks 1–4), Task-2 findings, Task-5 numbers, and references to the `out/` plots.
(No prose writeup — user writes that.)

## Files

- Modify: `residual_lift/section2_residual.py` (residual + TD3+BC), `residual_lift/section3_shield.py` (shield),
  `residual_lift/section4_eval.py` (optional obs-trace + clip-rate), `residual_lift/config.py` (plot/ablation paths).
- New: `scripts/diagnose_bc.py`, `scripts/ablation_residual.py`, `notes.md`.
- Prereq: `pip install matplotlib` into `origin10x` (for plots).

## Verification (all in `origin10x`, from `10x/`)

1. `python scripts/diagnose_bc.py` → prints failure summary, writes BC diagnostic plots to `out/`.
2. `python scripts/train_residual.py` → diagnostic line every 500 steps (q_mean/delta_mag/critic_loss/actor_loss),
   saves `out/residual.pt` + training-curve plot. Sanity: `delta_mag` stays < bound, `q_mean` doesn't diverge.
3. `python scripts/ablation_residual.py` → overlay plot showing clip-ON vs clip-OFF behavior.
4. `python scripts/run_eval.py` → comparison table; confirm residual ≥ BC (or, if not, a clean diagnosed result —
   the brief says that's acceptable), videos written to `out/`.
5. Skim `notes.md` — every Task-3 decision has a one-line defense and the required diagnostics are present.

## Status — COMPLETE (lean run)

- [x] Setup (env, dataset, package port)
- [x] Task 1 — BC (frozen, 86.7%)
- [x] Task 2 — BC diagnostics (4 failures, all stuck at grasp/lift phase)
- [x] Task 3 — Residual (TD3+BC), shipped bound=0.005 + clip-in-target ablation
- [x] Task 4 — Safety shield (per-dim, 5% margin, 6.3% clip rate, NaN guard)
- [x] Task 5 — Final eval: BC 0.867 → Residual+Shield 0.900 (+3.3pp)
- [x] notes.md

Key finding: residual recovers/slightly-improves BC but cannot beat it by much because
lift-ph is all-expert data (no improvement signal for the offline critic). Bound is the
critical knob: 0.05 destroys BC (0.0-0.15), 0.005 stays safe (0.90). See notes.md.

Still out of scope: notebook port, prose writeup.md, IQL/AWAC, multi-seed.

## Out of scope (this run)

Porting final code back into the `.ipynb` deliverable, the prose `writeup.md`, IQL/AWAC alternatives, and
multi-seed sweeps. Can follow up later.
