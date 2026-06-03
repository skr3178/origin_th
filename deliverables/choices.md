# Deliverables — what must be submitted

Source of truth: `take_home_Robot_learning_assignment.pdf` + `origin_assignment_takehome.ipynb`.

## A. Files in the final `.zip`

- [ ] **`origin_assignment_colab.ipynb`** — the completed notebook (all `[TODO]` cells filled, runs end-to-end from a fresh kernel, reproducible from seed 42).
- [ ] **`writeup.md`** — defends every design decision (see Section C below).
- [ ] **`rollout_residual.mp4`** — rendered rollout of the residual+shield policy (produced by the Section 5 eval cell, `save_video="/content/rollout_residual.mp4"`).
- [ ] *(optional)* **`notes.md`** — scratch notes / failed experiments with diagnosis (encouraged).

## B. Code to implement in the notebook (the TODO cells)

### Section 1 — BC baseline (frozen backbone) — cell 7
- [ ] `BCPolicy` — keep or modify the suggested 3-layer / 256-hidden / ReLU / `tanh`-output MLP with obs-normalization buffers.
- [ ] **BC training loop**: Adam (`BC_LR=1e-3`), `DataLoader(LiftPHDataset, bs=256, shuffle, drop_last)`, MSE on demo actions (or defended alternative), per-epoch loss prints.
- [ ] After training: `bc_policy.eval()` + freeze all params (`requires_grad=False`). **BC stays frozen for the rest of the assignment.**
- [ ] Save to `/content/bc.pt`.
- [ ] Target ~75–90% rollout success.

### Section 2 — Residual policy — cell 9
- [ ] `ResidualPolicy` — define `delta_net`; implement:
  - `forward(obs)` → executed action `clip(a_BC + δ, -1, +1)`
  - `raw_delta(obs)` → bounded residual (for diagnostics)
- [ ] `QCritic` — `__init__` + `forward(obs, act)`; **twin Q recommended**.
- [ ] **Training loop**: critic update, actor update, Polyak target update.
- [ ] **Required diagnostic print every 500 steps** (do NOT remove):
  - `q_mean` = `min(q1, q2).mean()`
  - `delta_mag` = `residual.raw_delta(obs).abs().mean()`
  - `critic_loss`
  - `actor_loss`
- [ ] Save to `/content/residual.pt`. Target 5,000–10,000 steps, bs 256.

### Section 3 — Safety shield — cell 11
- [ ] `SafetyShield.__call__(action, prev_action=None)` — per-dimension clip to `[low, high]`.
- [ ] Compute `act_low` / `act_high` from training-data action distribution **+ a chosen margin**.
- [ ] *(bonus)* report clip rate during a residual rollout.

### Section 5 — Final eval — cell 13 (prefilled, run as-is)
- [ ] Produces BC vs Residual+Shield table (`success_rate`, `mean_steps_on_success`, `p99_latency_ms`), param counts, and saves both rollout videos. **`rollout_residual.mp4` is the deliverable.**

## C. Decisions to defend in `writeup.md`

**Task 1 — BC (4):** (1) architecture · (2) loss function · (3) training duration · (4) stopping criterion.

**Task 2 — BC failure-mode investigation:** which rollouts fail, *when*, and *what* BC does wrong (have an opinion before designing the residual).

**Task 3 — Residual (8):** (1) architecture `δ(s)` vs `δ(s, a_BC)` · (2) activation on δ · (3) bound magnitude (inspect action dist; is 0.05 right?) · (4) algorithm — TD3+BC / IQL / AWAC / BCQ / distillation, **+ ≥1 ablation** · (5) reward (sparse vs shaped) · (6) clip δ inside target Q? (*there is a right answer*) · (7) critic target update soft vs hard · (8) training step count / convergence.

**Task 4 — Shield (2):** (1) how the margin was picked (numbers, not vibes) · (2) why per-dimension vs global L2.

**Real-world instincts:** report latency, safety bounds with margin, residual stays close to BC.

## D. Hard rules / constraints
- BC frozen after Section 1.
- `lift-ph` only.
- Reproducible from fixed seed (42); final eval = 30 rollouts, seed 42, same starts as BC eval.
- A residual that *doesn't* beat BC + a clear writeup is still valuable; success rate is **not** in the rubric.
- LLM coding assistants OK; human help / copying submissions not OK.
