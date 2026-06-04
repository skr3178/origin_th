# Notes — residual policy on robomimic lift-ph

Decision-by-decision rationale + diagnostics, for writing up `writeup.md`. All
numbers reproduced by the `scripts/` in `origin10x`. Plots referenced live in `out/`.

Pipeline: `train_bc.py` → `diagnose_bc.py` → `train_residual.py` → `ablation_residual.py` → `run_eval.py`.

---

## Headline result (30 rollouts, seed 42, same starts)

The **notebook deliverable** (`origin_assignment_takehome.ipynb`, run end-to-end) gives:

| Metric | BC | Residual + Shield | Δ |
|---|---|---|---|
| success_rate | 0.867 (26/30) | 0.800 (24/30) | −0.067 |
| p99 latency (ms) | ~0.3 | ~0.5 | negligible |
| shield clip rate | — | 0.063 | — |

**The residual is statistically indistinguishable from BC.** Two identical-config runs
landed at 0.80 (notebook) and 0.90 (an earlier script run) — both within ~1 standard
error (±~2 successes) of BC's 26/30. The *sign* of the delta flips between runs; there is
no real improvement. `delta_mag` settles ≈0.0048 (inside the 0.005 bound), so the residual
genuinely stays close to BC. This null result is the honest finding (see "Why the residual
can't beat BC" below) — the brief explicitly values it over a cherry-picked high score.

---

## Dataset facts that drove decisions

- 200 proficient-human demos, 9,666 transitions, OSC_POSE @ 20 Hz, **sparse reward**.
- Action = 7-dim OSC deltas: dpos[0:3], daxis-angle[3:6], gripper[6]. Per-dim ranges
  are **heterogeneous**: dims 0/2/6 swing the full ±1 (std 0.26/0.49/0.91) while the
  rotation dims 3/4/5 barely move (std 0.02–0.08). → a uniform residual bound means
  very different things per dim (motivates Task 4 being **per-dim**).
- `reward ∈ {0,1}` (10.4% nonzero) and **`done` ≡ `reward`** (both fire on the lift
  step). → the TD target `y = r + γ(1−done)Q'` correctly cuts the bootstrap at success.
- Data is **all-expert / all-success** — this is the single most important fact for
  Task 3 (see "Why the residual can't beat BC by much").

---

## Task 1 — BC  (`section1_bc.py`)

- **(1) Architecture**: kept the suggested 3-layer/256/ReLU + **tanh** output (73k
  params). tanh matches the [−1,1] action box; the map is a simple 19→7 regression.
- **(2) Loss**: **MSE** to the expert action = MLE under a fixed-covariance Gaussian
  policy. This is robomimic's default BC loss. (GMM-NLL would be the upgrade if the
  expert were multimodal; lift-ph is single proficient-human → near-unimodal → MSE fine.)
- **(3) Duration / (4) Stopping**: **early stopping on held-out validation MSE** using
  the file's *own* train/valid demo masks (180/20), keeping best-val weights. Normalization
  stats computed on the **train split only** (no leakage). Stopped at **epoch 5**: val MSE
  bottomed (0.0237) then **rose** (→0.028) while train MSE kept falling — the imitation
  overfitting curve the brief warns about, observed directly.
- Result: **86.7%** success (in the 75–90% target band).

## Task 2 — BC failure diagnostics  (`diagnose_bc.py`)

Replayed all 30 BC rollouts, classified each by manipulation phase. Of the 4 failures:

| failure phase | count |
|---|---|
| reached cube, no grasp | 2 |
| grasped, no lift | 1 |
| never reached cube | 1 |

- **All failures time out at step 400 (never crash)** — BC gets *stuck* near the cube.
- Plot `out/bc_diag_lift_traces.png`: every failure (red) stays flat at ~0 lift; the cube
  never comes up. Successes (green) lift cleanly around step 40–60.
- **Takeaway**: BC's residual errors are concentrated in the **fine grasp/lift** phase
  (3 of 4 failures are at/after reaching the cube). This is precisely the regime where a
  small correction *could* help — but it's also precision-critical (see Task 3).

## Task 3 — Residual (TD3+BC)  (`section2_residual.py`)

`a_exec(s) = clip(a_BC(s) + δ_θ(s), −1, +1)`, with `a_BC` frozen.

1. **Architecture**: δ(s) only (state-conditioned, 2×128 MLP). a_BC(s) is a deterministic
   function of s, so feeding it adds no information.
2. **Activation**: `tanh × bound` — smooth and hard-bounded.
3. **Bound magnitude — the key empirical decision.** The scaffold's **0.05 is far too large**.
   Bound sweep (20 rollouts each):

   | bound | settled `delta_mag` | success |
   |---|---|---|
   | 0.05 | 0.047 (saturated) | 0.00–0.15 ❌ |
   | 0.02 | 0.019 | 0.75 |
   | 0.01 | 0.0095 | 0.85 (=BC) |
   | **0.005** | **0.0048** | **0.90 (best)** |

   The residual **always saturates its budget** (the offline Q keeps "wanting" more), and
   a 0.05 perturbation on the precision dims wrecks the grasp. We ship **0.005**.
   (Control test: the residual wrapper with δ forced to 0 reproduces BC exactly (0.85),
   so the degradation at large bounds is the δ, not a wrapper bug.)
4. **Algorithm**: **TD3+BC** — twin critics, delayed actor (policy_delay=2), target
   smoothing, and a BC-anchor `MSE(a_exec, a_demo)` keeping the residual near demo actions.
   λ = α/|Q|.mean(), α=2.5. (Ablation on α showed large α over-trusts a miscalibrated Q.)
5. **Reward**: sparse terminal as given — no shaping. With `done ≡ reward`, terminal
   bootstrap is handled correctly; shaping (e.g. −|cube−eef|) was unnecessary and risks
   biasing toward reaching over grasping.
6. **Clip δ inside target Q = YES** — see ablation below.
7. **Critic target update**: soft Polyak, τ=0.005 (both critic and target actor).
8. **Step count**: 10k. `q_mean` rises smoothly and stays bounded (no divergence);
   `critic_loss` → ~2e-4; `delta_mag` settles fast and flat. Plot: `out/residual_training_curves.png`.

**Required diagnostics** (`q_mean`, `delta_mag`, `critic_loss`, `actor_loss`) printed every
500 steps — see `train_residual.py` stdout and the 4-panel curve plot.

### Ablation — decision #6, clip δ inside the target Q  (`ablation_residual.py`)
Run at bound 0.05 (effect most visible there). Identical training except the target
action is/ isn't clipped to the executable set.
- Plot `out/residual_ablation_clip.png`: clip-**OFF** (red) **consistently overestimates
  `q_mean`** vs clip-ON (green), final +0.573 vs +0.532.
- **Right answer = clip ON.** Not clipping evaluates Q at actions the policy can never
  execute (outside [−1,1]) → out-of-distribution **value overestimation**. The effect is
  modest here because the small residual keeps the OOD region thin; it amplifies with a
  larger bound / longer horizon.

### Why the residual can't beat BC by much (the real diagnosis)
lift-ph is **all proficient-human, all-success** data. The critic only ever sees
near-optimal actions, so it has **no signal about what is *better* than the demos** — and
offline RL (TD3+BC) deliberately won't extrapolate beyond the data. So the residual's
ceiling *is* BC: small bound → recovers BC (and a noisy +3 pp); large bound → exploits the
miscalibrated Q and regresses. A residual genuinely *beating* BC would need sub-optimal /
exploratory data (or online interaction) to learn an improvement direction. This matches
the bound-sweep curve exactly and is the honest result the brief invites.

## Task 4 — Safety shield  (`section3_shield.py`)
- **Per-dimension** clip to demo-action [min,max] **+ 5% margin**, then clamped to the
  env's hard ±1. Per-dim (not global L2) because the action dims are heterogeneous: an
  L2 ball would over-constrain the big dims (0/2/6) or leave the rotation dims unbounded.
- **Margin rationale**: 5% of each dim's data span gives the learned residual head-room
  beyond exactly-seen actions (don't clip valid in-distribution corrections) while still
  catching gross outliers.
- **Clip rate**: **6.3%** of steps with the shipped 0.005 residual (re-measured over the
  30-rollout seed-42 eval → shield is a non-intrusive safety net), vs **37%** measured on
  the broken 0.05 residual — the shield correctly stays out of the way when the policy is
  well-behaved and would clamp hard if it weren't.
- **NaN/Inf guard**: actions are passed through `np.nan_to_num` (→ 0.0) before clipping, so
  a non-finite policy output can never reach the env (`np.clip` alone passes NaN through).

## Task 5 — Final eval  (`section4_eval.py`, prefilled)
BC vs Residual+Shield, 30 rollouts, seed 42 (same starts). Videos saved to
`out/rollout_bc.mp4` and `out/rollout_residual.mp4`. Latency reported (p99 ≈ 0.5 ms —
negligible vs the 20 Hz control budget).

---

## Artifacts
- Checkpoints: `out/bc.pt`, `out/residual.pt`
- Plots: `out/bc_diag_outcomes.png`, `out/bc_diag_lift_traces.png`,
  `out/residual_training_curves.png`, `out/residual_ablation_clip.png`
- Videos: `out/rollout_bc.mp4`, `out/rollout_residual.mp4`

## Caveats / honesty
- The residual result is **within 30-rollout sampling noise of BC** (notebook −6.7pp /
  one script run +3.3pp). The defensible claim is "residual recovers BC and is
  well-behaved", NOT "residual beats BC". Reporting the unfavorable notebook run rather
  than the favorable script run is deliberate (no cherry-picking).
- The exact number is run-sensitive: BC normalization stats (full-data in the notebook vs
  train-split in the package) and GPU nondeterminism flip ~2-3 borderline rollouts. The
  qualitative story (bound sweep, ablation, all-expert-data ceiling) is stable across runs.
- Single seed for training (42). Multi-seed bands would quantify the ±noise directly.
- Lean scope: TD3+BC only (no IQL/AWAC comparison).
