# Residual Policy Learning on robomimic lift-ph — modular layout

Script/module port of `origin_assignment_takehome.ipynb`, so you can implement
and iterate on each section in a real editor instead of the notebook. The
notebook is left untouched; this package mirrors its sections 1:1.

## Layout

```
10x/
├── residual_lift/             # importable package — one module per notebook section
│   ├── config.py              # MuJoCo backend (EGL), device, dims, paths, out/ artifacts
│   ├── data.py                # dataset resolution (local copy), LiftPHDataset, obs stats
│   ├── section1_bc.py         # BCPolicy + train_bc()                    [TODO]
│   ├── section2_residual.py   # ResidualPolicy, QCritic + train_residual()  [TODO]
│   ├── section3_shield.py     # SafetyShield + build_shield()            [TODO]
│   └── section4_eval.py       # rollout/eval helpers + compare()         [prefilled]
├── scripts/
│   ├── train_bc.py            # -> out/bc.pt
│   ├── train_residual.py      # loads out/bc.pt   -> out/residual.pt
│   └── run_eval.py            # loads both, builds shield, prints comparison + videos
├── data/lift/ph/low_dim_v141.hdf5   # permanent local dataset copy (no re-download)
└── out/                       # checkpoints + rollout videos (was Colab /content/)
```

## Where the notebook's `[TODO]`s live

| Notebook section | Module | What you implement |
|---|---|---|
| Section 1 — BC | `section1_bc.py` → `train_bc()` | the BC training loop |
| Section 2 — Residual | `section2_residual.py` → `ResidualPolicy`, `QCritic`, `train_residual()` | residual net, twin critic, RL update |
| Section 3 — Shield | `section3_shield.py` → `SafetyShield.__call__`, `build_shield()` | per-dim clip + bounds-from-data |

`section4_eval.py` is prefilled — don't edit it; if it errors, the bug is in your
Section 1/2/3 code.

## Running

```bash
conda activate origin10x          # the env built for this assignment
cd 10x

python scripts/train_bc.py        # Section 1  -> out/bc.pt
python scripts/train_residual.py  # Section 2  -> out/residual.pt
python scripts/run_eval.py        # Section 4  -> comparison table + out/*.mp4
```

(Until you fill in the `[TODO]`s, `train_bc.py` raises `NotImplementedError` by
design — that's the placeholder you replace.)

## Notes / changes from the notebook

- **Dataset** loads from the permanent local copy at `data/lift/ph/low_dim_v141.hdf5`;
  it only downloads if no local copy and no robomimic cache exist.
- **Output paths**: the notebook's Colab `/content/{bc,residual}.pt` and
  `/content/rollout_*.mp4` now live under `out/` (see `config.py`).
- **EGL backend** (`MUJOCO_GL=egl`) is set in `config.py`, imported before any
  `robosuite`/`mujoco` import — no per-cell env-var dance needed.
- **Checkpoint round-trips**: `load_bc()` / `load_residual()` reconstruct policies
  from disk so the three stages run as independent processes.
- Policies take `obs_mean` / `obs_std` as constructor args (the notebook read them
  from globals) — the only structural change, to keep modules import-safe.
```
