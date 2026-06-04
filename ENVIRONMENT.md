# Environment

The exact Python environment used for every run in this repo (BC, residual, all
ablations, BC-RNN). Conda env name: **`s2s_10x`**, **Python 3.12**, CUDA build
(`torch==2.11.0+cu128`, verified on an RTX 3060).

## Files

| File | What it is | Use when |
|---|---|---|
| `requirements.txt` | `pip freeze` — full pinned list (277 pkgs), the source of truth | recreating with pip/venv |
| `environment.yml` | `conda env export` — full conda+pip snapshot | recreating the exact conda env |
| `environment.from-history.yml` | only the explicitly-requested conda packages (python 3.12; everything else is pip) | a clean, portable starting point |

## Recreate

**Conda (closest to the original):**
```bash
conda env create -n s2s_10x -f environment.yml
conda activate s2s_10x
```

**pip / venv:**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Minimal (just the pinned essentials, lets the rest resolve):**
```bash
pip install robomimic==0.3.0 robosuite==1.4.1 torch h5py imageio imageio-ffmpeg tqdm matplotlib scipy
```
This is exactly what the notebook's setup cell installs.

## Key pins

`robomimic==0.3.0` · `robosuite==1.4.1` (pinned to the dataset's collection version —
matches `low_dim_v141.hdf5`) · `torch==2.11.0+cu128` · `mujoco==3.9.0` · `numpy==2.0.2` ·
`h5py==3.16.0` · `scipy==1.17.1` · `matplotlib==3.10.9` · `imageio==2.37.3`.

Rendering uses the EGL MuJoCo backend (`MUJOCO_GL=egl`) for offscreen rollouts.
