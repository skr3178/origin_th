# origin10x environment — exported libraries

The conda env used for this assignment (`origin10x`), exported three ways:

| File | What it is | Use when |
|---|---|---|
| `origin10x_requirements.txt` | `pip freeze` (exact pip versions) | recreate with `pip` in any Python 3.10 env |
| `origin10x_environment.yml` | `conda env export` (with build strings) | exact recreate on the **same OS/arch** (Linux x86-64) |
| `origin10x_environment_nobuilds.yml` | `conda env export --no-builds` | cross-platform conda recreate (no build pins) |

## Key pins
Python 3.10 · **torch 2.12.0+cu130** (CUDA 13.0) · robomimic 0.3.0 · robosuite 1.4.1 ·
mujoco 3.9.0 · numpy 2.2.6 · matplotlib 3.10.9 · h5py 3.16.0 · imageio 2.37.3 ·
nbconvert 7.17.1 · ipykernel 7.2.0.

## ⚠️ Reproducibility gotcha — the torch CUDA build
`pip freeze` records `torch==2.12.0` but the **actual build is `2.12.0+cu130`**
(the `+cu130` CUDA tag is dropped on export). Installing `torch==2.12.0` plainly will
grab the wrong (default) build. Install torch/torchvision from the CUDA index explicitly:

```bash
conda create -n origin10x python=3.10 -y
conda activate origin10x
# CUDA 13.0 build (matches this machine's driver; pick the cuXXX index for your GPU):
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
# then the rest:
pip install -r origin10x_requirements.txt
```

(Any recent CUDA build — cu121/cu124/cu130 — works on the RTX 3060; the exact `+cu130`
isn't required, just a CUDA-enabled torch.)

## Minimal alternative
This whole env is heavier than the assignment needs. The notebook's own setup cell only
requires: `robomimic==0.3.0 robosuite==1.4.1 torch h5py imageio imageio-ffmpeg tqdm`
(+ `matplotlib` for the plots). The full export above is for exact reproduction.
