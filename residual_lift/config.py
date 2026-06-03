"""Shared configuration: MuJoCo backend, device, dims, and file paths.

IMPORT THIS FIRST. Setting MUJOCO_GL / PYOPENGL_PLATFORM must happen before any
`import mujoco` / `import robosuite` anywhere in the process, so every other
module in this package imports `config` before touching the simulator.
"""
import os

# CRITICAL: must be set before importing mujoco or robosuite.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

# --- Device ---------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    print("[config] WARNING: CUDA not available — falling back to CPU (slow). "
          "The original notebook asserts a GPU.")

# --- Observation / action layout -----------------------------------------
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
# OBS_DIM = object(10: cube_pos[3] + cube_quat[4] + gripper_to_cube_pos[3])
#         + eef_pos(3) + eef_quat(4) + gripper_qpos(2 — Panda has 2 finger joints) = 19
OBS_DIM = 19
ACT_DIM = 7

# --- Paths ----------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
REPO_10X  = os.path.dirname(_HERE)                       # .../sensor2sensor/10x
DATA_DIR  = os.path.join(REPO_10X, "data")               # local dataset copy lives here
OUT_DIR   = os.path.join(REPO_10X, "out")                # checkpoints + videos (was /content)
os.makedirs(OUT_DIR, exist_ok=True)

# Artifacts (replace the notebook's /content/* Colab paths).
BC_CKPT        = os.path.join(OUT_DIR, "bc.pt")
RESIDUAL_CKPT  = os.path.join(OUT_DIR, "residual.pt")
BC_VIDEO       = os.path.join(OUT_DIR, "rollout_bc.mp4")
RESIDUAL_VIDEO = os.path.join(OUT_DIR, "rollout_residual.mp4")
