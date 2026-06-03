#!/usr/bin/env python
"""Entrypoint: final evaluation (Section 4) — BC vs Residual+Shield.

Loads out/bc.pt and out/residual.pt, builds the shield, runs 30 rollouts each,
saves videos to out/, and prints the comparison table.
Run:  python scripts/run_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from residual_lift.config import DEVICE, BC_CKPT, RESIDUAL_CKPT
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import load_residual
from residual_lift.section3_shield import build_shield
from residual_lift.section4_eval import compare

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    for ckpt, script in [(BC_CKPT, "train_bc.py"), (RESIDUAL_CKPT, "train_residual.py")]:
        if not os.path.exists(ckpt):
            sys.exit(f"Missing {ckpt}. Run scripts/{script} first.")
    bc_policy = load_bc()
    residual  = load_residual(bc_policy)
    shield    = build_shield()
    compare(bc_policy, residual, shield, n_rollouts=30, seed=42)
