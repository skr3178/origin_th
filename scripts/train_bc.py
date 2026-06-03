#!/usr/bin/env python
"""Entrypoint: train the BC backbone (Section 1) and save out/bc.pt.

Run from anywhere:  python scripts/train_bc.py
"""
import os
import sys

# Put 10x/ on the path so `import residual_lift` works regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from residual_lift.config import DEVICE
from residual_lift.section1_bc import train_bc

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    train_bc()
