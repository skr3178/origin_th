#!/usr/bin/env python
"""Task 3 ablation — decision #6: clip delta inside the target Q computation.

Trains the residual twice, identically, except for whether the target action is
clipped to the executable set (residual bounded, sum clipped to [-1,1]):

    ON  (correct)   target action is realizable -> Q stays calibrated.
    OFF (ablation)  target action may exceed [-1,1] -> Q is queried OOD ->
                    value overestimation, delta saturates at the bound.

Overlays q_mean and delta_mag for both runs to out/residual_ablation_clip.png.
Run:  python scripts/ablation_residual.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residual_lift.config import DEVICE, BC_CKPT, OUT_DIR
from residual_lift.section1_bc import load_bc
from residual_lift.section2_residual import train_residual

STEPS = 10000
# Run the ablation at the larger 0.05 bound: the unrealizable-target-action effect
# (decision #6) is most visible when the residual has more room to leave [-1,1].
ABLATION_BOUND = 0.05

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if not os.path.exists(BC_CKPT):
        sys.exit(f"Missing {BC_CKPT}. Run scripts/train_bc.py first.")

    bc = load_bc()
    print("\n=== clip-in-target ON (correct) ===")
    _, hist_on  = train_residual(bc, steps=STEPS, ablate_clip_in_target=False,
                                 delta_bound=ABLATION_BOUND, save=False)
    print("\n=== clip-in-target OFF (ablation) ===")
    _, hist_off = train_residual(bc, steps=STEPS, ablate_clip_in_target=True,
                                 delta_bound=ABLATION_BOUND)

    fig, (axq, axd) = plt.subplots(1, 2, figsize=(12, 4.5))
    axq.plot(hist_on["step"],  hist_on["q_mean"],  label="clip ON (correct)",  c="tab:green")
    axq.plot(hist_off["step"], hist_off["q_mean"], label="clip OFF (ablation)", c="tab:red")
    axq.set_title("q_mean — target-Q overestimation"); axq.set_xlabel("step")
    axq.grid(alpha=0.3); axq.legend()

    axd.plot(hist_on["step"],  hist_on["delta_mag"],  label="clip ON (correct)",  c="tab:green")
    axd.plot(hist_off["step"], hist_off["delta_mag"], label="clip OFF (ablation)", c="tab:red")
    axd.axhline(DELTA_BOUND, ls="--", c="k", lw=0.8, label=f"bound={DELTA_BOUND}")
    axd.set_title("delta_mag — residual saturation"); axd.set_xlabel("step")
    axd.grid(alpha=0.3); axd.legend()

    fig.suptitle("Ablation: clipping delta inside the target Q (decision #6)")
    path = os.path.join(OUT_DIR, "residual_ablation_clip.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"\nFinal q_mean  ON {hist_on['q_mean'][-1]:+.3f}  vs  OFF {hist_off['q_mean'][-1]:+.3f}")
    print(f"Ablation plot saved: {path}")
