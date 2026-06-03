"""Section 3 — Safety shield.

A minimal, production-flavoured shield: per-dimension clipping to bounds learned
from the training data, with a margin. (A real shield would also rate-limit and
handle NaNs; clip-only is in scope here.)

Decisions (defended in notes.md):
  1. Margin — bounds = per-dim demo-action [min, max] expanded by MARGIN_FRAC of the
     per-dim range, then clamped to the env's hard [-1, 1]. The margin gives the
     learned residual a little head-room beyond exactly-seen actions (so we don't
     clip valid in-distribution corrections) while still catching gross outliers.
  2. Per-dimension (not a global L2 norm) — the action dims are heterogeneous: dims
     0/2/6 use the full +-1 range while the rotation dims 3/4/5 barely move
     (std ~0.02-0.08). A single L2 ball would either over-constrain the big dims or
     leave the small dims unbounded; per-dim bounds respect each dim's own scale.
"""
import numpy as np

from .data import LiftPHDataset, read_mask

MARGIN_FRAC = 0.05   # expand each dim's data range by 5% of its span


class SafetyShield:
    """Clip-only shield. Per-dimension bounds from data with margin."""
    def __init__(self, act_low, act_high):
        self.low  = np.asarray(act_low,  dtype=np.float32)
        self.high = np.asarray(act_high, dtype=np.float32)

    def __call__(self, action, prev_action=None):
        return np.clip(action, self.low, self.high)


def build_shield(margin_frac=MARGIN_FRAC, verbose=True):
    """Compute per-dim bounds from the train-split demo actions (+ margin) and
    return a configured SafetyShield."""
    train_actions = LiftPHDataset(demo_keys=read_mask("train")).actions.numpy()
    lo = train_actions.min(axis=0)
    hi = train_actions.max(axis=0)
    span = hi - lo
    # Expand by the margin, then clamp to the env's hard action limits.
    act_low  = np.clip(lo - margin_frac * span, -1.0, 1.0)
    act_high = np.clip(hi + margin_frac * span, -1.0, 1.0)

    if verbose:
        np.set_printoptions(precision=3, suppress=True)
        print(f"shield bounds low:  {act_low}")
        print(f"shield bounds high: {act_high}")
    return SafetyShield(act_low, act_high)
