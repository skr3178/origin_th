"""Residual policy learning on robomimic lift-ph.

Modular port of `origin_assignment_takehome.ipynb`. One module per notebook
section:

    config           shared constants, device, paths, output locations
    data             dataset resolution + LiftPHDataset + normalization stats
    section1_bc      Section 1 — BC training            [TODO]
    section2_residual Section 2 — Residual policy        [TODO]
    section3_shield  Section 3 — Safety shield           [TODO]
    section4_eval    Section 4 — Final evaluation        [prefilled]

Run the pipeline via the scripts in ../scripts/:
    python scripts/train_bc.py
    python scripts/train_residual.py
    python scripts/run_eval.py
"""
