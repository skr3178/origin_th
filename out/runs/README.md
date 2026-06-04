# out/runs/ — archived run snapshots

`out/` is the **live** output folder that `writeup.md` and the notebook reference
(via `out/*.png` links). To avoid losing results when re-running, each meaningful run
is **copied** into a snapshot folder here. `out/` itself is never moved, so the
writeup image links keep working; this folder is the history.

**Naming:** short + descriptive of the run's defining config, e.g. `main_b0.005`
(main BC→residual→eval pipeline, residual delta_bound=0.005). Future runs that vary a
knob get their own folder, e.g. `b0.010_seed42`, `b0.005_seed7`, `iql_baseline`.

## Snapshots

- **`main_b0.005/`** — the shipped submission run: frozen BC (86.7%), TD3+BC residual
  at delta_bound=0.005, ablation (clip-in-target), final eval (BC vs residual+shield,
  0.867 vs 0.800), Task-2 failure videos, and the BC failure-mode-vs-duration sweep.
