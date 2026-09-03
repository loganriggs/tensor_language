# Addendum to RADIAL_GAUGE_MAP_PROBE_PREREGISTRATION.md — device move to CUDA (lane 1)

Registered 2026-09-03 19:58Z (box clock), BEFORE any GPU number exists. Parent prereg sha
7a5bfdc9a17d504c109d47d30c5925e6a118169191b92b7c438e0ba2fd3eaae9 is unchanged and stays frozen in the script.

## Why
The CPU copy (ops/radial_gauge_map_probe.py, lane 2, started 19:38Z) was measured at 200–420 s per 64-doc arm under box load
30 (its `R` import set 16 torch threads, ignoring lane 2's 4-thread cap — fixed 19:57Z in mlp_in_situ_usage_rank_map_probe.py).
At that rate the 108 arms need 4–7 h and block lane 2's FIFO for Codex's short audits. Lane 1 (the GPU) has been EMPTY since
19:24Z. The same 4,680 doc-forwards take minutes on the 5090. The CPU copy is ABORTED (kill, exit code recorded by bqrunner2)
once the CUDA copy is enqueued; none of its arm numbers are used. The only CPU numbers seen before this addendum: baseline stage
and attn0 DROP_RADIAL .00076, attn0 SCALE_RADIAL_2 .00083, mlp0 DROP_RADIAL .0249 — none enters pred_b (mlp2–15), pred_c (mlp17)
or pred_d (mlp1–3).

## What changes (and only this)
- Script: ops/radial_gauge_map_probe_gpu.py — byte-identical arms, sites, split, bars, nulls, eval docs 96–159, chunk 8; the
  model, rope tables, mask and rows live on `cuda`; the script raises if CUDA is absent (no silent fallback).
- pred_a(i) cross-device baseline: the parent bar "baseline 3.1124951 ± 1e-4" was written for a same-device (CPU float32)
  reproduction of §2696. Across devices the program's standing CUDA tolerance applies: |baseline_gpu − 3.1124951| ≤ 0.015
  (`xdev_tol`; ~0.003 wobble expected). The receipt ALSO records whether the 1e-4 same-device bar would have held
  (`same_device_baseline_abs_diff_le_1e-4`), so nothing is hidden.
- pred_a(ii) identity arms ≤ 1e-4 (same device, unchanged) and pred_a(iii) mlp1 DROP_RADIAL reproduces §2702's .0079 within
  .003 (unchanged; a CE-added difference, both terms on the same device).
- pred_b, pred_c, pred_d, all nulls: unchanged.
- Price: 4,680 GPU doc-forwards, ~5–10 min on lane 1 (was 4,680 CPU forwards on lane 2).
