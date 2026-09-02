"""DETERMINISTIC-KERNEL CURE PROBE (instrument, preregistered): does pinning
the cuBLAS workspace restore cross-process reproducibility on the hook-laden
474 pathway -- and is kernel/workspace selection therefore the breach
mechanism?

Design: run ops/subtractive_replication_probe.py TWICE as fresh subprocesses
with CUBLAS_WORKSPACE_CONFIG=":4096:8" (arms DET1, DET2), snapshotting the
plain queued probe's receipt (arm RUN2, no pinning, runs before this wrapper
in the queue) and comparing against the frozen RUN1 constants (08:12
process, recorded in HOURLY_STRATEGIC_REVIEW_2026-09-02_0830.md).

Frozen predictions (keys in code below): pred a -- both DET children are
instrument-exact (their own three pred fields: bundle-reproduction verdict
as measured, replay/empty exact, counts exact -- note the child's
reproduces-bundle pred may be FALSE, that is the breach, not a failure
here). pred b -- THE CURE: DET1 and DET2 agree exactly (0.0 difference) on
fresh_vs_bundle_max and every per-subset value; workspace pinning makes the
pathway cross-process reproducible. pred c -- THE MECHANISM SIGNATURE: the
unpinned RUN2 receipt differs from frozen RUN1 by at least .001 nat on some
per-subset value (per-process variation, H-A), OR differs from DET1
likewise.  Null: pred b fails -- workspace pinning is NOT the mechanism and
no cure exists at this knob; H-A weakens toward the mixed hypothesis.

Price: two child processes (model load + 35 forwards each), ~3-4 min GPU,
0 deployed parameters; already-opened objects only; the child performs all
scientific hygiene (hash checks) itself.
"""
# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BQ = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
CHILD = BQ / "ops/subtractive_replication_probe.py"
CHILD_OUT = BQ / "subtractive_replication_probe_results.json"
OUT = BQ / "det_replication_probe_results.json"
SCRATCH = BQ / "runlogs"
RUN1 = {  # frozen 08:12 values (HOURLY_STRATEGIC_REVIEW_2026-09-02_0830.md)
    "m8": 0.08405804634094238, "m9": 0.06967806816101074,
    "m12": 0.03833127021789551, "m8+m9": 0.05142807960510254,
    "m8+m12": 0.0832054615020752, "m9+m12": 0.04619002342224121,
    "m8+m9+m12": 0.06456136703491211,
}


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert CHILD.exists(), "child probe must be on disk"
        out = subprocess.run(
            [sys.executable, str(CHILD)],
            env={**os.environ, "BQLIB_DRYRUN": "1", "BQLIB_NO_MODEL": "1"},
            capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stdout + out.stderr
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "det_replication_probe", "model_loaded": False}))
        return
    started = time.time()
    run2 = None
    if CHILD_OUT.exists():
        run2 = json.loads(CHILD_OUT.read_text())
        shutil.copy(CHILD_OUT, SCRATCH / "replication_probe_run2_snapshot.json")
    children = []
    for arm in ("det1", "det2"):
        out = subprocess.run(
            [sys.executable, str(CHILD)],
            env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            raise RuntimeError(f"{arm} child failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
        receipt = json.loads(CHILD_OUT.read_text())
        shutil.copy(CHILD_OUT, SCRATCH / f"replication_probe_{arm}_snapshot.json")
        children.append(receipt)
    det1, det2 = children
    per1 = det1["fresh_vs_bundle_per_subset_max_abs_nat"]
    per2 = det2["fresh_vs_bundle_per_subset_max_abs_nat"]
    det_pair_diff = max(abs(per1[k] - per2[k]) for k in per1)
    det_vs_max = abs(det1["fresh_vs_bundle_max_abs_nat"] - det2["fresh_vs_bundle_max_abs_nat"])
    child_ok = all(
        r["pred_b_replay_empty_exact"] and r["pred_c_counts_exact"] for r in children)
    run2_vs_run1 = (max(abs(run2["fresh_vs_bundle_per_subset_max_abs_nat"][k] - RUN1[k])
                        for k in RUN1) if run2 else None)
    run2_vs_det1 = (max(abs(run2["fresh_vs_bundle_per_subset_max_abs_nat"][k] - per1[k])
                        for k in RUN1) if run2 else None)
    det1_vs_run1 = max(abs(per1[k] - RUN1[k]) for k in RUN1)
    pred_a = bool(child_ok)
    pred_b = bool(det_pair_diff == 0.0 and det_vs_max == 0.0)
    pred_c = bool((run2_vs_run1 is not None and run2_vs_run1 >= .001)
                  or (run2_vs_det1 is not None and run2_vs_det1 >= .001))
    result = {
        "status": "complete", "rung": "det_replication_probe",
        "run2_snapshot_present": run2 is not None,
        "det_pair_max_abs_diff": det_pair_diff,
        "det_pair_bundle_max_diff": det_vs_max,
        "det1_per_subset": per1, "det2_per_subset": per2,
        "run2_vs_frozen_run1_max_abs_diff": run2_vs_run1,
        "run2_vs_det1_max_abs_diff": run2_vs_det1,
        "det1_vs_frozen_run1_max_abs_diff": det1_vs_run1,
        "det1_reproduces_bundle": det1["pred_a_474_code_reproduces_bundle"],
        'pred_a_children_instrument_exact': pred_a,
        'pred_b_workspace_pinning_cures': pred_b,
        'pred_c_unpinned_varies_per_process': pred_c,
        "strong_null": bool(not pred_b),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(f"pred_a={pred_a} pred_b={pred_b} pred_c={pred_c} "
          f"det_pair_diff={det_pair_diff:.3e} run2_vs_run1={run2_vs_run1} "
          f"det1_vs_run1={det1_vs_run1:.4f} ({result['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
