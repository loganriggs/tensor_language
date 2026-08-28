#!/usr/bin/env python3
"""Prospective same-run identity repair for the S1714 exact middle-MLP arm.

The completed ksweep2 artifact is retained as a failed identity check: its supposedly
exact hook omitted ``Down_bias``.  This narrow run corrects that source and compares
two arms using one shared compiled program and the same covered rows:

``exact``
    Recompute every MLP4--15 write as
    ``Down(Left(z)*Right(z)) + Down_bias`` inside its forward hook.

``exempt``
    Remove only those twelve hooks, leaving MLP4--15 live.  Every fitted attention
    and non-middle MLP component is the same Python tensor object in both arms.

Registered identity gate: pooled CE differs by at most ``1e-7``, per-row covered
loss sums by at most ``1e-5`` (float32 CE accumulation), counts are bit-identical,
and an exact-arm replay is bit-identical in row sums/counts.  The legacy 67.553%
ceiling is descriptive only; it is no longer the identity denominator.

This run fits no feature sweep and chooses no scientific candidate.  It exists only
to validate the corrected algebraic construction before the empirical ridge curve is
used as family evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

import mid_band_feature_ksweep2 as ks


HERE = Path(__file__).resolve().parent
OUT = HERE / "mid_band_feature_exact_identity_results.json"
SOURCE = Path(__file__).resolve()
KS_SOURCE = HERE / "mid_band_feature_ksweep2.py"
POOLED_TOL = 1e-7
ROW_SUM_TOL = 1e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit() -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
    ).strip()
    if len(commit) != 40:
        raise RuntimeError("source commit is malformed")
    return commit


@torch.no_grad()
def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"create-only identity artifact already exists: {OUT}")
    started = time.time()
    evaluation = ks.load(ks.EVAL_ROWS)
    mask_rows = ks.load(ks.MASK_ROWS)
    seen = ks.seen_mask(mask_rows)
    ks.SEENREF["m"] = seen
    del mask_rows
    fit = ks.load(ks.FIT_ROWS)

    ks.CFG["v1"] = None
    ks.V1P.pop("W", None)
    ks.FEAT.update({"k": ks.DH, "sel": {}, "mode": "exact"})
    program = ks.compile_stack(fit, ("mlp", "attn"))
    expected_exact = {("mlp", layer) for layer in ks.MID}
    observed_exact = {
        key for key, value in program.items()
        if isinstance(value, tuple) and value == ("exact_mlp", key[1])
    }
    if observed_exact != expected_exact:
        raise RuntimeError(
            f"exact middle hook set changed: {observed_exact} != {expected_exact}"
        )
    exempt_program = {
        key: value for key, value in program.items() if key not in expected_exact
    }
    shared_ids = {
        str(key): id(value) for key, value in exempt_program.items()
    }
    if any(program[key] is not value for key, value in exempt_program.items()):
        raise RuntimeError("exact/exempt programs do not share non-middle objects")

    exempt_ce, exempt_sums, exempt_counts = ks.ce_rows(
        evaluation, seen, hooks=ks.install(exempt_program),
    )
    exact_ce, exact_sums, exact_counts = ks.ce_rows(
        evaluation, seen, hooks=ks.install(program),
    )
    replay_ce, replay_sums, replay_counts = ks.ce_rows(
        evaluation, seen, hooks=ks.install(program),
    )

    pooled_difference = abs(exact_ce - exempt_ce)
    row_sum_difference = float((exact_sums - exempt_sums).abs().max())
    counts_equal = bool(torch.equal(exact_counts, exempt_counts))
    replay_sums_equal = bool(torch.equal(exact_sums, replay_sums))
    replay_counts_equal = bool(torch.equal(exact_counts, replay_counts))
    replay_ce_equal = exact_ce == replay_ce
    gates = {
        "pooled_ce_abs_difference_le_1e-7": pooled_difference <= POOLED_TOL,
        "max_row_sum_abs_difference_le_1e-5": row_sum_difference <= ROW_SUM_TOL,
        "counts_bit_identical": counts_equal,
        "exact_replay_row_sums_bit_identical": replay_sums_equal,
        "exact_replay_counts_bit_identical": replay_counts_equal,
        "exact_replay_ce_bit_identical": replay_ce_equal,
    }
    result = {
        "status": "pass" if all(gates.values()) else "failed_identity",
        "question": "corrected exact middle-MLP hooks equal leaving MLP4-15 live",
        "source_commit": source_commit(),
        "source_sha256": sha256(SOURCE),
        "ksweep_source_sha256": sha256(KS_SOURCE),
        "rows": {
            "fit": ks.FIT_ROWS,
            "mask": ks.MASK_ROWS,
            "evaluation": ks.EVAL_ROWS,
            "coverage": "same frozen seen-token mask and covered positions as ksweep2",
        },
        "construction": "Down(Left(z)*Right(z)) + Down_bias",
        "comparison": {
            "exact_ce": exact_ce,
            "exempt_ce": exempt_ce,
            "exact_replay_ce": replay_ce,
            "pooled_ce_abs_difference": pooled_difference,
            "max_row_sum_abs_difference": row_sum_difference,
            "shared_nonmiddle_object_count": len(shared_ids),
            "shared_nonmiddle_object_ids_sha256": hashlib.sha256(
                json.dumps(shared_ids, sort_keys=True).encode()
            ).hexdigest(),
        },
        "gates": gates,
        "legacy_ksweep2_exact_arm": {
            "status": "invalid_identity_omitted_Down_bias",
            "artifact": str(HERE / "mid_band_feature_ksweep2_results.json"),
            "observed_ceiling": 0.68059,
            "legacy_target_ceiling": ks.S1703_BAND_EXEMPT_CEILING,
            "use": "failure provenance only; never an identity or promoted ceiling",
        },
        "runtime_s": round(time.time() - started, 1),
    }
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps(result, indent=2), flush=True)
    if not all(gates.values()):
        raise RuntimeError(f"corrected exact identity failed: {gates}")


if __name__ == "__main__":
    main()
