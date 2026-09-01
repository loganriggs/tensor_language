#!/usr/bin/env python3
"""Optional static preflight for new rung scripts (ops lane, advisory only).

Targets the three instrument-clause failure classes that cost reruns on
2026-09-01 (rungs 419 and 424: 5 extra GPU runs + audit cycles):

  1. HASH CLASS   -- comparing a raw-byte digest against a rows receipt's
                     semantic `tensor_sha256` field (never equal).
  2. RETRACTION   -- optimizing bases with Adam then gating pred_a on
                     `_orth_error(...) <= tol` without a QR retraction.
  3. ABS-VS-REL   -- absolute `*_max_abs` tolerance bars on float32
                     replay quantities (prefer relative-squared bars).
  4. CONTROL WIN  -- absolute numeric windows on shuffle/permutation
                     control statistics (rungs 428/429/432: three window
                     mis-derivations; prefer matched-control EXCESS).

Usage: python ops/preflight.py ops/<script>.py   (warnings only; exit 0)
Never a gate: registration bars are the registrant's; this only warns.
"""
import re
import sys
from pathlib import Path


def check(path: Path) -> list[str]:
    text = path.read_text()
    warnings = []
    raw_digest = re.search(r"numpy\(\)\.tobytes\(\)|_digest\(", text)
    semantic = re.search(r"['\"]tensor_sha256['\"]", text)
    uses_semantic_fn = "rows_life.base.tensor_sha256" in text
    if raw_digest and semantic and not uses_semantic_fn:
        warnings.append(
            "HASH CLASS: file compares against receipt 'tensor_sha256' but "
            "computes raw-byte digests; use rows_life.base.tensor_sha256 "
            "(rung 424 cost 2 reruns on this).")
    optimizes = re.search(r"Adam|_optimize\(", text)
    orth_gate = re.search(r"_orth_error\([^)]*\)\s*<=", text)
    retracts = re.search(r"linalg\.qr|retraction", text)
    if optimizes and orth_gate and not retracts:
        warnings.append(
            "RETRACTION: bases are optimized and pred gates on _orth_error "
            "but no QR retraction found (rung 424 cost 1 rerun on this).")
    control_window = any(
        re.search(r"control", line)
        and re.search(r"(<=?|>=?)\s*\.\d|\.\d+\s*<", line)
        for line in text.splitlines())
    excess = re.search(r"excess[_a-z]*\s*=|control_gap|minus_control", text)
    if control_window and not excess:
        warnings.append(
            "CONTROL WINDOW: absolute numeric bounds on a shuffle/control "
            "statistic; three window mis-derivations on 2026-09-01 "
            "(428/429/432) -- prefer matched-control excess statistics.")
    abs_replay = re.search(r"replay[a-z_]*max_abs[\"'\]]*\s*<=\s*[0-9.e-]+", text)
    rel_available = re.search(r"relative_squared", text)
    if abs_replay and not rel_available:
        warnings.append(
            "ABS-VS-REL: absolute max-abs bar on a replay quantity with no "
            "relative-squared companion; float32 magnitudes broke this bar "
            "on rung 419 (1 rerun).")
    return warnings


def main() -> None:
    bad = 0
    for arg in sys.argv[1:]:
        for warning in check(Path(arg)):
            bad += 1
            print(f"{arg}: WARN {warning}")
    if not bad:
        print("preflight: no findings")


if __name__ == "__main__":
    main()
