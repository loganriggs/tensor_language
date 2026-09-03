#!/usr/bin/env python3
"""Frozen launcher for the rung-521 whole-attention8 Stage-A power test.

The managed instrument-only smoke produced an attention-write edit RMS of
46.6581916809082 without retaining any task or circuit outcome.  This launcher
freezes the registered liveness floor at one tenth of that value and calls the
separately audited Stage-A executable.  Stage A has no optimizer or backward
path and stops regardless of whether its power predicate passes.
"""

# BQGATE: EXPERIMENT
# pred_a: whole-attention8 effects are material and selective in both FIT halves
# pred_b: independent donor ensembles reproduce signed token effects in both directions
# pred_c: the 32-circuit effect fingerprint reproduces across halves above its permutation null

from __future__ import annotations

import hashlib
import os
from pathlib import Path


REGISTERED_PREDICTIONS = {
    "pred_a": "whole-attention8 effects are material and selective in both FIT halves",
    "pred_b": "independent donor ensembles reproduce signed token effects in both directions",
    "pred_c": "the 32-circuit effect fingerprint reproduces above its permutation null",
}

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
EDIT_RMS_FLOOR = 4.66581916809082
FROZEN_FILES = {
    ROOT / "attention8_shared_private_das_rung521_gpu_smoke.json":
        "ab4b079843a8b4b1e8bd0082c8b9d29cd8552253c2468bd2de851a4f46a7e9c3",
    ROOT / "attention8_shared_private_das_rung521_preflight.json":
        "42639d35ef6317104c6e0e684aeb00cb4c550df77d496733bcfe8be790fed650",
    ROOT / "ops/attention8_shared_private_das_rung521.py":
        "d5ca962c16cd8f454adac79916a9cf3272b91debac0d27ebba2ce77804fb9ebd",
    POLY / "ATTENTION8_SHARED_PRIVATE_DAS_RUNG521_PREREGISTRATION.md":
        "e40ca9654485d8fcc04dd09e0b86628fa633e98d97c0b444c6661f56f73461de",
    POLY / "ATTENTION8_SHARED_PRIVATE_DAS_RUNG521_PREFLIGHT_ADDENDUM.md":
        "5758cc99e59050d80c1eb94071a4be3f595196355ea9c3e4900ea747059eaa09",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frozen_files() -> None:
    for path, expected in FROZEN_FILES.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"rung521 frozen launcher dependency changed: {path}")


if os.environ.get("BQLIB_DRYRUN") == "1":
    _validate_frozen_files()
    print(
        "DRYRUN OK: rung521 Stage A; smoke-frozen edit RMS floor "
        f"{EDIT_RMS_FLOOR:.15g}; 2,698 inference forwards; zero backwards",
        flush=True,
    )
    raise SystemExit(0)


import attention8_shared_private_das_rung521 as stage_a  # noqa: E402


def main() -> dict:
    _validate_frozen_files()
    if os.environ.get("BQLIB_NO_MODEL") == "1":
        raise RuntimeError("BQLIB_NO_MODEL forbids the rung521 Stage-A science run")
    return stage_a.main(["--edit-rms-floor", repr(EDIT_RMS_FLOOR)])


if __name__ == "__main__":
    main()
