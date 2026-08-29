#!/usr/bin/env python3
"""Fresh v2 namespace for the terminal-copy rows after v1 failed pre-model.

The scientific selection plan is unchanged.  V2 changes only registry replay: it can
exclude an authority's missing row output when its own terminal failure proves that no
rows, manifest, or receipt ever materialized.  All other missing references remain
fatal.
"""

from __future__ import annotations

import json
from pathlib import Path

import prepare_terminal_copy_induction_v1_rows as base
import terminal_copy_registry_recovery_v2 as recovery


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
FREEZER = Path(__file__).resolve()
TEST = HERE / "test_prepare_terminal_copy_induction_v2_rows.py"
RECOVERY = HERE / "terminal_copy_registry_recovery_v2.py"
RECOVERY_ADDENDUM = HERE / "TERMINAL_COPY_INDUCTION_V2_ROW_RECOVERY_ADDENDUM.md"
V1_FAILURE = BQ / "terminal_copy_induction_v1_rows_failure.json"
FAILED_AUTHORITY = HERE / "gauge_transport_triangle_unique_rows_v1_authority.json"
FAILED_AUTHORITY_FAILURE = HERE / "gauge_transport_triangle_unique_rows_v1_failure.json"
FAILED_AUTHORITY_RUNNER = HERE / "freeze_gauge_transport_triangle_unique_rows_v1.py"
FAILED_AUTHORITY_TEST = HERE / "test_freeze_gauge_transport_triangle_unique_rows_v1.py"
FAILED_AUTHORITY_PREREG = HERE / "GAUGE_TRANSPORT_TRIANGLE_UNIQUE_ROWS_V1_PREREGISTRATION.md"

CACHE = BQ / ".rowcache_terminal_copy_induction_v2"
RECEIPT = BQ / "terminal_copy_induction_v2_rows_receipt.json"
FAILURE = BQ / "terminal_copy_induction_v2_rows_failure.json"
LOCK = Path("/workspace/runs/.terminal_copy_induction_v2_rows.lock")
AUDIT = HERE / "terminal_copy_induction_v2_rows_audit.json"


def configure() -> None:
    original_sources = tuple(base.SOURCE_PATHS)
    base.FREEZER = FREEZER
    base.TEST = TEST
    base.CACHE = CACHE
    base.RECEIPT = RECEIPT
    base.FAILURE = FAILURE
    base.LOCK = LOCK
    base.AUDIT = AUDIT
    base.RECEIPT_KIND = "terminal_copy_induction_v2_rows"
    base.FAILURE_KIND = "terminal_copy_induction_v2_rows_failure"
    base.SOURCE_PATHS = tuple(dict.fromkeys((
        FREEZER, TEST, RECOVERY, RECOVERY_ADDENDUM, V1_FAILURE,
        FAILED_AUTHORITY, FAILED_AUTHORITY_FAILURE, FAILED_AUTHORITY_RUNNER,
        FAILED_AUTHORITY_TEST, FAILED_AUTHORITY_PREREG, *original_sources,
    )))
    base.load_prior_registry = recovery.load_registry_exclusions
    base.verify_prior_snapshot = recovery.verify_snapshot


def freeze():
    configure()
    return base.freeze()


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
