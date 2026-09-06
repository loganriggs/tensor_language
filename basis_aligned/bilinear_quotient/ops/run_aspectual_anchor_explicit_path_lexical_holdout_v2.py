#!/usr/bin/env python3
"""Token-alignment-only correction of the frozen prospective path test."""

from __future__ import annotations

import hashlib
from pathlib import Path


SCRIPT_FILE = Path(globals()["__file__"]).resolve()
ROOT = SCRIPT_FILE.parents[1]
V1 = ROOT / "ops/run_aspectual_anchor_explicit_path_lexical_holdout_v1.py"
EXPECTED_V1_SHA256 = "5c324f6162ab4f13ee17e117ff1ab055422eaf298ead0c886b94a225d2025a5c"
STATIC_PREDICATES = {
    "pred_a_native_capability": "all frozen native cells",
    "pred_b_writer_transfer": "fixed two-term writer",
    "pred_c_four_head_path_transfer": "fixed attention5 heads",
    "pred_d_source_identity_transfer": "fixed source bank",
    "pred_e_four_head_compression_transfer": "four versus nine heads",
    "pred_f_exact_coverage": "all frozen rows and arms",
}


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one v1 source occurrence: {old!r}")
    return source.replace(old, new)


def main() -> None:
    payload = V1.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != EXPECTED_V1_SHA256:
        raise RuntimeError(
            f"v1 runner changed: expected={EXPECTED_V1_SHA256} observed={observed}"
        )
    source = payload.decode("utf-8")
    replacements = (
        (
            "circuit_candidate_aspectual_lexical_holdout_v2 as holdout",
            "circuit_candidate_aspectual_lexical_holdout_v5 as holdout",
        ),
        (
            "aspectual_anchor_explicit_path_lexical_holdout_v1.json",
            "aspectual_anchor_explicit_path_lexical_holdout_v2.json",
        ),
        (
            "circuit_candidate_aspectual_lexical_holdout_v2.py",
            "circuit_candidate_aspectual_lexical_holdout_v5.py",
        ),
        (
            "aspectual_anchor_explicit_path_lexical_holdout_v1_result.json",
            "aspectual_anchor_explicit_path_lexical_holdout_v2_result.json",
        ),
        (
            "aspectual_anchor.has_vs_had.explicit_path_lexical_holdout_v1",
            "aspectual_anchor.has_vs_had.explicit_path_lexical_holdout_v2",
        ),
        (
            "4d38531edcf97eed13d2724362a7d17eb2d2a0fbeed00208f92dd3e6028a014e",
            "df2a699d25311db0f20ba1ca82a7ec7aaae3e0c8516b320419a48832a19c14a8",
        ),
        (
            "d4f37373ab52be5faf98fb1576179d659bbded8b4a5b75c7b7d7ec1fb567116a",
            "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859",
        ),
        (
            "1418bfb6e0eb69a788cd11bd1da7b77585bd65cb5868fb7382f946a2072a1a25",
            "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493",
        ),
        (
            "aspectual-anchor-explicit-path-lexical-holdout-v1",
            "aspectual-anchor-explicit-path-lexical-holdout-v2",
        ),
        (
            "aspectual_anchor_explicit_path_lexical_holdout_dryrun_v1",
            "aspectual_anchor_explicit_path_lexical_holdout_dryrun_v2",
        ),
        (
            "aspectual_anchor_explicit_path_lexical_holdout_result_v1",
            "aspectual_anchor_explicit_path_lexical_holdout_result_v2",
        ),
    )
    for old, new in replacements:
        source = replace_once(source, old, new)
    namespace = {
        "__name__": "__main__",
        "__file__": str(SCRIPT_FILE),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, str(SCRIPT_FILE), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
