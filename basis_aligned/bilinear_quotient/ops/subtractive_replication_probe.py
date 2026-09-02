"""SUBTRACTIVE REPLICATION PROBE (instrument, non-scoring): does rung 474's
OWN collect_window, called byte-for-byte today, reproduce its own bundle?

Context: the projector rung's singleton bridge failed uniformly (.11-.20
nat); the link diagnostic passed L1-L4 (live==captured 0.0; both update
forms reach absent; projector==subtractive damage 1e-6) and isolated L5:
freshly recomputed subtractive damage differs from the 05:57 rung474 bundle
by .028-.036 nat.  CPU checks: the 472 and 474 bundles agree at exactly 0.0,
no parent-chain file changed since 05:57, and 474's receipt recorded its own
fresh-vs-472-bundle singleton error as 0.0 at run time.

This probe calls sub474.collect_window ITSELF (no copied code) on the first
code_validation batch (truncated selection, full arm sequence: native,
replay, absent, both sources, empty, both slots, all 7 subsets) and compares
every produced effect to the bundle at the same coordinates.

Frozen forks: pred_a (474 code reproduces bundle) TRUE (max abs <= 1e-6)
=> today's physics matches 05:57 and my projector rung's COPIED collector
deviates -- diff the copy next, no program alarm.  FALSE with error ~.03-.2
=> rung 474's own frozen code no longer reproduces its own receipt on
unchanged data/weights/env -- a temporal reproducibility alarm that Codex
must see before any further cross-session bundle comparison is trusted.
pred_b: replay and empty-patch checks 0 as in 474.
pred_c: 35 forwards, patch calls formula-exact for one batch.
No null; nothing scientific is licensed either way.  Price: 35 forwards,
<120s, 0 deployed parameters; already-opened objects only.
"""
# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "subtractive_replication_probe_results.json"
for _p in (ROOT, ROOT / "basis_aligned/polynomial_causal", ROOT / "basis_aligned/qk_mdl", BQ, OPS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import equality_query_subtractive_factorial_rung474 as sub474  # noqa: E402
from receipt import dump  # noqa: E402

WINDOW = "code_validation"
SUB_BUNDLE = BQ / "equality_query_subtractive_factorial_rung474_per_token.pt"


def main():
    (roles, scale, old_effects, selections, old_position,
     old_factorial, metadata) = sub474.validate_inputs()
    role = next(role for name, role, _, _ in sub474.WINDOWS if name == WINDOW)
    payload, _ = roles[role]
    selection = selections[WINDOW]
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, extra) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append(output_index)
    first_doc = min(by_doc)
    keep = [i for i, (doc, _, _) in enumerate(coordinates)
            if first_doc <= doc < first_doc + sub474.BATCH]
    truncated = dict(selection)
    truncated["coordinates"] = [coordinates[i] for i in keep]
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert "rows" in payload and len(keep) >= 2
        print(json.dumps({"status": "dry_run_passed",
                          "rung": "subtractive_replication_probe",
                          "model_loaded": False, "kept_coordinates": len(keep)}))
        return
    started = time.time()
    model, _ = sub474.facade.load_bilin18()
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    window, reconstruction = sub474.collect_window(
        model, payload, scale, truncated, audit_totals, replay,
    )
    bundle = torch.load(SUB_BUNDLE, map_location="cpu", weights_only=True)
    bundle_effects = bundle["windows"][WINDOW]["effects"].double()
    fresh = window["effects"].double()
    diff = (fresh - bundle_effects[:, :, keep]).abs()
    forwards = sum(row.get("forwards", 0) for row in audit_totals.values())
    per_subset = {sub474.SUBSET_NAMES[j]: float(diff[:, j, :].max())
                  for j in range(len(sub474.SUBSETS))}
    max_abs = float(diff.max())
    pred_a = bool(max_abs <= 1e-6)
    pred_b = bool(replay["max_abs"] == 0.0 and window["empty_patch_max_abs"] == 0.0)
    pred_c = bool(forwards == sub474.FORWARDS_PER_BATCH)
    result = {
        "status": "complete", "rung": "subtractive_replication_probe",
        "window": WINDOW, "kept_coordinates": len(keep),
        "fresh_vs_bundle_max_abs_nat": max_abs,
        "fresh_vs_bundle_per_subset_max_abs_nat": per_subset,
        "replay": replay,
        "empty_patch_max_abs": window["empty_patch_max_abs"],
        "factor_reconstruction_max": reconstruction,
        "forwards": forwards,
        "expected_forwards_one_batch": sub474.FORWARDS_PER_BATCH,
        'pred_a_474_code_reproduces_bundle': pred_a,
        'pred_b_replay_empty_exact': pred_b,
        'pred_c_counts_exact': pred_c,
        "raw_tokens_logits_or_hidden_states_included": False,
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, dict)}, indent=1))
    print("per-subset:", {k: f"{v:.3e}" for k, v in per_subset.items()})


if __name__ == "__main__":
    main()
