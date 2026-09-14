#!/usr/bin/env python3
"""Reconcile the stale successor-pointer registry with its primary receipts."""
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SEM = BASE / "qk_mdl/algo_tasks/semantics_successor"
OOD = BASE / "bilinear_quotient/circuits/fast_screens/attn8_h3_h7_cached_successor_final_ood_v1_result.json"
OUT = Path(__file__).resolve().parent / "SUCCESSOR_POINTER_AUTHORITY_RECONCILIATION_V1_RESULT.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gate = json.loads((SEM / "s2_gate.json").read_text())
    natural = json.loads((SEM / "s2b_natural_fineweb.json").read_text())
    agreement = json.loads((SEM / "s2c_agreement.json").read_text())
    redteam = json.loads((SEM / "s6_redteam.json").read_text())
    ood = json.loads(OOD.read_text())

    assert gate["baseline"]["success_rate"] == gate["B_code_self"]["success_rate"]
    assert gate["B_zero"]["success_rate"] < 0.05
    assert agreement["B_coded_vs_real_agree"] > 0.90
    assert abs(natural["B_coded_restricted"]["dce"]) < 1e-6
    assert natural["B_coded_all"]["dce"] > 0.40

    evidence = [row for row in ood["evidence"] if row["family_id"] == "list_step_two_conflict"]
    direction = {}
    for name in ("lower_to_higher", "higher_to_lower"):
        rows = [row for row in evidence if row["direction"] == name]
        assert rows
        direction[name] = {
            "rows": len(rows),
            "donorward_fraction": sum(row["donor_margin_effect"] > 0 for row in rows) / len(rows),
            "donor_answer_win_fraction": sum(row["donor_answer_win"] for row in rows) / len(rows),
        }
    assert direction["lower_to_higher"]["donor_answer_win_fraction"] == 1.0
    assert direction["higher_to_lower"]["donor_answer_win_fraction"] == 0.0

    holdout = redteam["holdout_elements"]
    failed_holdout = sum(
        value["holdfit"]["B_follow"] < value["fullfit"]["B_follow"]
        for value in holdout.values()
    )
    out = {
        "experiment": "successor_pointer_authority_reconciliation_v1",
        "analysis_status": "retrospective_primary_receipt_audit",
        "registered_status_supported": "site_live",
        "metrics": {
            "baseline_success": gate["baseline"]["success_rate"],
            "coded_self_success": gate["B_code_self"]["success_rate"],
            "zero_pointer_success": gate["B_zero"]["success_rate"],
            "coded_wrong_pointer_follow": gate["B_code_placebo"]["success_rate"],
            "real_wrong_pointer_follow": gate["B_real_wrongcontent"]["success_rate"],
            "coded_real_prediction_agreement": agreement["B_coded_vs_real_agree"],
            "restricted_fineweb_dce": natural["B_coded_restricted"]["dce"],
            "full_vocabulary_fineweb_dce": natural["B_coded_all"]["dce"],
            "heldout_element_failures_vs_full_fit": failed_holdout,
            "heldout_elements": len(holdout),
            "recent_direction_screen": direction,
        },
        "supported_boundary": (
            "Legacy causal receipts support a token-identity pointer in the layer-0 "
            "value cache with distributed attention readers and identity-keyed successor "
            "tables. The code is lossless on its calibrated element domain and causally "
            "moves successor predictions."
        ),
        "unresolved_boundary": (
            "The code does not generate unseen element identities, is not a full-vocabulary "
            "law, and the recent H3+H7 cached-value transfer is directionally gated by "
            "prefix context. A prospective pointer-by-context interaction test is required."
        ),
        "claim_boundary": (
            "Registry reconciliation of legacy evidence, not a new prospective model run, "
            "stable identification claim, standalone lookup-table extraction, or compression adoption."
        ),
        "sources": {
            name: digest(SEM / name)
            for name in ("report.md", "s2_gate.json", "s2b_natural_fineweb.json", "s2c_agreement.json", "s6_redteam.json")
        } | {"attn8_h3_h7_cached_successor_final_ood_v1_result.json": digest(OOD)},
        "source_sha256": digest(Path(__file__)),
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
