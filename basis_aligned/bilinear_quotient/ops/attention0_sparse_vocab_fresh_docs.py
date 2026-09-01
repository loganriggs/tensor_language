"""RUNG 427 (Claude red-team lane) -- FRESH-DOCUMENT REPLICATION OF 426's VOCABULARY ORDERING.

426 delivered the arc's first surviving cross-head discrete vocabulary:
the global sparse dictionary beats 18 per-head dictionaries at equal or
18.84% lower literal byte price, with derangement catastrophic
downstream.  Its document-level metrics were measured on the arc's
standard SELECT documents.  This rung retrains the identical
deterministic pipeline (426's exact seeds) and evaluates the four
document arms on 96 documents never used anywhere in the program:
mlp2_rank512_refit_v1 TRAIN[0:96] (sha-pinned; in-run zero row-hash
overlap with FIT+SELECT+FINAL; disjoint from my 421/425 EVALUATION
guards).

Arms (426's): I72 (18 independent per-entry dictionaries, k4 each),
G54 (global 512-atom, k54), G72 (same global, k72 = equal code count),
D54 (G54 with decoder atom identities deranged across entry slices).

Frozen predictions (CE damage = CE ADDED ABOVE NATIVE -- LOWER IS BETTER)
------------------
pred_a (instrument + bridge): float64 fold gate <= 1e-10; fresh file sha
    matches; row overlap 0; SELECT bridge reproduces 426's stored values
    -- each arm's SELECT CE damage within 2e-3 abs of
    (I72 .0207265, G54 .0214862, G72 .0175580, D54 .1243076) and G54
    balanced FVU within 5e-3 of .4615068.
pred_b (ordering replicates): fresh G72 CE <= fresh I72 CE + .002 AND
    fresh G72 write rel-sq <= fresh I72 write rel-sq AND fresh G54 CE
    <= fresh I72 CE + .005.
pred_c (controls + stability): fresh D54 CE >= fresh G54 CE + .05 AND
    fresh D54 write >= 1.25x fresh G54 write AND every arm's fresh CE
    damage within +-.01 of its SELECT value.

Null: fresh I72 CE < fresh G72 CE - .002 (ordering reversed off-docs),
or fresh D54 CE < fresh G54 CE + .02 (derangement not costly off-docs),
or any arm's fresh CE deviates > .02 from SELECT => 426's document
conclusions are document-specific.

Price: screen only; 426's literal bills restated (G54 15,583,320 B vs
I72/G72 19,201,824 B); no compression or adoption claim; no 426 bar is
altered by any outcome here.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_sparse_vocab_fresh_docs_results.json"
SV = OPS / "attention0_cross_head_sparse_qk_vocabulary.py"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
FRESH_RECEIPT = BQ / "mlp2_rank512_refit_v1_rows_receipt.json"
FRESH_ROLE = "TRAIN"
FRESH_COUNT = 96

REF_CE = {"I72": 0.020726530502239715, "G54": 0.02148618921637535,
          "G72": 0.017558004707098007, "D54": 0.12430756539106369}
REF_G54_FVU = 0.4615067814423032


def _row_hashes(rows: torch.Tensor) -> set[str]:
    return {hashlib.sha256(row.contiguous().numpy().tobytes()).hexdigest()
            for row in rows.cpu()}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert FRESH_COUNT == 96 and FRESH_ROLE == "TRAIN"
        assert SV.exists() and ROWS_RECEIPT.exists() and FRESH_RECEIPT.exists()
        entries = json.loads(FRESH_RECEIPT.read_text())["entries"]
        assert FRESH_ROLE in entries and Path(entries[FRESH_ROLE]["path"]).exists()
        print("ATTENTION0 SPARSE VOCAB FRESH DOCS | dry run: 426 arms on never-used documents")
        return

    started = time.time()
    sys.path.insert(0, str(QK))
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    spec = importlib.util.spec_from_file_location("sv_mod", SV)
    sv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sv)
    from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
    from tier2_folding import branch_factors, scores_from_factors
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring

    edge_spec = importlib.util.spec_from_file_location(
        "edge_mod", OPS / "attention0_realized_edge_block_term.py")
    edge_mod = importlib.util.module_from_spec(edge_spec)
    edge_spec.loader.exec_module(edge_mod)
    base_spec = importlib.util.spec_from_file_location("ov_base", sv.OV_BASE)
    base = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(base)

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    final_rows = rows_parent.load_role(receipt["entries"]["FINAL"])

    fresh_entry = json.loads(FRESH_RECEIPT.read_text())["entries"][FRESH_ROLE]
    fresh_path = Path(fresh_entry["path"])
    file_sha = hashlib.sha256(fresh_path.read_bytes()).hexdigest()
    fresh_rows = torch.load(fresh_path, weights_only=True)[
        :FRESH_COUNT].contiguous()
    used = (_row_hashes(fit_rows) | _row_hashes(select_rows)
            | _row_hashes(final_rows))
    overlap = sum(1 for h in _row_hashes(fresh_rows) if h in used)

    # Exact float64 fold gate, exactly as 426.
    exact_model, _cfg = load_elriggs("bilin18", device=device, dtype=torch.float64)
    exact_factors = {branch: branch_factors(exact_model, branch, dtype=torch.float64)
                     for branch in (1, 2)}
    captured = {}

    def capture(layer, score1, score2):
        if layer == 0:
            captured[1] = score1.detach()
            captured[2] = score2.detach()
        return score1, score2

    gate_tokens = select_rows[:1, :-1].to(device)
    reference_forward(exact_model, gate_tokens, "bf16", capture)
    fold_errors = {}
    for branch in (1, 2):
        folded = scores_from_factors(
            *exact_factors[branch], gate_tokens, sv.HD, table_dtype="bf16")
        fold_errors[str(branch)] = float((folded - captured[branch]).abs().max())
    del exact_model, exact_factors, captured, folded
    torch.cuda.empty_cache()

    model, _cfg2 = load_elriggs("bilin18", device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    factors = {branch: branch_factors(model, branch, dtype=torch.float32)
               for branch in (1, 2)}
    target_entries = sv._entries_from_factors(factors)
    x_global = target_entries.reshape(sv.VOCAB, sv.GLOBAL_DIM)
    token_ids = torch.arange(sv.VOCAB, device=device)
    fit_ids = token_ids[token_ids.remainder(5) != 4]
    select_ids = token_ids[token_ids.remainder(5) == 4]

    independent = sv._train_independent(target_entries, fit_ids, seed=426)
    global_model = sv._train_global(x_global, fit_ids, sv.K_GLOBAL, seed=427)

    rec_i72, _ii, _ci = sv._encode_independent(target_entries, independent)
    rec_g54, _ig, _cg = sv._encode_global(x_global, global_model, sv.K_GLOBAL)
    rec_g72, _ig2, _cg2 = sv._encode_global(x_global, global_model, sv.K_EQUAL)
    deranged_decoder, _dh = sv._deranged_decoder(global_model)
    rec_d54, _id, _cd = sv._encode_global(
        x_global, global_model, sv.K_GLOBAL, decoder_override=deranged_decoder)

    g54_fvu, _entries_fvu = sv._balanced_fvu(target_entries, rec_g54, select_ids)

    candidate_tables = {
        "I72": sv._tables(rec_i72), "G54": sv._tables(rec_g54),
        "G72": sv._tables(rec_g72), "D54": sv._tables(rec_d54),
    }
    with torch.no_grad():
        bridge_metrics, _bridge_replay = sv._document_metrics(
            model, select_rows, candidate_tables, base, edge_mod,
            scoring, scores_from_factors)
        fresh_metrics, fresh_replay = sv._document_metrics(
            model, fresh_rows, candidate_tables, base, edge_mod,
            scoring, scores_from_factors)

    bridge_ce = {arm: bridge_metrics["ce"][arm]["damage"] for arm in sv.ARMS}
    fresh_ce = {arm: fresh_metrics["ce"][arm]["damage"] for arm in sv.ARMS}
    bridge_ok = all(abs(bridge_ce[arm] - REF_CE[arm]) <= 2e-3 for arm in sv.ARMS)
    fresh_write = fresh_metrics["full_attention0_write_relative_squared_error"]
    stability = {arm: abs(fresh_ce[arm] - bridge_ce[arm]) for arm in sv.ARMS}

    pred_a = (
        max(fold_errors.values()) <= 1e-10
        and file_sha == fresh_entry["file_sha256"]
        and overlap == 0
        and bridge_ok
        and abs(g54_fvu - REF_G54_FVU) <= 5e-3)
    pred_b = (
        fresh_ce["G72"] <= fresh_ce["I72"] + .002
        and fresh_write["G72"] <= fresh_write["I72"]
        and fresh_ce["G54"] <= fresh_ce["I72"] + .005)
    pred_c = (
        fresh_ce["D54"] >= fresh_ce["G54"] + .05
        and fresh_write["D54"] >= 1.25 * fresh_write["G54"]
        and all(value <= .01 for value in stability.values()))
    null = (
        fresh_ce["I72"] < fresh_ce["G72"] - .002
        or fresh_ce["D54"] < fresh_ce["G54"] + .02
        or any(value > .02 for value in stability.values()))

    result = {
        "status": "attention0_sparse_vocab_fresh_docs_complete",
        "rung": 427,
        "claim_level": "fresh_document_replication_of_426_screen_not_compression",
        "fresh_source": {"receipt": FRESH_RECEIPT.name, "role": FRESH_ROLE,
                         "count": FRESH_COUNT, "file_sha256": file_sha,
                         "row_overlap_with_426_roles": overlap},
        "instrument": {"fold_max_abs_by_branch": fold_errors,
                       "g54_balanced_fvu": g54_fvu,
                       "bridge_ce_damage": bridge_ce,
                       "fresh_no_native_qk_replay_rel_sq": fresh_replay},
        "reference_426": {"select_ce_damage": REF_CE,
                          "g54_balanced_fvu": REF_G54_FVU},
        "fresh_document_metrics": fresh_metrics,
        "fresh_ce_damage": fresh_ce,
        "fresh_write_relative_squared_error": fresh_write,
        "ce_stability_abs_shift": stability,
        'pred_a_instrument_and_select_bridge': bool(pred_a),
        'pred_b_global_beats_independent_off_docs': bool(pred_b),
        'pred_c_derangement_costly_and_damage_stable': bool(pred_c),
        'null_426_ordering_is_document_specific': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": ("426_ordering_document_stable" if pred_a and pred_b
                      and pred_c and not null
                      else "426_document_claims_need_diagnosis"),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
