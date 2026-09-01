"""RUNG 422 (Claude red-team lane) -- CARRIER-RANK SWEEP: HOW DISTRIBUTED IS QK SHARING?

420 falsified my rank-24 common-carrier hypothesis (heldout private
remainder .143 vs the <=.03 collapse bar) and its average-projector
spectrum decays slowly.  This rung quantifies the falsification: sweep
the carrier rank r in {24,40,56,72,88,104} with 420's exact machinery and
seeds (module-global CARRIER_RANK; at r=24 the pipeline is bit-identical
to 420) and measure, per side, heldout carrier-only overlap, private
remainder overlap, Haar-removal control, and permuted-carrier control.

Question scored: what rank would a global carrier need before the
residual pairwise overlap collapses to 420's .03 bar -- a hidden
moderate-rank carrier (cheap collapse) or genuinely distributed sharing
(collapse only near the full 128-dim branch space)?

Frozen predictions
------------------
pred_a (instrument): fold gate <= 1e-10; component orth errors <= 2e-4;
    at r=24 reproduce 420's stored heldout means to <= 1e-3 abs
    (carrier-only q .3164771/k .2803331; remainder q .1428493/k .1430793).
pred_b (distributed decline): remainder mean strictly decreasing in r on
    both sides (slack 1e-4) AND no single step drops more than .06 (no
    hidden-carrier cliff).
pred_c (no cheap collapse): smallest r with remainder <= .03 is > 88 on
    BOTH sides, and at every r the carrier beats Haar removal
    (haar_remainder - remainder >= .01).

Null: some r <= 56 reaches remainder <= .03 (a moderate hidden carrier
exists -- my distributed reading is wrong), or remainder is non-monotone
by > .01, or the carrier fails to beat Haar removal at any r.

Price: identification screen only; no shipped object; no compression or
adoption claim; no 420 bar is altered.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
QK = ROOT / "basis_aligned/qk_mdl"
OUT = BQ / "attention0_qk_carrier_rank_sweep_results.json"
BASE = OPS / "attention0_qk_common_carrier.py"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
RANKS = (24, 40, 56, 72, 88, 104)
HD = 128
VOCAB = 50_257

REF_R24 = {
    "q": {"carrier_only": 0.3164771269592974, "remainder": 0.14284933555043405},
    "k": {"carrier_only": 0.28033309740324813, "remainder": 0.14307926517600814},
}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert RANKS[0] == 24 and all(r < HD for r in RANKS)
        assert all(b > a for a, b in zip(RANKS, RANKS[1:]))
        assert BASE.exists() and ROWS_RECEIPT.exists()
        print("ATTENTION0 QK CARRIER RANK SWEEP | dry run: ranks 24-104, 420-exact machinery")
        return

    started = time.time()
    sys.path.insert(0, str(QK))
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    spec = importlib.util.spec_from_file_location("cc_base", BASE)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    from tier2_model import load_elriggs, reference_forward
    from tier2_folding import branch_factors, scores_from_factors
    import attention0_cross_head_qk_shared_half as parent
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    device = torch.device("cuda")
    model, _config = load_elriggs("bilin18", device=device, dtype=torch.float64)
    gate_factors = {
        branch: branch_factors(model, branch, dtype=torch.float64)
        for branch in (1, 2)}
    factors = {
        branch: branch_factors(model, branch, dtype=torch.float32)
        for branch in (1, 2)}
    receipt = json.loads(ROWS_RECEIPT.read_text())
    rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    fold_errors = parent._fold_gate(
        model, gate_factors, rows, scores_from_factors, reference_forward)
    del gate_factors

    token_ids = torch.arange(VOCAB, device=device)
    fit_mask = token_ids.remainder(5) != 4
    select_mask = ~fit_mask
    full_bases = {side: {} for side in cc.SIDES}
    for entry in cc.ENTRIES:
        for side in cc.SIDES:
            value = parent._factor(factors, entry, side)
            basis, _error = cc._whiten_full(value, fit_mask)
            full_bases[side][entry] = basis

    sweep = {side: [] for side in cc.SIDES}
    max_orth = 0.0
    for side_index, side in enumerate(cc.SIDES):
        fit_bases = {
            entry: full_bases[side][entry][fit_mask] for entry in cc.ENTRIES}
        for rank in RANKS:
            cc.CARRIER_RANK = rank
            carrier, spectrum, _resid = cc._top_average_projector(fit_bases, 420)
            permuted, _pspec, _presid = cc._permuted_carrier(
                fit_bases, 420_000, side_index)
            components = cc._component_bases(
                full_bases[side], fit_mask, select_mask, carrier,
                420_500, side_index)
            permuted_components = cc._component_bases(
                full_bases[side], fit_mask, select_mask, permuted,
                420_700, side_index)
            max_orth = max(max_orth, components["orth_error"],
                           permuted_components["orth_error"])
            row = {
                "rank": rank,
                "carrier_only": cc._stats(
                    cc._pair_values(components["carrier"]))["mean"],
                "remainder": cc._stats(
                    cc._pair_values(components["remainder"]))["mean"],
                "haar_remainder": cc._stats(
                    cc._pair_values(components["haar_remainder"]))["mean"],
                "permuted_carrier_only": cc._stats(
                    cc._pair_values(permuted_components["carrier"]))["mean"],
                "capture_mean": sum(components["captures"].values())
                    / len(components["captures"]),
                "spectrum_tail": float(spectrum[-1]),
            }
            sweep[side].append(row)
    cc.CARRIER_RANK = 24

    def collapse_rank(rows_):
        for row in rows_:
            if row["remainder"] <= .03:
                return row["rank"]
        return None

    collapse = {side: collapse_rank(sweep[side]) for side in cc.SIDES}
    repro = {
        side: {
            "carrier_only_abs_err": abs(
                sweep[side][0]["carrier_only"] - REF_R24[side]["carrier_only"]),
            "remainder_abs_err": abs(
                sweep[side][0]["remainder"] - REF_R24[side]["remainder"]),
        } for side in cc.SIDES}
    monotone = {}
    max_step = {}
    haar_margin_min = {}
    for side in cc.SIDES:
        rem = [row["remainder"] for row in sweep[side]]
        steps = [a - b for a, b in zip(rem, rem[1:])]
        monotone[side] = all(step >= -1e-4 for step in steps)
        max_step[side] = max(steps) if steps else 0.0
        haar_margin_min[side] = min(
            row["haar_remainder"] - row["remainder"] for row in sweep[side])

    pred_a = (
        max(fold_errors.values()) <= 1e-10
        and max_orth <= 2e-4
        and all(repro[side]["carrier_only_abs_err"] <= 1e-3
                and repro[side]["remainder_abs_err"] <= 1e-3
                for side in cc.SIDES))
    pred_b = all(
        monotone[side] and max_step[side] <= .06 for side in cc.SIDES)
    pred_c = all(
        (collapse[side] is None or collapse[side] > 88)
        and haar_margin_min[side] >= .01
        for side in cc.SIDES)
    null = (
        any(collapse[side] is not None and collapse[side] <= 56
            for side in cc.SIDES)
        or any(not monotone[side] and max_step[side] < -.01 for side in cc.SIDES)
        or any(haar_margin_min[side] < 0 for side in cc.SIDES))

    result = {
        "status": "attention0_qk_carrier_rank_sweep_complete",
        "rung": 422,
        "claim_level": "carrier_rank_identification_screen_not_compression",
        "ranks": list(RANKS),
        "sweep": sweep,
        "collapse_rank_at_03": collapse,
        "r24_reproduction": repro,
        "monotone": monotone,
        "max_step": max_step,
        "haar_margin_min": haar_margin_min,
        "fold_gate_max": max(fold_errors.values()),
        "component_orth_max": max_orth,
        'pred_a_instrument_reproduces_420_at_r24': bool(pred_a),
        'pred_b_distributed_monotone_decline_no_cliff': bool(pred_b),
        'pred_c_no_cheap_collapse_above_rank88_and_beats_haar': bool(pred_c),
        'null_moderate_hidden_carrier_or_incoherent_curve': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": "report_effective_sharing_dimensionality_only",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
