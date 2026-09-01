"""RUNG 360 -- PHYSICAL FP16-STORED CONTEXT-QK56 FACTORS.

Keep the exact gated split-B context-QK56 program, store all factor tensors in
IEEE fp16, and explicitly dequantize to fp32 for contraction.  There are no
learned scales or extra tensors.  This tests the precision axis suggested by
the MDL schedule rather than assuming quantization is harmless.

Frozen predictions
------------------
pred_a_fp16_storage_preserves_census_and_certificates:
    Total census <=.014, increment over fp32-QK56 <=.0015, >=42 certificates.
pred_b_fp16_storage_transfers_and_matches_fp32_cev:
    New WT103 mean/p95/max <=.025/.060/.120 and census CEV mean absolute
    difference from fp32-QK56 <=.0015.
pred_c_fp16_identity_and_literal_byte_bill_hold:
    Exactly 440 rank56 factor pairs, every stored tensor fp16, split-B fit,
    active set, new corpus, saved CEV, 512,561,462 scalars / 1,871,225,452
    bytes, and fresh max <=.025.

Null: quantization census increment >=.005 or <=38 certificates.  A full pass
advances the same tightened signed-a16 gate; no rank/precision tuning follows.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed56_context_qk_fp16_storage_ood_results.json"
CEV = ROOT / "cev_mixed56_context_qk_fp16_storage.pt"
REFERENCE_CEV = ROOT / "cev_mixed56_context_metric_qk.pt"
REFERENCE = ROOT / "mixed56_context_metric_qk_newcorpus_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 56
N_ROWS = 120
WIKI_SKIP = 133_640
WIKI_STOP = WIKI_SKIP + N_ROWS * 257
FACTOR_MAPS = 440
FACTOR_SCALARS = FACTOR_MAPS * (128 + 1152) * RANK
SCALARS = 512_561_462
BYTES = 1_871_225_452


def _certificate_count(census_lib, battery, damage):
    valid = 0
    for tag, receipt in battery.items():
        try:
            member = census_lib.leaf(tag)["member"].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(damage[member].abs().mean())
        valid += int(value < .5 * receipt["mean_ablation"]["top"][0]["abs_dce_members"])
    return valid


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert REFERENCE.exists() and REFERENCE_CEV.exists()
        assert (ROOT / f".rowcache/{FIT_CACHE}").exists()
        reference = json.loads(REFERENCE.read_text())
        assert reference["qk_rank"] == RANK and reference["qk_factorized_maps"] == FACTOR_MAPS
        assert FACTOR_SCALARS == 31_539_200
        assert reference["literal_raw_tensor_bytes"] - 2 * FACTOR_SCALARS == BYTES
        assert reference["literal_standalone_scalars"] == SCALARS
        assert WIKI_SKIP == 133_640 and WIKI_STOP == 164_480
        print("FP16-STORED CONTEXT-QK56 | dry run: reference, dtype, bytes, population, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mixed96_context_metric_qk import _attention_input_covariances
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    rows_ood, fingerprint, token_count = _wikitext103_train_rows(
        n=N_ROWS, width=257, skip=WIKI_SKIP)
    cached = torch.load(ROOT / f".rowcache/{FIT_CACHE}", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    covariances = _attention_input_covariances(C.m, fit_rows, _manual_logits)

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": RANK, "qk_rmap": {},
        "qk_extra_tail": 0, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "qk_context_covariances": covariances,
        "qk_factor_storage_dtype": "float16",
        "extra_eval_rows": rows_ood,
        "extra_eval_name": f"wikitext-103-raw-v1-train-skip{WIKI_SKIP}",
    })
    print("ARM: split-B context-QK56 with fp16-stored/fp32-compute factors", flush=True)
    run = C.main()

    wanted_indices = tuple(range(RANK))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    factor_pairs = [factor for heads in qk.values() for factors in heads.values()
                    for factor in factors]
    widths = {int(factor[0].shape[1]) for factor in factor_pairs}
    dtypes = {str(tensor.dtype) for factor in factor_pairs for tensor in factor}
    stored_factor_scalars = sum(tensor.numel() for factor in factor_pairs for tensor in factor)
    factor_maps = len(factor_pairs)
    active = tuple(C.SEL.get("_ORDER2", ()))
    metric = C.SEL.get("_QK_METRIC")
    storage_dtype = C.SEL.get("_QK_STORAGE_DTYPE")
    context_layers = tuple(C.SEL.get("_QK_CONTEXT_LAYERS", ()))
    if (metric != "context_rrr" or storage_dtype != "float16"
            or context_layers != LAYERS or set(index_sets) != set(LAYERS)
            or any(value != wanted_indices for value in index_sets.values())
            or widths != {RANK} or dtypes != {"torch.float16"}
            or factor_maps != FACTOR_MAPS or stored_factor_scalars != FACTOR_SCALARS
            or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: fp16 context-QK56 identity changed")

    cev = C.SEL["cev"].float().reshape(-1).cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    reference_cev = torch.load(REFERENCE_CEV, map_location="cpu").float().reshape(-1)
    assert reference_cev.shape == cev.shape
    cev_mean_abs_difference = float((cev - reference_cev).abs().mean())
    damage_vector = cev - base_ce
    census = float(damage_vector.mean())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    certificates = _certificate_count(CN, battery, damage_vector)
    reference = json.loads(REFERENCE.read_text())
    quantization_increment = census - float(reference["census_damage"])
    extra = C.SEL["extra_eval"]
    by_row = torch.tensor(extra["damage_by_row"], dtype=torch.float64)
    fresh = [float(value) for value in run["fresh8"]]
    p95 = float(torch.quantile(by_row, .95))
    maximum = float(by_row.max())

    pred_a = census <= .014 and quantization_increment <= .0015 and certificates >= 42
    pred_b = (extra["damage_mean"] <= .025 and p95 <= .060 and maximum <= .120
              and cev_mean_abs_difference <= .0015)
    pred_c = (fingerprint == "7dabb830ac9ebb0d" and token_count == 675_457
              and metric == "context_rrr" and storage_dtype == "float16"
              and context_layers == LAYERS and widths == {RANK}
              and dtypes == {"torch.float16"} and factor_maps == FACTOR_MAPS
              and stored_factor_scalars == FACTOR_SCALARS
              and all(value == wanted_indices for value in index_sets.values())
              and max(fresh) <= .025 and SCALARS == 512_561_462
              and BYTES == 1_871_225_452 and CEV.exists())
    null = quantization_increment >= .005 or certificates <= 38
    result = {
        "status": "mixed56_context_qk_fp16_storage_ood_complete",
        "rung": 360,
        "claim_level": "physical_fp16_storage_fp32_compute_census_certificate_ood_screen",
        "convention": "compiled CE minus native CE on identical positions",
        "fit_cache": FIT_CACHE,
        "fit_rows_half_open": list(FIT_SLICE),
        "dataset_fingerprint": fingerprint,
        "row_construction": {"skip_tokens": WIKI_SKIP, "n_rows": N_ROWS,
                             "tokens_per_row": 257},
        "census_damage": census,
        "fp32_reference_census_damage": float(reference["census_damage"]),
        "quantization_census_increment": quantization_increment,
        "census_cev_mean_abs_difference_from_fp32": cev_mean_abs_difference,
        "certificates_valid": certificates,
        "shifted_damage_mean": float(extra["damage_mean"]),
        "shifted_damage_row_p50": float(torch.quantile(by_row, .50)),
        "shifted_damage_row_p95": p95,
        "shifted_damage_row_max": maximum,
        "shifted_damage_by_row": [float(value) for value in by_row],
        "fresh8": fresh,
        "max_fresh_damage": max(fresh),
        "qk_metric": metric,
        "qk_storage_dtype": storage_dtype,
        "qk_compute_dtype": "float32_explicit_dequantization",
        "qk_context_layers": list(context_layers),
        "qk_rank": RANK,
        "qk_factorized_maps": factor_maps,
        "qk_factor_scalars": stored_factor_scalars,
        "qk_factor_tensor_dtypes": sorted(dtypes),
        "active_replacements": list(active),
        "literal_standalone_scalars": SCALARS,
        "literal_raw_tensor_bytes": BYTES,
        "byte_saving_vs_native": 2_067_669_612 - BYTES,
        "byte_saving_fraction_vs_native": (2_067_669_612 - BYTES) / 2_067_669_612,
        "saved_census_cev_file": CEV.name,
        'pred_a_fp16_storage_preserves_census_and_certificates': bool(pred_a),
        'pred_b_fp16_storage_transfers_and_matches_fp32_cev': bool(pred_b),
        'pred_c_fp16_identity_and_literal_byte_bill_hold': bool(pred_c),
        "null_fp16_storage_breaks_qk56": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print("FP16-STORED CONTEXT-QK56 DONE", flush=True)


if __name__ == "__main__":
    main()
