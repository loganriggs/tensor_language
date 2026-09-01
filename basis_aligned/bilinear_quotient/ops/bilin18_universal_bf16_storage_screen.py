"""RUNG 362 -- UNIVERSAL BF16-STORAGE / FP32-COMPUTE SCREEN.

Use checkpoint metadata to round every source-fp32 tensor once through
bfloat16, then dequantize to fp32 for the unchanged computation.  Tensors
already stored as bfloat16 remain bit-exact.  This asks whether the full model
can be represented at two bytes per semantic scalar before composing with the
structural QK56 program.

Frozen predictions
------------------
pred_a_bf16_storage_preserves_mean_on_both_corpora:
    Mean damage <=.006 on FineWeb and WikiText-103 train.
pred_b_bf16_storage_preserves_tails_and_transfers:
    p95/max <=.015/.035 on both and absolute corpus mean gap <=.004.
pred_c_checkpoint_identity_and_two_byte_bill_hold:
    Exactly 218 tensors, 545,902,902 scalars, native bytes2,067,669,612;
    source dtypes only fp32/bf16, two-byte bill1,091,805,804; names/shapes
    unchanged and source-bf16 tensors bit-exact.

Null: mean >=.020 on either corpus or max >=.080.  This is a screen; a pass
licenses one universal-bf16 + fp16-QK56 physical composition, not tuning.
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
OUT = ROOT / "bilin18_universal_bf16_storage_screen_results.json"
FINEWEB_ROWS = (120, 160)
WIKI_SKIP = 640 * 257
EVAL_ROWS = 40
NATIVE_SCALARS = 545_902_902
NATIVE_BYTES = 2_067_669_612
BF16_BYTES = 2 * NATIVE_SCALARS


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert (ROOT / ".rowcache/fineweb_n192_skip7000.pt").exists()
        assert FINEWEB_ROWS == (120, 160)
        assert WIKI_SKIP == 164_480 and WIKI_SKIP + EVAL_ROWS * 257 == 174_760
        assert BF16_BYTES == 1_091_805_804 and BF16_BYTES < .53 * NATIVE_BYTES
        print("UNIVERSAL BF16 STORAGE | dry run: populations, exact bill, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mixed56_context_metric_qk_newcorpus_ood import _wikitext103_train_rows
    from mlp0_tail_robust_context_metric_screen import _score_rows, _summary
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import REPOS, _fetch, load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == 1152 and cfg["n_layer"] == 18
    checkpoint_path = _fetch(REPOS["bilin18"], "pytorch_model.bin")
    source = torch.load(checkpoint_path, map_location="meta", weights_only=True, mmap=True)
    assert len(source) == 218
    source_scalars = sum(tensor.numel() for tensor in source.values())
    source_bytes = sum(tensor.numel() * tensor.element_size() for tensor in source.values())
    assert source_scalars == NATIVE_SCALARS and source_bytes == NATIVE_BYTES

    cached = torch.load(ROOT / ".rowcache/fineweb_n192_skip7000.pt", map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fineweb = cached[FINEWEB_ROWS[0]:FINEWEB_ROWS[1], :257].long().contiguous()
    wikitext, fingerprint, token_count = _wikitext103_train_rows(
        n=EVAL_ROWS, width=257, skip=WIKI_SKIP)
    native = {
        "fineweb": _score_rows(model, fineweb, _manual_logits),
        "wikitext": _score_rows(model, wikitext, _manual_logits),
    }

    parameters = dict(model.named_parameters())
    assert set(parameters) == set(source)
    shapes_before = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
    dtype_scalars = {"torch.float32": 0, "torch.bfloat16": 0}
    changed_tensors = 0
    changed_scalars = 0
    bf16_source_exact = True
    rounding_mean_abs_numerator = 0.0
    for name, parameter in parameters.items():
        source_dtype = str(source[name].dtype)
        if source_dtype not in dtype_scalars:
            raise RuntimeError(f"unsupported source dtype {source_dtype} for {name}")
        dtype_scalars[source_dtype] += parameter.numel()
        rounded = parameter.data.bfloat16().float()
        difference = (rounded - parameter.data).abs()
        if source_dtype == "torch.bfloat16":
            bf16_source_exact = bf16_source_exact and bool(torch.equal(rounded, parameter.data))
        else:
            changed_tensors += int(bool((difference != 0).any()))
            changed_scalars += int((difference != 0).sum())
            rounding_mean_abs_numerator += float(difference.double().sum().cpu())
            parameter.data.copy_(rounded)
    assert sum(dtype_scalars.values()) == NATIVE_SCALARS
    assert changed_tensors > 0 and changed_scalars > 0 and bf16_source_exact
    assert {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before

    rounded_scores = {
        "fineweb": _score_rows(model, fineweb, _manual_logits),
        "wikitext": _score_rows(model, wikitext, _manual_logits),
    }
    summaries = {}
    for corpus in ("fineweb", "wikitext"):
        summaries[corpus] = _summary(rounded_scores[corpus] - native[corpus])
        print(f"universal_bf16 {corpus}: {summaries[corpus]}", flush=True)

    pred_a = all(summaries[corpus]["mean"] <= .006 for corpus in ("fineweb", "wikitext"))
    pred_b = (all(summaries[corpus]["p95"] <= .015 and summaries[corpus]["max"] <= .035
                  for corpus in ("fineweb", "wikitext"))
              and abs(summaries["fineweb"]["mean"] - summaries["wikitext"]["mean"]) <= .004)
    pred_c = (len(source) == 218 and source_scalars == NATIVE_SCALARS
              and source_bytes == NATIVE_BYTES and set(dtype_scalars) == {"torch.float32", "torch.bfloat16"}
              and all(count > 0 for count in dtype_scalars.values())
              and BF16_BYTES == 1_091_805_804 and bf16_source_exact
              and set(parameters) == set(source)
              and {name: tuple(parameter.shape) for name, parameter in parameters.items()} == shapes_before)
    null = (any(summaries[corpus]["mean"] >= .020 for corpus in ("fineweb", "wikitext"))
            or any(summaries[corpus]["max"] >= .080 for corpus in ("fineweb", "wikitext")))
    result = {
        "status": "bilin18_universal_bf16_storage_screen_complete",
        "rung": 362,
        "claim_level": "two_corpus_global_precision_storage_screen_only",
        "program": {
            "storage": "all semantic tensors bfloat16",
            "compute": "explicitly dequantized fp32 parameters",
            "scales_or_codebooks": 0,
            "semantic_scalars": NATIVE_SCALARS,
            "literal_raw_tensor_bytes": BF16_BYTES,
            "byte_saving_vs_native": NATIVE_BYTES - BF16_BYTES,
            "byte_saving_fraction_vs_native": (NATIVE_BYTES - BF16_BYTES) / NATIVE_BYTES,
        },
        "checkpoint": {
            "entries": len(source),
            "native_scalars": source_scalars,
            "native_raw_tensor_bytes": source_bytes,
            "source_dtype_scalars": dtype_scalars,
            "source_bfloat16_tensors_bit_exact": bf16_source_exact,
            "rounded_fp32_tensors_changed": changed_tensors,
            "rounded_fp32_scalars_changed": changed_scalars,
            "mean_abs_rounding_over_source_fp32_scalars": (
                rounding_mean_abs_numerator / dtype_scalars["torch.float32"]),
            "names_and_shapes_exact": True,
        },
        "evaluation": {
            "fineweb_cache": "fineweb_n192_skip7000.pt",
            "fineweb_rows_half_open": list(FINEWEB_ROWS),
            "wikitext103_train_token_span_half_open": [WIKI_SKIP, WIKI_SKIP + EVAL_ROWS * 257],
            "dataset_fingerprint": fingerprint,
            "source_token_count": token_count,
        },
        "row_damage_summaries": summaries,
        'pred_a_bf16_storage_preserves_mean_on_both_corpora': bool(pred_a),
        'pred_b_bf16_storage_preserves_tails_and_transfers': bool(pred_b),
        'pred_c_checkpoint_identity_and_two_byte_bill_hold': bool(pred_c),
        "null_universal_bf16_storage_breaks_prediction": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "dtype_scalars": dtype_scalars,
                      "predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("BILIN18 UNIVERSAL BF16 STORAGE SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
