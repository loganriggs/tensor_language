#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_fresh_response_replay pred_c_fresh_bidirectional_causality pred_d_fresh_preservation pred_e_rank8_compression pred_f_beats_response_basis pred_g_random_specificity
"""Frozen second-panel test of the exported rank-eight source-PCA port."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import run_setting2_regional_attention17h2_response_basis_fresh_v1 as base


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_SOURCE_PCA_FRESH_V1_PREREGISTRATION.md"
ROWS = P / "REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_SOURCE_PCA_FRESH_V1_BINDING.json"
PRIOR = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_response_basis_fresh_v1_result.json"
EXPORT = base.PARENT
ARTIFACT = base.ARTIFACT
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_source_pca_fresh_v1_result.json"
PRICE = {"physical_model_executions": 20, "full_model_sequences": 96, "basis_candidates": 7, "candidate_suffix_sequences": 672, "complete_corner_suffix_sequences": 96, "backwards": 0, "fits": 0, "parameter_updates": 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authority():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "rows": ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "prior_fresh_result": PRIOR,
        "export_result": EXPORT,
        "basis_artifact": ARTIFACT,
        "base_runner": Path(base.__file__).resolve(),
        "export_runner": Path(base.exporter.__file__).resolve(),
        "head_runner": Path(base.head.__file__).resolve(),
        "search_runner": Path(base.search.__file__).resolve(),
        "port_runner": Path(base.port.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    prior = json.loads(PRIOR.read_text())
    if prior["terminal"] != "valid_fresh_response_basis_null" or not prior["same_rank_source_pca"]["passes_response"] or not prior["same_rank_source_pca"]["passes_causality"] or not prior["same_rank_source_pca"]["passes_controls"]:
        raise ValueError("source-PCA discovery authority changed")
    export = json.loads(EXPORT.read_text())
    if export["terminal"] != "consumer_response_basis_rank8_exported" or export["artifact_sha256"] != sha(ARTIFACT):
        raise ValueError("export authority changed")
    return binding, prior, export


def load_bound_second():
    _, _, export = authority()
    artifact = base.torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    rows = json.loads(ROWS.read_text())["rows"]
    checks = base.validate(rows)
    discovery = json.loads((P / "ODD_FRAMING_FRESH_V1_ROWS.json").read_text())["rows"]
    overlap = len({row["text"] for row in rows} & {row["text"] for row in discovery})
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or overlap != 0 or executions != PRICE["physical_model_executions"]:
        raise ValueError("row overlap or execution price changed")
    return json.loads(BINDING.read_text()), export, artifact, rows, checks, buckets, overlap


original_atomic_create_json = base.managed.atomic_create_json


def source_pca_atomic_create_json(path, result):
    authority()
    response = result["selected"]
    pca = result["same_rank_source_pca"]
    random_median = result["same_rank_random_median_minimax_score"]
    instrument = result["predictions"]["pred_a_exact_instrument"]
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_fresh_response_replay": bool(instrument and pca["passes_response"]),
        "pred_c_fresh_bidirectional_causality": bool(instrument and pca["passes_causality"]),
        "pred_d_fresh_preservation": bool(instrument and pca["passes_controls"]),
        "pred_e_rank8_compression": bool(instrument and pca["source_norm_retained_fraction"] <= .65),
        "pred_f_beats_response_basis": bool(instrument and pca["minimax_normalized_gate_score"] <= .90 * response["minimax_normalized_gate_score"]),
        "pred_g_random_specificity": bool(instrument and pca["minimax_normalized_gate_score"] + .10 <= random_median),
    }
    result["schema"] = "setting2_regional_attention17h2_source_pca_fresh_v1_result"
    result["terminal"] = "fresh_head17_2_rank8_source_pca_port" if all(predictions.values()) else "valid_fresh_source_pca_null" if instrument else "invalid"
    result["predictions"] = predictions
    result["selected"] = pca
    result["same_rank_response_basis"] = response
    result.pop("same_rank_source_pca", None)
    result["selection_rule"] = "frozen exported source-PCA rank eight; no second-panel selection"
    result["price"] = PRICE
    result["scope"] = "Second-panel frozen rank-eight source-PCA projector on short-context unseen-endpoint regional rows; eight native source-edge deltas remain live ports."
    result["runner_sha256"] = sha(RUNNER)
    result["binding_sha256"] = sha(BINDING)
    original_atomic_create_json(path, result)


def configure():
    authority()
    base.RUNNER = RUNNER
    base.PREREG = PREREG
    base.ROWS = ROWS
    base.BINDING = BINDING
    base.PARENT = EXPORT
    base.ARTIFACT = ARTIFACT
    base.OUT = OUT
    base.PRICE = PRICE
    base.load_bound = load_bound_second
    base.managed.atomic_create_json = source_pca_atomic_create_json


def main():
    configure()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        planned = base.plan()
        planned["schema"] = "setting2_regional_attention17h2_source_pca_fresh_v1_plan"
        planned["selected_basis"] = "source_pca"
        print(json.dumps(planned, sort_keys=True))
        return
    base.main()


if __name__ == "__main__":
    main()
