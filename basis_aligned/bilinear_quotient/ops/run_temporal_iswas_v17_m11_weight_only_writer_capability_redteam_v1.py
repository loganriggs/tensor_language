#!/usr/bin/env python3
"""Red-team whether exact M11 weight capability enriches known causal writers."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_basis_formula_finiteness_and_exact_price pred_b_both_causal_writers_are_enriched_by_weight_only_capability pred_c_causal_pair_has_a_weight_only_capability_margin pred_d_scope_remains_diagnostic_and_opens_no_new_causal_outcome
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import fastload
from circuit_fast_screen_managed_runner import atomic_create_json
import subspace_weight_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v17_m11_weight_only_writer_capability_redteam_v1.json"
GREEDY = ROOT / "circuits/followups/temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1_result.json"
U8_RESULT = ROOT / "circuits/followups/temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1_result.json"
ATLAS = ROOT / "ops/subspace_weight_atlas.py"
FASTLOAD = ROOT / "ops/fastload.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v17_m11_weight_only_writer_capability_redteam_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v17_m11_weight_only_writer_capability_redteam_v1"
EXPECTED = {
    "prior": "f06241a6b24e976580f124ee1494198d10c7a88bbf67df689aafb291ca2d5235",
    "greedy": "6a2ae6598ae95269f7d9e24e13196155342f690d331c78efed5115a8cf84bda2",
    "u8_result": "f267b3ebbe151077f0aa44939e6f01e78fc8ae30e85d9b52e0858383ef893972",
    "atlas": "076c63208ed10658702e1779ce336a3d85ba89191542408f45eb3bc9e0c6cf68",
    "fastload": "5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90",
    "checkpoint": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    "basis": "e14ff7b7bdae17c0339744b2c37a5bb77b80e973e9d70b78f742e67aa1e4835f",
}
HEADS = ((7, 7), (9, 4), (6, 7), (9, 1), (3, 4), (5, 1))
POSITIVES = ("L07H07", "L09H04")
PRICE = {"checkpoint_loads": 1, "model_forwards": 0, "sequence_evaluations": 0,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0,
         "head_weight_tensors": 6}
BARS = {"basis_orthonormal": 1e-5, "formula_relative_error": 1e-5,
        "enriched_rank_max": 4, "positive_mean_ratio": 1.10}
PREDICTION_KEYS = (
    "pred_a_authority_basis_formula_finiteness_and_exact_price",
    "pred_b_both_causal_writers_are_enriched_by_weight_only_capability",
    "pred_c_causal_pair_has_a_weight_only_capability_margin",
    "pred_d_scope_remains_diagnostic_and_opens_no_new_causal_outcome",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def spectrum_record(reports) -> list[dict]:
    return [{"mode": report["mode"], "shape": list(report["shape"]),
             "stable_rank": report["stable_rank"],
             "singular_values": report["singular_values"].tolist()}
            for report in reports]


def main():
    config, checkpoint, _module = fastload._paths()
    paths = {"prior": PRIOR, "greedy": GREEDY, "u8_result": U8_RESULT,
             "atlas": ATLAS, "fastload": FASTLOAD}
    observed = {name: sha(path) for name, path in paths.items()}
    greedy, u8_result = json.loads(GREEDY.read_text()), json.loads(U8_RESULT.read_text())
    basis_record = u8_result.get("basis", {})
    authority_ok = bool(observed == {name: EXPECTED[name] for name in paths}
        and greedy.get("terminal") == "compact_source_writer_prefixes"
        and greedy.get("selected_prefixes", {}).get("iswas", {}).get("heads")
            == ["L07H07", "L09H04"]
        and u8_result.get("terminal") == "compact_shared_weight_tensor"
        and basis_record.get("sha256") == EXPECTED["basis"]
        and basis_record.get("shape") == [1152, 8]
        and len(basis_record.get("values", [])) == 1152 * 8)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "candidate_heads": [f"L{layer:02d}H{head:02d}" for layer, head in HEADS],
           "causal_positive_set": list(POSITIVES), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("weight-only writer red-team authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if sha(checkpoint) != EXPECTED["checkpoint"]:
        raise RuntimeError("checkpoint weights changed")
    if config != {"vocab_size": 50304, "n_layer": 18, "n_head": 9, "n_embd": 1152,
                  "squared_mlp": False, "bilinear": True, "expansion_factor": 4,
                  "gated": False, "squared_attn": True, "bilinear_attn": True}:
        raise RuntimeError("model configuration changed")

    started_utc, started = utc_now(), time.perf_counter()
    model = fastload.load_model_fast().eval()
    import torch
    torch.set_num_threads(4)
    basis = torch.tensor(basis_record["values"], dtype=torch.float32).reshape(1152, 8)
    basis_error = float((basis.T @ basis - torch.eye(8)).abs().max())
    m11 = model.transformer.h[11].mlp
    x = torch.linspace(-1.0, 1.0, 1152, dtype=torch.float32)
    x = x / x.norm()
    z = torch.linspace(-0.5, 0.5, 128, dtype=torch.float32)
    z = z / z.norm()
    records = {}
    max_formula_error = 0.0
    with torch.no_grad():
        for layer, head in HEADS:
            label = f"L{layer:02d}H{head:02d}"
            attention = model.transformer.h[layer].attn
            start = head * int(attention.head_dim)
            writer = attention.c_proj.weight.detach().float()[:, start:start + int(attention.head_dim)]
            capability = atlas.bilinear_mlp_writer_capability(m11, writer, basis.T)
            direct_base = basis.T @ m11.Down.weight.detach().float() @ (
                (m11.Left.weight.detach().float() @ x)
                * (m11.Right.weight.detach().float() @ x))
            changed_x = x + writer @ z
            direct_changed = basis.T @ m11.Down.weight.detach().float() @ (
                (m11.Left.weight.detach().float() @ changed_x)
                * (m11.Right.weight.detach().float() @ changed_x))
            replay = (torch.einsum("aik,i,k->a", capability["cross"], x, z)
                      + torch.einsum("akl,k,l->a", capability["self"], z, z))
            reference = direct_changed - direct_base
            formula_error = float(torch.linalg.vector_norm(replay - reference)
                                  / torch.linalg.vector_norm(reference).clamp_min(1e-30))
            max_formula_error = max(max_formula_error, formula_error)
            record = {"scores": capability["scores"], "formula_relative_error": formula_error}
            if label in POSITIVES:
                record["cross_unfoldings"] = spectrum_record(
                    atlas.tensor_unfolding_spectra(capability["cross"]))
                record["self_unfoldings"] = spectrum_record(
                    atlas.tensor_unfolding_spectra(capability["self"]))
            records[label] = record
            del capability
    order = sorted(records, key=lambda label: (-records[label]["scores"]["cross_normalized"], label))
    ranks = {label: order.index(label) + 1 for label in order}
    positive_mean = sum(records[label]["scores"]["cross_normalized"] for label in POSITIVES) / 2
    remainder = [label for label in order if label not in POSITIVES]
    remainder_mean = sum(records[label]["scores"]["cross_normalized"] for label in remainder) / len(remainder)
    mean_ratio = positive_mean / max(remainder_mean, 1e-30)
    A = bool(authority_ok and basis_error <= BARS["basis_orthonormal"]
             and max_formula_error <= BARS["formula_relative_error"]
             and finite(records) and PRICE["head_weight_tensors"] == len(HEADS))
    B = all(ranks[label] <= BARS["enriched_rank_max"] for label in POSITIVES)
    C = mean_ratio >= BARS["positive_mean_ratio"]
    D = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else
                "weight_capability_enriches_causal_writers" if B and C else
                "weight_capability_requires_occupancy")
    result = {"schema": "temporal_iswas_v17_m11_weight_only_writer_capability_redteam_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_cpu_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "instrument": {"basis_orthonormal_max_abs_error": basis_error,
                       "max_formula_relative_error": max_formula_error},
        "records": records, "normalized_cross_order": order, "ranks": ranks,
        "causal_positive_mean_to_remainder_ratio": mean_ratio,
        "causal_selector_or_new_edge_identified": None,
        "v20_hr_outcome_opened": False, "v21_causal_outcomes_opened": False,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("instrument", "normalized_cross_order",
        "ranks", "causal_positive_mean_to_remainder_ratio", "predictions", "terminal",
        "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
