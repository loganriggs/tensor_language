#!/usr/bin/env python3
"""Build and causally test the H3/M11 shared-basis restricted weight tensor."""

# BQGATE: EXPERIMENT pred_a_authority_capture_replay_basis_roundtrip_finiteness_and_exact_price pred_b_shared_U8_reaches_both_causal_readers_on_holdout pred_c_shared_U8_is_causally_sufficient_for_H3_A11_and_M11 pred_d_exact_checkpoint_writer_reader_and_M11_tensor_contractions_close pred_e_top32_weight_defined_M11_factors_are_structurally_and_causally_compact
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import module_reader_loss_rescue as intervention
import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as parent
import subspace_weight_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1.json"
MODULE_RESULT = ROOT / "circuits/followups/temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1_result.json"
HEAD_RESULT = ROOT / "circuits/followups/temporal_iswas_v17_a11_head_endpoint_ordered_cumulative_v1_result.json"
MODULE_RUNNER = ROOT / "ops/run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1.py"
HEAD_RUNNER = ROOT / "ops/run_temporal_iswas_v17_a11_head_endpoint_ordered_cumulative_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17.py"
ATLAS = ROOT / "ops/subspace_weight_atlas.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v17_h3_m11_shared_basis_restricted_weight_tensor_v1"
EXPECTED = {
    "prior": "5a94c0542c81a946fe2a16c9179af1724c3aa2c79acc529bf89254e68c106bd0",
    "module_result": "4883354bf39fe6bfe9ffd5c2f97475be749a0aec78d8aa13985739bcf4b93062",
    "head_result": "7fa315b84f71f267c3ed4cea67231cbad9dd0ef2b5ad25a3b18133fc6a6e9c85",
    "module_runner": "a34217de464b988f10be94b60d7526146b9f86c4ad717502e4dfe852e1f2b0d7",
    "head_runner": "0e529b948a065dd5ae5c317e0517303f11e012b1925df73f554e0bca8dfdde0e",
    "builder": "7e59800324e5b6a4a9c5af564cbe8fb5cf642cfd6e702e1e97c32bc0cc54fe9e",
    "atlas": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5",
}
RANK, TOP = 8, 32
PRICE = {"checkpoint_loads": 1, "model_forwards": 9, "sequence_evaluations": 144,
         "scored_token_positions": 288, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "orthonormal": 1e-5, "energy": .80,
        "effect_fraction": .80, "effect_cosine": .95,
        "tensor_residual": .50, "top_effect_fraction": .70,
        "top_effect_cosine": .95, "formula": 1e-5}
PREDICTION_KEYS = (
    "pred_a_authority_capture_replay_basis_roundtrip_finiteness_and_exact_price",
    "pred_b_shared_U8_reaches_both_causal_readers_on_holdout",
    "pred_c_shared_U8_is_causally_sufficient_for_H3_A11_and_M11",
    "pred_d_exact_checkpoint_writer_reader_and_M11_tensor_contractions_close",
    "pred_e_top32_weight_defined_M11_factors_are_structurally_and_causally_compact",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def capture_sites(forward, model):
    sites = {"A11": (model.transformer.h[11].attn,
                     model.transformer.h[11].attn.c_proj),
             "M11": model.transformer.h[11].mlp}
    return intervention.capture_reader_outputs(forward, sites)


def selected_rows(tensor, rows, positions, phase):
    chunks = []
    for row, (record, row_positions) in enumerate(zip(rows, positions)):
        chosen = record["group_number"] < 8 if phase == "FIT" else record["group_number"] >= 8
        if chosen and row_positions:
            chunks.append(tensor[row, list(row_positions)].detach().float().cpu())
    return __import__("torch").cat(chunks, dim=0)


def fit_basis(absent, present, rows, positions, rank=RANK):
    torch = __import__("torch")
    matrix = torch.cat([selected_rows(present[label] - absent[label], rows, positions, "FIT")
                        for label in ("A11", "M11")], dim=0).double()
    _u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    basis = vh[:rank].T.contiguous()
    for column in range(rank):
        pivot = int(torch.argmax(torch.abs(basis[:, column])))
        if float(basis[pivot, column]) < 0:
            basis[:, column] *= -1
    return basis.float(), singular.float()


def project_reader(absent, present, basis):
    basis = basis.to(device=present.device, dtype=torch_float(present))
    delta = (present - absent).float()
    projected = (delta @ basis.float()) @ basis.float().T
    return absent + projected.to(absent)


def torch_float(tensor):
    return tensor.float().dtype


def run_arm(backend, batch, base_heads, donor_heads, source_positions, positions,
            replacements, *, output_replacement=None):
    calls = {label: 0 for label in replacements}
    output_calls, handles = 0, []
    def pre(label):
        def hook(_module, arguments):
            calls[label] += 1
            changed = intervention.replace_positions(arguments[0], replacements[label], positions)
            return (changed,) + tuple(arguments[1:])
        return hook
    targets = {"A11": backend.model.transformer.h[11].attn,
               "M11": backend.model.transformer.h[11].mlp}
    for label in replacements:
        handles.append(targets[label].register_forward_pre_hook(pre(label)))
    if output_replacement is not None:
        def post(_module, _arguments, output):
            nonlocal output_calls
            output_calls += 1
            return intervention.replace_positions(output, output_replacement, positions)
        handles.append(targets["M11"].register_forward_hook(post))
    try:
        output, source_calls = parent.with_source_heads(
            lambda: parent.full_forward(backend, batch), backend.model, batch,
            base_heads, donor_heads, source_positions)
    finally:
        for handle in handles: handle.remove()
    if (calls and set(calls.values()) != {1}) or output_calls != int(output_replacement is not None):
        raise RuntimeError(f"restricted tensor arm coverage changed: {calls}/{output_calls}")
    return output, {"readers": calls, "output": output_calls, "source": source_calls}


def phase_mask(rows, phase):
    return np.asarray([row["group_number"] < 8 if phase == "FIT"
                       else row["group_number"] >= 8 for row in rows])


def response_metrics(absent_margin, present_margin, reference):
    return parent.metrics(present_margin - absent_margin, reference)


def tensor_record(tensor):
    value = tensor.detach().float().cpu().contiguous()
    return {"shape": list(value.shape), "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "values": value.reshape(-1).tolist()}


def main():
    paths = {"prior": PRIOR, "module_result": MODULE_RESULT, "head_result": HEAD_RESULT,
             "module_runner": MODULE_RUNNER, "head_runner": HEAD_RUNNER,
             "builder": BUILDER, "atlas": ATLAS}
    observed = {name: sha(path) for name, path in paths.items()}
    module_result, head_result = json.loads(MODULE_RESULT.read_text()), json.loads(HEAD_RESULT.read_text())
    rows = [row for row in fresh.build_rows() if row["transform_id"] == "A2"]
    authority_ok = bool(observed == EXPECTED and len(rows) == 16
        and module_result.get("predictions", {}).get(
            "pred_c_M11_reader_loss_is_stable_and_complete_M11_output_rescues") is True
        and head_result.get("terminal") == "h3_compact_A11_endpoint")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": len(rows), "basis_rank": RANK, "top_factors": TOP, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("restricted tensor authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    source_positions = parent.postcue_rows(batch, donor_batch)
    calls = {}
    with torch.no_grad():
        native_bundle, base_heads, calls["native_source"] = parent.capture_source_heads(
            lambda: capture_sites(lambda: parent.full_forward(backend, batch), model), model, batch)
        native, absent_readers, absent_outputs, calls["native_sites"] = native_bundle
        donor, donor_heads, calls["donor_source"] = parent.capture_source_heads(
            lambda: parent.full_forward(backend, donor_batch), model, donor_batch)
        writer_bundle, calls["writer_source"] = parent.with_source_heads(
            lambda: capture_sites(lambda: parent.full_forward(backend, batch), model),
            model, batch, base_heads, donor_heads, source_positions)
        writer, present_readers, present_outputs, calls["writer_sites"] = writer_bundle
        basis_cpu, singular = fit_basis(absent_readers, present_readers, rows, positions)
        basis = basis_cpu.to(backend.device)
        projected = {label: project_reader(absent_readers[label], present_readers[label], basis)
                     for label in ("A11", "M11")}
        arm_specs = {
            "writer_replay": {}, "A11_absent": {"A11": absent_readers["A11"]},
            "A11_U8": {"A11": projected["A11"]},
            "M11_absent": {"M11": absent_readers["M11"]},
            "M11_U8": {"M11": projected["M11"]},
        }
        outputs = {}
        for label, replacements in arm_specs.items():
            outputs[label], calls[label] = run_arm(
                backend, batch, base_heads, donor_heads, source_positions, positions, replacements)

        restricted = atlas.mlp_subspace_tensor(model.transformer.h[11].mlp, basis, basis)
        left, right, down, tensor = (restricted[key] for key in ("left", "right", "down", "tensor"))
        scores = torch.linalg.vector_norm(down, dim=0) * torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(right, dim=1)
        factor_order = torch.argsort(scores, descending=True, stable=True)
        top = factor_order[:TOP]
        top_tensor = torch.einsum("an,ni,nj->aij", down[:, top], left[top], right[top])
        tensor_residual = float(torch.linalg.vector_norm(tensor - top_tensor)
                                / max(float(torch.linalg.vector_norm(tensor)), 1e-30))
        curve = {}
        for count in (1, 2, 4, 8, 16, 32, 64, 128):
            chosen = factor_order[:count]
            approximation = torch.einsum("an,ni,nj->aij", down[:, chosen], left[chosen], right[chosen])
            curve[str(count)] = float(torch.linalg.vector_norm(tensor - approximation)
                / max(float(torch.linalg.vector_norm(tensor)), 1e-30))

        mlp = model.transformer.h[11].mlp
        x0, xu = absent_readers["M11"].float(), projected["M11"].float()
        hidden_delta = ((xu @ mlp.Left.weight.detach().float().T)
                        * (xu @ mlp.Right.weight.detach().float().T)
                        - (x0 @ mlp.Left.weight.detach().float().T)
                        * (x0 @ mlp.Right.weight.detach().float().T))
        top_delta = hidden_delta[..., top] @ mlp.Down.weight.detach().float()[:, top].T
        top_output = absent_outputs["M11"] + top_delta.to(absent_outputs["M11"])
        outputs["M11_U8_top32"], calls["M11_U8_top32"] = run_arm(
            backend, batch, base_heads, donor_heads, source_positions, positions,
            {"M11": projected["M11"]}, output_replacement=top_output)

    native_margin, writer_margin = parent.toward_donor_margin(native), parent.toward_donor_margin(writer)
    margins = {label: parent.toward_donor_margin(output) for label, output in outputs.items()}
    replay_error = float(np.max(np.abs(margins["writer_replay"] - writer_margin)))
    reachability = {label: {} for label in ("A11", "M11")}
    for label in reachability:
        delta = present_readers[label] - absent_readers[label]
        for phase in ("FIT", "HOLDOUT"):
            matrix = selected_rows(delta, rows, positions, phase)
            numerator = float(((matrix @ basis_cpu) @ basis_cpu.T).square().sum())
            reachability[label][phase] = numerator / max(float(matrix.square().sum()), 1e-30)
    causal = {phase: {} for phase in ("FIT", "HOLDOUT")}
    for phase in causal:
        chosen = phase_mask(rows, phase)
        for label in ("A11", "M11"):
            full = writer_margin[chosen] - margins[f"{label}_absent"][chosen]
            projected_effect = margins[f"{label}_U8"][chosen] - margins[f"{label}_absent"][chosen]
            causal[phase][label] = parent.metrics(projected_effect, full)
        m11_u8 = margins["M11_U8"][chosen] - margins["M11_absent"][chosen]
        top_effect = margins["M11_U8_top32"][chosen] - margins["M11_absent"][chosen]
        causal[phase]["M11_top32"] = parent.metrics(top_effect, m11_u8)

    gram_error = float((basis_cpu.T @ basis_cpu - torch.eye(RANK)).abs().max())
    generator = torch.Generator(device="cpu").manual_seed(1701)
    coordinate = torch.randn(RANK, generator=generator, dtype=torch.float32).to(backend.device)
    coordinate = coordinate / torch.linalg.vector_norm(coordinate)
    x = basis @ coordinate
    exact = basis.T @ (model.transformer.h[11].mlp.Down.weight.detach().float()
        @ ((model.transformer.h[11].mlp.Left.weight.detach().float() @ x)
           * (model.transformer.h[11].mlp.Right.weight.detach().float() @ x)))
    formula = torch.einsum("aij,i,j->a", tensor, coordinate, coordinate)
    formula_error = float((exact - formula).abs().max())
    width = model.config.n_embd // model.config.n_head
    contractions = {"writers": {}, "H3_readers": {}}
    for layer, head in parent.SOURCE_HEADS:
        sl = slice(head * width, (head + 1) * width)
        contractions["writers"][f"L{layer:02d}H{head:02d}"] = tensor_record(
            basis.T @ model.transformer.h[layer].attn.c_proj.weight.detach().float()[:, sl])
    attention = model.transformer.h[11].attn
    sl = slice(3 * width, 4 * width)
    for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v"):
        contractions["H3_readers"][name] = tensor_record(
            getattr(attention, name).weight.detach().float()[sl] @ basis)
    contractions["H3_output"] = tensor_record(basis.T @ attention.c_proj.weight.detach().float()[:, sl])
    contractions["M11_left"] = tensor_record(left)
    contractions["M11_right"] = tensor_record(right)
    contractions["M11_down"] = tensor_record(down)
    contractions["M11_tensor"] = tensor_record(tensor)
    basis_storage = tensor_record(basis_cpu)
    restored_basis = torch.tensor(basis_storage["values"], dtype=torch.float32).reshape(
        basis_storage["shape"])
    basis_roundtrip_error = float((restored_basis - basis_cpu).abs().max())
    basis_roundtrip_hash_ok = hashlib.sha256(
        restored_basis.contiguous().numpy().tobytes()).hexdigest() == basis_storage["sha256"]
    A = bool(authority_ok and replay_error <= BARS["replay"] and gram_error <= BARS["orthonormal"]
             and basis_roundtrip_error == 0.0 and basis_roundtrip_hash_ok
             and finite({"calls": calls, "reachability": reachability, "causal": causal,
                         "curve": curve, "contractions": contractions})
             and PRICE["model_forwards"] * len(rows) == PRICE["sequence_evaluations"])
    B = all(reachability[label]["HOLDOUT"] >= BARS["energy"] for label in ("A11", "M11"))
    C = all(causal[phase][label]["signed_recovery"] >= BARS["effect_fraction"]
        and causal[phase][label]["cosine"] >= BARS["effect_cosine"]
        for phase in ("FIT", "HOLDOUT") for label in ("A11", "M11"))
    expected_shapes = {"M11_left": [4608, 8], "M11_right": [4608, 8],
                       "M11_down": [8, 4608], "M11_tensor": [8, 8, 8],
                       "H3_output": [8, 128]}
    shapes_ok = (all(contractions[key]["shape"] == shape for key, shape in expected_shapes.items())
        and all(item["shape"] == [8, 128] for item in contractions["writers"].values())
        and all(item["shape"] == [128, 8] for item in contractions["H3_readers"].values()))
    D = bool(formula_error <= BARS["formula"] and shapes_ok)
    E = bool(tensor_residual <= BARS["tensor_residual"]
        and all(causal[phase]["M11_top32"]["signed_recovery"] >= BARS["top_effect_fraction"]
                and causal[phase]["M11_top32"]["cosine"] >= BARS["top_effect_cosine"]
                for phase in ("FIT", "HOLDOUT")))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A or not D else
        "activation_specific_interface" if not B or not C else
        "compact_shared_weight_tensor" if E else "shared_but_diffuse_weight_tensor")
    result = {"schema": "temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "instrument": {"writer_replay_max_abs_margin_error": replay_error,
                       "basis_orthonormal_max_abs_error": gram_error, "calls": calls},
        "basis": basis_storage, "basis_roundtrip_max_abs_error": basis_roundtrip_error,
        "basis_roundtrip_hash_ok": basis_roundtrip_hash_ok,
        "fit_singular_values": singular[:32].tolist(),
        "reachability": reachability, "causal": causal, "contractions": contractions,
        "M11_factor_order": factor_order[:128].detach().cpu().tolist(),
        "M11_factor_score_top128": scores[factor_order[:128]].detach().cpu().tolist(),
        "M11_tensor_relative_residual_curve": curve,
        "M11_top32_tensor_relative_residual": tensor_residual,
        "tensor_formula_max_abs_error": formula_error, "tensor_shapes_ok": shapes_ok,
        "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("instrument", "reachability", "causal",
        "M11_tensor_relative_residual_curve", "M11_top32_tensor_relative_residual",
        "tensor_formula_max_abs_error", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
