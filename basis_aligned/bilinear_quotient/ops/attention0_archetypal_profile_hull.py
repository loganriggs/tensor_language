"""RUNG 439 -- CAUSAL-PROFILE-SCORED ARCHETYPAL Q/K HULL.

Project rung430's score-trained sparse query/key atoms onto exact unsigned,
signed, relaxed-signed, and entry-permuted token-row hulls.  Convex mixtures
use global Frank-Wolfe linear minimization over every FIT token.  Judge
identification in observable rotary score profiles and judge computation on
the previously unopened FINAL documents.

This is a structural identifiability screen, not adoption or semantics.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_archetypal_profile_hull_results.json"
BUNDLE = BQ / "attention0_archetypal_profile_hull_bundle.pt"
PREREG = POLY / "ATTENTION0_ARCHETYPAL_PROFILE_HULL_PREREGISTRATION.md"
PARENT_RESULT = BQ / "attention0_coupled_sparse_qk_score_product_results.json"
PARENT_BUNDLE = BQ / "attention0_coupled_sparse_qk_score_product_bundle.pt"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
R430_PATH = OPS / "attention0_coupled_sparse_qk_score_product.py"
R426_PATH = OPS / "attention0_cross_head_sparse_qk_vocabulary.py"
OV_BASE = OPS / "attention0_ov_downstream_codebook.py"

VOCAB = 50_257
N_HEAD = 9
N_BRANCH = 2
N_ENTRY = 18
HD = 128
SIDE_DIM = 2_304
N_ATOM = 512
K_SIDE = 27
FW_STEPS = 16
RELAX = .25
PROFILE_ANCHORS = 16
PROFILE_OFFSETS = (1, 4, 16, 64)
ARMS = ("U54", "H54", "SH54", "RSH54", "PH54")
QK54_BYTES = 15_583_320
RUNG = 439


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decoder(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _unit_entries(value: torch.Tensor) -> torch.Tensor:
    shaped = value.reshape(value.shape[0], N_ENTRY, HD)
    return F.rms_norm(shaped, (HD,))


def _normal_rows(value: torch.Tensor) -> torch.Tensor:
    return _decoder(value.reshape(value.shape[0], SIDE_DIM))


def _fw_project(target: torch.Tensor, rows: torch.Tensor, signed: bool,
                steps: int = FW_STEPS) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """Global Frank-Wolfe projection; every LMO scans every supplied row."""
    target = _decoder(target.float())
    rows = _normal_rows(rows.float())
    linear = target @ rows.T
    if signed:
        first = linear.abs().argmax(1)
        first_value = linear.gather(1, first[:, None]).squeeze(1)
        first_sign = first_value.sign().masked_fill(first_value == 0, 1.0)
    else:
        first = linear.argmax(1)
        first_sign = torch.ones_like(first, dtype=target.dtype)
    current = rows[first] * first_sign[:, None]
    support = [first]
    signs = [first_sign]
    weights = [torch.ones(N_ATOM, device=target.device, dtype=target.dtype)]
    objective = [float((current - target).square().sum(1).mean())]
    lmo_calls = 1
    final_gap = torch.full((N_ATOM,), float("nan"), device=target.device)

    for _ in range(steps):
        residual = current - target
        linear = residual @ rows.T
        lmo_calls += 1
        if signed:
            chosen = linear.abs().argmax(1)
            chosen_value = linear.gather(1, chosen[:, None]).squeeze(1)
            chosen_sign = -chosen_value.sign().masked_fill(chosen_value == 0, -1.0)
        else:
            chosen = linear.argmin(1)
            chosen_sign = torch.ones_like(chosen, dtype=target.dtype)
        vertex = rows[chosen] * chosen_sign[:, None]
        direction = vertex - current
        gamma = (-(residual * direction).sum(1)
                 / direction.square().sum(1).clamp_min(1e-30)).clamp(0, 1)
        weights = [weight * (1 - gamma) for weight in weights]
        weights.append(gamma)
        support.append(chosen)
        signs.append(chosen_sign)
        current = current + gamma[:, None] * direction
        objective.append(float((current - target).square().sum(1).mean()))
        residual = current - target
        final_gap = 2 * ((current - vertex) * residual).sum(1)

    support_tensor = torch.stack(support, 1)
    sign_tensor = torch.stack(signs, 1)
    weight_tensor = torch.stack(weights, 1)
    selected = rows[support_tensor] * sign_tensor[..., None]
    reconstructed = (weight_tensor[..., None] * selected).sum(1)
    weight_sum_error = float((weight_tensor.sum(1) - 1).abs().max())
    reconstruction_error = float((reconstructed - current).abs().max())
    objective_increases = [objective[i + 1] - objective[i]
                           for i in range(len(objective) - 1)]
    certificate = {
        "signed_symmetric_hull": signed,
        "rows_scanned_per_lmo": rows.shape[0],
        "global_lmo_calls": lmo_calls,
        "support_width": support_tensor.shape[1],
        "minimum_weight": float(weight_tensor.min()),
        "maximum_weight_sum_error": weight_sum_error,
        "materialized_reconstruction_max_abs": reconstruction_error,
        "objective_path": objective,
        "maximum_objective_increase": max(objective_increases, default=0.0),
        "mean_squared_projection_residual": float((current - target).square().sum(1).mean()),
        "median_squared_projection_residual": float((current - target).square().sum(1).median()),
        "mean_final_frank_wolfe_dual_gap": float(final_gap.mean()),
        "max_final_frank_wolfe_dual_gap": float(final_gap.max()),
    }
    proof = {
        "token_row_indices": support_tensor.to(torch.int32).cpu(),
        "signs": sign_tensor.to(torch.int8).cpu(),
        "weights": weight_tensor.cpu(),
        "convex_points": current.cpu(),
    }
    return current, _decoder(current), certificate, proof


def _permuted_fit_rows(rows: torch.Tensor, fit_ids: torch.Tensor, seed: int) -> tuple[torch.Tensor, list[str]]:
    source = rows[fit_ids].reshape(-1, N_ENTRY, HD)
    result = torch.empty_like(source)
    hashes = []
    for entry in range(N_ENTRY):
        generator = torch.Generator(device="cpu").manual_seed(seed + entry)
        permutation = torch.randperm(len(fit_ids), generator=generator).to(rows.device)
        result[:, entry] = source[permutation, entry]
        hashes.append(hashlib.sha256(permutation.cpu().numpy().tobytes()).hexdigest())
    return result.reshape(-1, SIDE_DIM), hashes


def _query_profiles(query: torch.Tensor, key_anchors: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor, apply_rot) -> torch.Tensor:
    query = _unit_entries(query)
    key_anchors = _unit_entries(key_anchors)
    parts = []
    for start in range(0, len(query), 256):
        batch = query[start:start + 256]
        columns = []
        for offset in PROFILE_OFFSETS:
            rotated = apply_rot(batch, cos[offset], sin[offset])
            columns.append(torch.einsum("neh,aeh->nae", rotated, key_anchors) / HD)
        parts.append(torch.cat(columns, 1).flatten(1))
    return F.normalize(torch.cat(parts), dim=1)


def _key_profiles(key: torch.Tensor, query_anchors: torch.Tensor,
                  cos: torch.Tensor, sin: torch.Tensor, apply_rot) -> torch.Tensor:
    key = _unit_entries(key)
    query_anchors = _unit_entries(query_anchors)
    rotated_anchors = {
        offset: apply_rot(query_anchors, cos[offset], sin[offset])
        for offset in PROFILE_OFFSETS}
    parts = []
    for start in range(0, len(key), 256):
        batch = key[start:start + 256]
        columns = [torch.einsum("aeh,neh->nae", rotated_anchors[offset], batch) / HD
                   for offset in PROFILE_OFFSETS]
        parts.append(torch.cat(columns, 1).flatten(1))
    return F.normalize(torch.cat(parts), dim=1)


def _profile_match(left: torch.Tensor, right: torch.Tensor) -> dict:
    from scipy.optimize import linear_sum_assignment
    cosine = (left.double() @ right.double().T).abs().cpu().numpy()
    rows, cols = linear_sum_assignment(-cosine)
    values = torch.tensor(cosine[rows, cols], dtype=torch.float64)
    return {
        "mean_matched_absolute_cosine": float(values.mean()),
        "median_matched_absolute_cosine": float(values.median()),
        "minimum_matched_absolute_cosine": float(values.min()),
        "left_indices": torch.tensor(rows).tolist(),
        "right_indices": torch.tensor(cols).tolist(),
    }


def _same_atom_profile(left: torch.Tensor, right: torch.Tensor) -> dict:
    values = (left.double() * right.double()).sum(1).abs()
    return {"mean_absolute_cosine": float(values.mean()),
            "median_absolute_cosine": float(values.median()),
            "minimum_absolute_cosine": float(values.min())}


def _phase_rotate(value: torch.Tensor, theta: torch.Tensor, apply_rot) -> torch.Tensor:
    shaped = value.reshape(value.shape[0], N_ENTRY, HD)
    return apply_rot(shaped, theta.cos(), theta.sin()).reshape(value.shape[0], SIDE_DIM)


def _materialize(bundle: dict, q_decoder: torch.Tensor,
                 k_decoder: torch.Tensor) -> dict:
    result = {}
    for short, long, decoder in (("q", "query", q_decoder), ("k", "key", k_decoder)):
        indices = bundle[f"{long}54_indices_uint16"]
        coefficients = bundle[f"{long}54_coefficients_fp16"]
        bias = bundle[f"{long}_bias_fp16"].float().cuda()
        physical_decoder = _decoder(decoder.float()).half().float().cuda()
        chunks = []
        for start in range(0, VOCAB, 512):
            idx = indices[start:start + 512].long().cuda()
            coef = coefficients[start:start + 512].float().cuda()
            raw = bias + (coef[..., None] * physical_decoder[idx]).sum(1)
            chunks.append(_unit_entries(raw))
        result[short] = torch.cat(chunks)
        result[f"{short}_decoder"] = physical_decoder
        result[f"{short}_bias"] = bias
        result[f"{short}_indices"] = indices
        result[f"{short}_coefficients"] = coefficients
    return result


def _tables(encoded: dict) -> dict[str, torch.Tensor]:
    result = {}
    for branch in range(N_BRANCH):
        entries = torch.arange(branch, N_ENTRY, N_BRANCH, device=encoded["q"].device)
        result[f"q{branch + 1}"] = encoded["q"][:, entries]
        result[f"k{branch + 1}"] = encoded["k"][:, entries]
    return result


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (PREREG, PARENT_RESULT, PARENT_BUNDLE, ROWS_RECEIPT,
                     R430_PATH, R426_PATH, OV_BASE):
            assert path.exists(), path
        parent = json.loads(PARENT_RESULT.read_text())
        assert parent["rung"] == 430 and parent["instrument"]["artifact_checks"]["bills"]
        assert parent["bundle"]["file_sha256"] == _sha256(PARENT_BUNDLE)
        receipt = json.loads(ROWS_RECEIPT.read_text())
        assert receipt["entries"]["FINAL"]["shape"] == [96, 257]
        assert not parent["FINAL_opened"]
        assert SIDE_DIM == N_ENTRY * HD and QK54_BYTES == 15_583_320
        print("ATTENTION0 ARCHETYPAL PROFILE HULL | dry run: parent, FINAL, hull arms, bill valid")
        return

    started = time.time()
    torch.set_float32_matmul_precision("high")
    sys.path[:0] = [str(QK), str(POLY), str(OPS)]
    from tier2_model import load_elriggs, rope_tables, apply_rot
    from tier2_folding import branch_factors, scores_from_factors
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring

    r430 = _load_module("archetype_r430", R430_PATH)
    r426 = _load_module("archetype_r426", R426_PATH)
    ov_base = _load_module("archetype_ov", OV_BASE)
    edge_mod = _load_module("archetype_edge", OPS / "attention0_realized_edge_block_term.py")
    parent_result = json.loads(PARENT_RESULT.read_text())
    parent_bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=False)
    receipt = json.loads(ROWS_RECEIPT.read_text())
    parent_hash = _sha256(PARENT_BUNDLE)
    parent_hash_match = parent_hash == parent_result["bundle"]["file_sha256"]

    model, config = load_elriggs("bilin18", device="cuda", dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    factors = {branch: branch_factors(model, branch, dtype=torch.float32)
               for branch in (1, 2)}
    entries = r426._entries_from_factors(factors)
    q, k = r430._split(entries)
    ids = torch.arange(VOCAB, device="cuda")
    fit_ids = ids[ids.remainder(5) != 4]
    select_ids = ids[ids.remainder(5) == 4]
    fit_a = ids[ids.remainder(5) <= 1]
    fit_b = ids[(ids.remainder(5) == 2) | (ids.remainder(5) == 3)]

    generator = torch.Generator(device="cpu").manual_seed(43_901)
    anchor_ids = fit_ids[torch.randperm(
        len(fit_ids), generator=generator)[:PROFILE_ANCHORS].cuda()]
    anchor_hash = hashlib.sha256(anchor_ids.cpu().numpy().tobytes()).hexdigest()
    cos, sin = rope_tables(max(PROFILE_OFFSETS) + 1, HD, "cuda", torch.float32, "bf16")

    primary_q = _decoder(parent_bundle["query_decoder_fp16"].float().cuda())
    primary_k = _decoder(parent_bundle["key_decoder_fp16"].float().cuda())
    print("TRAIN frozen-seed independent rung430 restart", flush=True)
    repeat_warm = r430._warm(q, k, fit_ids, 432)
    repeat_model = r430._fine(
        q, k, fit_ids, repeat_warm, "product", 433, rope_tables, apply_rot)
    repeat_q = _decoder(repeat_model["q_decoder"])
    repeat_k = _decoder(repeat_model["k_decoder"])

    perm_q, perm_q_hashes = _permuted_fit_rows(q, fit_ids, 439_100)
    perm_k, perm_k_hashes = _permuted_fit_rows(k, fit_ids, 439_200)
    projections = {}
    proofs = {}

    def project_pair(label, tq, tk, qrows, krows, signed):
        print(f"PROJECT {label} q over {len(qrows)} rows", flush=True)
        qc, qd, qcert, qproof = _fw_project(tq, qrows, signed)
        print(f"PROJECT {label} k over {len(krows)} rows", flush=True)
        kc, kd, kcert, kproof = _fw_project(tk, krows, signed)
        projections[label] = {"q_convex": qc, "q_decoder": qd, "q_certificate": qcert,
                              "k_convex": kc, "k_decoder": kd, "k_certificate": kcert}
        proofs[label] = {"q": qproof, "k": kproof}

    project_pair("H54", primary_q, primary_k, q[fit_ids], k[fit_ids], False)
    project_pair("SH54", primary_q, primary_k, q[fit_ids], k[fit_ids], True)
    project_pair("PH54", primary_q, primary_k, perm_q, perm_k, True)
    project_pair("SH54_repeat", repeat_q, repeat_k, q[fit_ids], k[fit_ids], True)
    project_pair("SH54_fit_a", primary_q, primary_k, q[fit_a], k[fit_a], True)
    project_pair("SH54_fit_b", primary_q, primary_k, q[fit_b], k[fit_b], True)

    projections["RSH54"] = {}
    for side, original in (("q", primary_q), ("k", primary_k)):
        convex = projections["SH54"][f"{side}_convex"]
        relaxed = convex + RELAX * (original - convex)
        projections["RSH54"][f"{side}_decoder"] = _decoder(relaxed)
        projections["RSH54"][f"{side}_distance_fraction"] = float(
            torch.linalg.vector_norm(relaxed - convex)
            / torch.linalg.vector_norm(original - convex).clamp_min(1e-30))

    q_anchor, k_anchor = q[anchor_ids], k[anchor_ids]
    profiles = {}
    decoder_pairs = {
        "U54": (primary_q, primary_k),
        "U54_repeat": (repeat_q, repeat_k),
        "SH54": (projections["SH54"]["q_decoder"], projections["SH54"]["k_decoder"]),
        "SH54_repeat": (projections["SH54_repeat"]["q_decoder"],
                         projections["SH54_repeat"]["k_decoder"]),
        "SH54_fit_a": (projections["SH54_fit_a"]["q_decoder"],
                        projections["SH54_fit_a"]["k_decoder"]),
        "SH54_fit_b": (projections["SH54_fit_b"]["q_decoder"],
                        projections["SH54_fit_b"]["k_decoder"]),
    }
    for name, (qd, kd) in decoder_pairs.items():
        profiles[name] = {
            "q": _query_profiles(qd, k_anchor, cos, sin, apply_rot),
            "k": _key_profiles(kd, q_anchor, cos, sin, apply_rot),
        }
    stability = {
        "unconstrained": {
            side: _profile_match(profiles["U54"][side], profiles["U54_repeat"][side])
            for side in ("q", "k")},
        "signed_hull": {
            side: _profile_match(profiles["SH54"][side], profiles["SH54_repeat"][side])
            for side in ("q", "k")},
        "fit_half_same_primary_atom": {
            side: _same_atom_profile(profiles["SH54_fit_a"][side],
                                     profiles["SH54_fit_b"][side])
            for side in ("q", "k")},
    }

    phase_generator = torch.Generator(device="cpu").manual_seed(439_300)
    theta = (2 * torch.pi * torch.rand(
        N_ENTRY, HD // 2, generator=phase_generator, dtype=torch.float64)).cuda()
    exact_cos, exact_sin = rope_tables(
        max(PROFILE_OFFSETS) + 1, HD, "cuda", torch.float64, "exact")
    exact_primary_q = primary_q.double()
    exact_primary_k = primary_k.double()
    exact_q_anchor = q_anchor.double()
    exact_k_anchor = k_anchor.double()
    gauge_reference_q = _query_profiles(
        exact_primary_q, exact_k_anchor, exact_cos, exact_sin, apply_rot)
    gauge_reference_k = _key_profiles(
        exact_primary_k, exact_q_anchor, exact_cos, exact_sin, apply_rot)
    rotated_q_anchor = _phase_rotate(exact_q_anchor, theta, apply_rot)
    rotated_k_anchor = _phase_rotate(exact_k_anchor, theta, apply_rot)
    rotated_primary_q = _phase_rotate(exact_primary_q, theta, apply_rot)
    rotated_primary_k = _phase_rotate(exact_primary_k, theta, apply_rot)
    gauge_q = _query_profiles(
        rotated_primary_q, rotated_k_anchor, exact_cos, exact_sin, apply_rot)
    gauge_k = _key_profiles(
        rotated_primary_k, rotated_q_anchor, exact_cos, exact_sin, apply_rot)
    gauge_max = max(float((gauge_q - gauge_reference_q).abs().max()),
                    float((gauge_k - gauge_reference_k).abs().max()))

    encoded = {
        "U54": _materialize(parent_bundle, primary_q, primary_k),
        "H54": _materialize(parent_bundle, projections["H54"]["q_decoder"],
                            projections["H54"]["k_decoder"]),
        "SH54": _materialize(parent_bundle, projections["SH54"]["q_decoder"],
                             projections["SH54"]["k_decoder"]),
        "RSH54": _materialize(parent_bundle, projections["RSH54"]["q_decoder"],
                              projections["RSH54"]["k_decoder"]),
        "PH54": _materialize(parent_bundle, projections["PH54"]["q_decoder"],
                             projections["PH54"]["k_decoder"]),
    }
    tables = {name: _tables(value) for name, value in encoded.items()}
    target_tables = {"q1": factors[1][0][:VOCAB], "k1": factors[1][1][:VOCAB],
                     "q2": factors[2][0][:VOCAB], "k2": factors[2][1][:VOCAB]}
    select_score = r426._score_metrics(
        target_tables, tables, select_ids, rope_tables, apply_rot)

    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    final_rows = rows_parent.load_role(receipt["entries"]["FINAL"])
    select_rows_hash = rows_parent.rows_life.base.tensor_sha256(select_rows)
    final_rows_hash = rows_parent.rows_life.base.tensor_sha256(final_rows)
    row_hashes_match = (
        select_rows_hash == receipt["entries"]["SELECT"]["tensor_sha256"]
        and final_rows_hash == receipt["entries"]["FINAL"]["tensor_sha256"])
    final_token_ids = torch.unique(final_rows[:, :-1]).cuda()
    final_score = r426._score_metrics(
        target_tables, tables, final_token_ids, rope_tables, apply_rot)
    r426.ARMS = ARMS
    tables["G54"] = tables["U54"]
    select_documents, select_no_native = r426._document_metrics(
        model, select_rows, tables, ov_base, edge_mod, scoring, scores_from_factors)
    final_documents, final_no_native = r426._document_metrics(
        model, final_rows, tables, ov_base, edge_mod, scoring, scores_from_factors)
    del tables["G54"]

    parent_pattern = parent_result["select_random_pair_score_metrics"]["CP54"][
        "complete_pattern_relative_squared_error"]
    parent_ce = parent_result["select_document_metrics"]["ce"]["CP54"]["damage"]
    bridge_pattern = select_score["U54"]["complete_pattern_relative_squared_error"]
    bridge_ce = select_documents["ce"]["U54"]["damage"]

    certificate_items = []
    for label in ("H54", "SH54", "PH54", "SH54_repeat", "SH54_fit_a", "SH54_fit_b"):
        for side in ("q", "k"):
            certificate_items.append(projections[label][f"{side}_certificate"])
    certificate_valid = all(
        item["minimum_weight"] >= -1e-8
        and item["maximum_weight_sum_error"] <= 2e-6
        and item["materialized_reconstruction_max_abs"] <= 2e-6
        and item["maximum_objective_increase"] <= 1e-6
        and item["global_lmo_calls"] == FW_STEPS + 1
        and torch.isfinite(torch.tensor(item["mean_final_frank_wolfe_dual_gap"]))
        for item in certificate_items)

    artifact = {
        "schema": "attention0_archetypal_profile_hull_bundle_v1",
        "parent_bundle_sha256": parent_hash,
        "anchor_ids": anchor_ids.cpu(),
        "anchor_ids_sha256": anchor_hash,
        "projected_decoders_fp16": {
            arm: {side: projections[arm][f"{side}_decoder"].half().cpu()
                  for side in ("q", "k")}
            for arm in ("H54", "SH54", "RSH54", "PH54")},
        "convex_certificates": proofs,
        "per_arm_deployed_price_bytes": QK54_BYTES,
        "transitive_parent_bundle": str(PARENT_BUNDLE),
    }
    torch.save(artifact, BUNDLE)
    decoder_shapes_dtypes = all(
        tensor.dtype == torch.float16 and tuple(tensor.shape) == (N_ATOM, SIDE_DIM)
        for arm in artifact["projected_decoders_fp16"].values()
        for tensor in arm.values())
    relaxed_exact = all(abs(projections["RSH54"][f"{side}_distance_fraction"] - RELAX) <= 2e-6
                        for side in ("q", "k"))

    u_pattern = final_score["U54"]["complete_pattern_relative_squared_error"]
    u_write = final_documents["full_attention0_write_relative_squared_error"]["U54"]
    u_consumer = final_documents["mean_consumer_relative_squared_error"]["U54"]
    u_ce = final_documents["ce"]["U54"]["damage"]

    pred_a = (
        parent_hash_match
        and abs(bridge_pattern - parent_pattern) <= .005
        and abs(bridge_ce - parent_ce) <= .002
        and certificate_valid and decoder_shapes_dtypes and relaxed_exact
        and max(select_no_native, final_no_native) <= 1e-12
        and gauge_max <= 2e-5
        and len(final_rows) == 96 and row_hashes_match and QK54_BYTES == 15_583_320)
    pred_b = all(
        projections["SH54"][f"{side}_certificate"]["mean_squared_projection_residual"]
        <= .90 * projections["PH54"][f"{side}_certificate"]["mean_squared_projection_residual"]
        for side in ("q", "k"))

    def consequence(arm: str, ratio: float, ce_slack: float) -> bool:
        return (
            final_score[arm]["complete_pattern_relative_squared_error"] <= ratio * u_pattern
            and final_documents["full_attention0_write_relative_squared_error"][arm] <= ratio * u_write
            and final_documents["mean_consumer_relative_squared_error"][arm] <= ratio * u_consumer
            and final_documents["ce"][arm]["damage"] <= u_ce + ce_slack)

    pred_c = consequence("SH54", 1.25, .005) and consequence("RSH54", 1.10, .003)
    pred_d = all(
        stability["signed_hull"][side]["median_matched_absolute_cosine"] >= .60
        and stability["signed_hull"][side]["median_matched_absolute_cosine"]
            >= 1.20 * stability["unconstrained"][side]["median_matched_absolute_cosine"]
        and stability["fit_half_same_primary_atom"][side]["median_absolute_cosine"] >= .80
        for side in ("q", "k"))
    null = (
        not pred_a
        or any(projections["SH54"][f"{side}_certificate"]["mean_squared_projection_residual"]
               >= projections["PH54"][f"{side}_certificate"]["mean_squared_projection_residual"]
               for side in ("q", "k"))
        or any(stability["signed_hull"][side]["median_matched_absolute_cosine"]
               <= stability["unconstrained"][side]["median_matched_absolute_cosine"]
               for side in ("q", "k"))
        or final_documents["ce"]["RSH54"]["damage"] > u_ce + .020
        or final_documents["full_attention0_write_relative_squared_error"]["RSH54"]
            >= 1.50 * u_write)

    result = {
        "status": "attention0_archetypal_profile_hull_complete",
        "rung": RUNG,
        "claim_level": "structural_identifiability_screen_not_adoption_or_semantics",
        "convention": "CE added above native; lower is better",
        "instrument": {
            "parent_bundle_sha256": parent_hash,
            "parent_bundle_hash_matches": parent_hash_match,
            "select_bridge_pattern_parent_reproduced": [parent_pattern, bridge_pattern],
            "select_bridge_ce_parent_reproduced": [parent_ce, bridge_ce],
            "hull_certificates_valid": certificate_valid,
            "projected_decoder_shapes_dtypes_valid": decoder_shapes_dtypes,
            "relaxed_distance_fraction_exact": relaxed_exact,
            "no_native_qk_suffix_replay_relative_squared_error": {
                "SELECT": select_no_native, "FINAL": final_no_native},
            "legal_rotary_gauge_profile_max_abs": gauge_max,
            "final_rows_opened": len(final_rows),
            "select_rows_tensor_sha256": select_rows_hash,
            "final_rows_tensor_sha256": final_rows_hash,
            "registered_row_hashes_match": row_hashes_match,
            "anchor_ids": anchor_ids.cpu().tolist(),
            "anchor_ids_sha256": anchor_hash,
            "fit_counts": {"all": len(fit_ids), "a": len(fit_a), "b": len(fit_b),
                           "select": len(select_ids)},
            "permuted_entry_hashes": {"q": perm_q_hashes, "k": perm_k_hashes},
        },
        "hull_projection": {
            label: {side: projections[label][f"{side}_certificate"]
                    for side in ("q", "k")}
            for label in ("H54", "SH54", "PH54", "SH54_repeat",
                          "SH54_fit_a", "SH54_fit_b")},
        "relaxed_signed_hull": {
            "relaxation_fraction": RELAX,
            "measured_distance_fraction": {
                side: projections["RSH54"][f"{side}_distance_fraction"]
                for side in ("q", "k")}},
        "observable_profile_stability": stability,
        "select_score_metrics": select_score,
        "select_document_metrics": select_documents,
        "final_score_metrics": final_score,
        "final_document_metrics": final_documents,
        "literal_raw_tensor_bytes_per_arm": QK54_BYTES,
        "native_layer0_qk_retained": False,
        "training_certificate_not_runtime_input": True,
        "bundle": {"path": str(BUNDLE), "file_sha256": _sha256(BUNDLE)},
        'pred_a_valid_isolated_instrument': bool(pred_a),
        'pred_b_real_signed_hull_is_geometrically_specific': bool(pred_b),
        'pred_c_anchoring_preserves_computation': bool(pred_c),
        'pred_d_anchoring_improves_gauge_safe_identification': bool(pred_d),
        "strong_null_archetypal_prior_is_not_useful_here": bool(null),
        "fresh_extraction_removal_gate_licensed": bool(
            pred_a and pred_b and pred_c and pred_d and not null),
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 1,
        "config": config,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "projection_residuals": {
            label: {side: result["hull_projection"][label][side][
                "mean_squared_projection_residual"] for side in ("q", "k")}
            for label in ("SH54", "PH54")},
        "profile_stability": stability,
        "final_pattern": {arm: final_score[arm][
            "complete_pattern_relative_squared_error"] for arm in ARMS},
        "final_write": final_documents["full_attention0_write_relative_squared_error"],
        "final_consumer": final_documents["mean_consumer_relative_squared_error"],
        "final_ce": {arm: final_documents["ce"][arm]["damage"] for arm in ARMS},
        "predicates": [pred_a, pred_b, pred_c, pred_d],
        "null": null,
        "licensed": result["fresh_extraction_removal_gate_licensed"],
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ATTENTION0 ARCHETYPAL PROFILE HULL DONE", flush=True)


if __name__ == "__main__":
    main()
