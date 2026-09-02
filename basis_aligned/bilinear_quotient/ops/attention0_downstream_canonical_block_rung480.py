#!/usr/bin/env python3
"""RUNG480 -- gauge-covariant downstream directions in attention0's r424 block.

The exact affine 7x7x33 triplet expansion is retained as an audit object.  Scientific
selection uses sym(z outer dCE/dz), which transforms by conjugation under every legal
orthogonal change of latent coordinates.  Discovery families only; no causal claim.
"""

# BQGATE: EXPERIMENT
# pred_a exact affine block, query-local gradients, hashes, and gauge identity
# pred_b downstream use fixes at least two reproducible latent projectors
# pred_c one projector slab has stable circuit-labelled response above controls
# pred_d winning slab survives leave-one-discovery-family checks
# pred_e slab and varying-space complement distinguish downstream uses

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
QK = ROOT.parent / "qk_mdl"
for path in (POLY, ROOT, ROOT / "ops", QK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import attention0_realized_edge_block_term as edge_parent
import attention0_ov_downstream_codebook as ov_parent
import equality_product_circuit_response_graph_rung477b as circuit_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent
import bilin18_observed_model_facade as facade
import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
import scoring
from tier2_model import rope_tables, apply_rot


PREREG = POLY / "ATTENTION0_DOWNSTREAM_CANONICAL_BLOCK_RUNG480_PREREGISTRATION.md"
R424_RESULT = ROOT / "attention0_realized_edge_block_term_results.json"
R424_SOURCE = ROOT / "ops/attention0_realized_edge_block_term.py"
R425_RESULT = ROOT / "attention0_edge_block_fresh_rows_results.json"
R425_SOURCE = ROOT / "ops/attention0_edge_block_fresh_rows.py"
R479_RESULT = ROOT / "equality_task_reader_commutant_rung479_results.json"
R479_SOURCE = ROOT / "ops/equality_task_reader_commutant_rung479.py"
R477B_RESULT = ROOT / "equality_product_circuit_response_graph_rung477b_results.json"
R477B_SOURCE = ROOT / "ops/equality_product_circuit_response_graph_rung477b.py"
ROWS_RECEIPT = ROOT / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
FRESH_RECEIPT = ROOT / "mlp2_rank512_refit_v1_rows_receipt.json"
OUT = ROOT / "attention0_downstream_canonical_block_rung480_results.json"
BUNDLE = ROOT / "attention0_downstream_canonical_block_rung480_bundle.pt"
HASHES = {
    PREREG: "ef8da89fb0bda4d5d64a70bc9cfc40873aeb557f97489ac68230a99918d40b9a",
    R424_RESULT: "33facaa3a8a4c12861f024824124174f5d7f45533232f264dbde15921c751e42",
    R424_SOURCE: "14c8dc91de45e7cf99dcf3033e8c3999e6fa7e50aa9a2effb6185ba2b0231b96",
    R425_RESULT: "3f6684b3102078f5ac0fab1b0f65f888a3630ab368f4d017c102d9a7872d1e71",
    R425_SOURCE: "d0de3260331af3d820dee5a190a9dacdf4dd59105faaa2ec0b64f79adf994b25",
    R479_RESULT: "39689d65413442e123930ab00317d689c0e3ecdacdc04350845e21ed59252ecc",
    R479_SOURCE: "f784437a8f5badc54999bab5f84788068068a84fc811086b5923ad96e31fbb18",
    R477B_RESULT: "38349612eb9ca8cf480afe63a1c9cad8c258948ed64383680f42dcf7876a2191",
    R477B_SOURCE: "ebf9c91e0a823cd263ec997ff185822323d41aadb5f53cdee031bfc8c908cd6b",
}

D = 1152
U = 16
MODE_DIMS = (6, 6, 32)
AUG_DIMS = (7, 7, 33)
MODE_NAMES = ("score_branch_1", "score_branch_2", "payload")
SOURCES = circuit_parent.SOURCES
MASK_TYPES = circuit_parent.MASK_TYPES
HALVES = circuit_parent.HALVES
DISCOVERY_STOP = circuit_parent.DISCOVERY_STOP
BATCH = circuit_parent.BATCH
CONTROL_SEEDS = tuple(range(2026090260, 2026090276))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine(left, right) -> float:
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    return float(torch.dot(left, right) / torch.linalg.vector_norm(left).mul(
        torch.linalg.vector_norm(right)).clamp_min(1e-30))


def _projector_overlap(left, right) -> float:
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    return float(torch.trace(left @ right) / max(float(torch.trace(left)), 1e-30))


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    r424 = json.loads(R424_RESULT.read_text())
    r425 = json.loads(R425_RESULT.read_text())
    r479 = json.loads(R479_RESULT.read_text())
    if not all(r424.get(key) is True for key in (
        "pred_a_instrument_valid", "pred_b_complete_edge_joint_block",
        "pred_c_downstream_transport", "pred_d_reproducible_specific_coupling",
    )) or r424.get("selected_restart") != 0:
        raise RuntimeError("rung424 positive changed")
    if not all(r425.get(key) is True for key in (
        "pred_a_instrument_and_select_bridge", "pred_b_fresh_row_block_replicates",
        "pred_c_controls_keep_ordering_off_rows",
    )) or r425.get("null_424_block_is_row_specific") is not False:
        raise RuntimeError("rung425 replication changed")
    if r479.get("pred_a_lawful_collection") is not True or any(
        r479.get(key) is not False for key in (
            "pred_b_nontrivial_shared_algebra", "pred_c_fit_block_structure",
            "pred_d_cross_view_blocks", "pred_e_circuit_labelled_block",
        )) or r479.get("strong_null") is not True:
        raise RuntimeError("rung479 route changed")
    rows, positive, masks, scale, tags, validation_tags, metadata, _ = \
        circuit_parent.validate_inputs()
    expected_per_fit = 0
    for half_start, half_stop in HALVES:
        row_gate = torch.zeros(circuit_parent.DOCUMENTS, circuit_parent.TOKENS, dtype=torch.bool)
        row_gate[half_start:half_stop] = True
        flat_gate = row_gate.flatten()
        expected_per_fit += sum(int((masks[tag][mask_type] & flat_gate).sum())
                                for tag in tags for mask_type in MASK_TYPES)
    expected_per_fit *= len(SOURCES)
    return rows, positive, masks, scale, tags, validation_tags, {
        **metadata, "expected_backwards_per_fit": expected_per_fit,
        "expected_fit_forwards": math.ceil(DISCOVERY_STOP / BATCH) * len(SOURCES),
        "r424_result_sha256": sha256(R424_RESULT),
        "r425_result_sha256": sha256(R425_RESULT),
        "r479_result_sha256": sha256(R479_RESULT),
    }


def _affine_parts(block):
    output = []
    for mean_key, basis_key in (("mean1", "basis1"), ("mean2", "basis2"),
                                ("meanv", "basisv")):
        mean, basis = block[mean_key].float(), block[basis_key].float()
        # _project(x) = mean + ((x-mean) @ basis) @ basis.T: the constant
        # augmented column is the full affine mean, not only its orthogonal part.
        output.append((mean, basis, torch.cat((mean.T, basis), dim=1)))
    return output


def _coordinates(score1, score2, tokens, all_payload, block):
    parts = _affine_parts(block)
    flat1 = score1.permute(0, 2, 3, 1)
    flat2 = score2.permute(0, 2, 3, 1)
    payload_native = all_payload[tokens].float()
    payload = payload_native.reshape(*payload_native.shape[:-2], -1)
    coords = []
    for value, (mean, basis, _aug) in zip((flat1, flat2, payload), parts):
        variable = (value.reshape(-1, value.shape[-1]) - mean) @ basis
        variable = variable.reshape(*value.shape[:-1], basis.shape[1])
        coords.append(torch.cat((torch.ones_like(variable[..., :1]), variable), dim=-1))
    aug1, aug2, augv = (row[2] for row in parts)
    core = torch.einsum(
        "hi,hj,huk->ijku", aug1, aug2, augv.reshape(edge_parent.N_HEAD, U, -1))
    return coords, core


def _align_block(reference, other):
    aligned = dict(other)
    maps = []
    for key in ("basis1", "basis2", "basisv"):
        target, source = reference[key].float(), other[key].float()
        u, _s, vh = torch.linalg.svd(source.T @ target)
        rotation = u @ vh
        aligned[key] = source @ rotation
        maps.append(rotation)
    return aligned, maps


def rebuild_block(model):
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    fresh_entry = json.loads(FRESH_RECEIPT.read_text())["entries"]["EVALUATION"]
    fresh_rows = torch.load(fresh_entry["path"], weights_only=True)[96:192].contiguous()
    block0 = model.transformer.h[0]
    with torch.no_grad():
        captured = ov_parent._capture_cproj_input(model, fit_rows, torch.device("cuda")).cuda()
        weight = block0.attn.c_proj.weight.detach().float()
        a_factor, _ = ov_parent._asvd(weight, captured)
        interface = torch.linalg.qr(a_factor[:, :U].float(), mode="reduced").Q.cuda()
        embedding = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))[:ov_parent.VOCAB]
        payload_error = ov_parent._payload_exactness(model, embedding)
        all_payload = ov_parent._payload_codes(model, interface, embedding)
        writes = []
        for start in range(0, len(fit_rows), edge_parent.DOC_BATCH):
            tokens = fit_rows[start:start + edge_parent.DOC_BATCH, :-1].cuda()
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
            attention0, _ = block0.attn(F.rms_norm(token_base, (D,)), None)
            writes.append(attention0.float()[:, edge_parent.POSITIONS].reshape(-1, D))
        writes = torch.cat(writes)
        sigma = torch.sqrt((writes @ interface).double().square().mean(0)).float()
        gram, _normalizers, _live = ov_parent._response_metric(
            model, fit_rows, interface, sigma, None, torch.device("cuda"))
        metric_factor, _ = ov_parent._metric_factor(gram)
        fit_edges = edge_parent._collect_edges(model, fit_rows, rope_tables, apply_rot)
    initial = edge_parent._fit_initial(fit_edges, all_payload)
    restarts = [edge_parent._optimize(
        fit_edges, all_payload, initial, metric_factor, seed) for seed in (422, 423)]
    selected_index = min(range(2), key=lambda index: restarts[index]["fit_objective"])
    main = restarts[selected_index]
    refit = restarts[1 - selected_index]
    half_models = []
    for low, high, seed in ((0, 48, 422_048), (48, 96, 422_096)):
        mask = (fit_edges["document"] >= low) & (fit_edges["document"] < high)
        half_initial = edge_parent._fit_initial(fit_edges, all_payload, mask)
        half_models.append(edge_parent._optimize(
            fit_edges, all_payload, half_initial, metric_factor, seed, mask))
    marginal = {key: initial[key] for key in
                ("mean1", "basis1", "mean2", "basis2", "meanv", "basisv")}
    models = {"marginal": marginal, "joint": main, "deranged": main}
    with torch.no_grad():
        select_edges = edge_parent._collect_edges(model, select_rows, rope_tables, apply_rot)
        fresh_edges = edge_parent._collect_edges(model, fresh_rows, rope_tables, apply_rot)
        select_metric = edge_parent._edge_metrics(
            select_edges, all_payload, main, metric_factor, "joint")
        fresh_metric = edge_parent._edge_metrics(
            fresh_edges, all_payload, main, metric_factor, "joint")
        select_transport = edge_parent._document_transport(
            model, select_rows, interface, all_payload, models,
            rope_tables, apply_rot, ov_parent, scoring)
        fresh_transport = edge_parent._document_transport(
            model, fresh_rows, interface, all_payload, models,
            rope_tables, apply_rot, ov_parent, scoring)
    aligned_refit, maps = _align_block(main, refit)
    half_overlap = {}
    for name, key in zip(MODE_NAMES, ("basis1", "basis2", "basisv")):
        half_overlap[name] = edge_parent._projector_overlap(
            half_models[0][key], half_models[1][key])
    return {
        "interface": interface, "all_payload": all_payload, "main": main,
        "refit": aligned_refit, "refit_maps": maps, "fit_edges": fit_edges,
        "payload_error": payload_error, "selected_index": selected_index,
        "half_overlap": half_overlap, "select_metric": select_metric,
        "fresh_metric": fresh_metric,
        "select_routed": select_transport["routed_u16_r2"]["joint"],
        "fresh_routed": fresh_transport["routed_u16_r2"]["joint"],
    }


def _forward_fitted(model, tokens, arm, scale, interface, all_payload, block):
    analytical = arm != "native"
    cached_early, holder = {}, {}
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0,
             "fitted_attention0": 0}
    max_reconstruction = 0.0

    def attention(event):
        nonlocal max_reconstruction
        if event.site == 0:
            native, next_value = event.block.attn(event.state, event.first_value)
            score1, score2 = edge_parent._score_halves(
                event.block, event.state, rope_tables, apply_rot)
            fitted_u = edge_parent._reconstruct_u(
                score1, score2, event.tokens, all_payload, block, "joint").float()
            coords, core = _coordinates(score1, score2, event.tokens, all_payload, block)
            leaf = fitted_u.detach().requires_grad_(True)
            native_u = native.float() @ interface.float()
            write = native.float() + (leaf - native_u) @ interface.float().T
            holder.update({"leaf": leaf, "coords": tuple(row.detach() for row in coords),
                           "core": core.detach(), "fitted_u": fitted_u.detach()})
            audit["fitted_attention0"] += 1
            return write.to(native.dtype), next_value
        if analytical and event.site in audit_parent.stage1.SITE_HEADS:
            write, factors, support, reconstruction = \
                audit_parent.factor_parent._factor_site(
                    event.state, event.first_value, event.block.attn, event.site, event.tokens)
            max_reconstruction = max(max_reconstruction, reconstruction)
            audit["replayed_attention"] += 1
            if arm != "replay":
                early, late = audit_parent.PAIR
                early_site = audit_parent.factor_parent.TERMS[early][1]
                late_site = audit_parent.factor_parent.TERMS[late][1]
                if event.site == early_site:
                    cached_early.update(factors[early])
                    write = write - factors[early]["native_term"]
                if event.site == late_site:
                    if not cached_early:
                        raise RuntimeError("early factors missing")
                    late_factor = factors[late]
                    if arm != "reference":
                        write = write - late_factor["native_term"]
                        if arm == "score":
                            score = cached_early["p"] * scale["score_ratio"]
                            write = write + torch.bmm(score * support, late_factor["u"]).to(write.dtype)
            return write, event.first_value
        write, next_value = event.block.attn(event.state, event.first_value)
        audit["native_attention"] += 1
        return write, next_value

    def mlp(event):
        audit["native_mlp"] += 1
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    if set(holder) != {"leaf", "coords", "core", "fitted_u"}:
        raise RuntimeError("attention0 fitted leaf not captured exactly once")
    return logits, holder, audit, max_reconstruction


def _target_contributions(holder, gradient, row, query, keep_triplets):
    a, b, c = holder["coords"]
    a, b, c = a[row, query, :query + 1], b[row, query, :query + 1], c[row, :query + 1]
    g, core = gradient[row, query].float(), holder["core"].float()
    core_g = torch.einsum("ijku,u->ijk", core, g)
    grad_a = torch.einsum("ijk,sj,sk->si", core_g, b, c)
    grad_b = torch.einsum("ijk,si,sk->sj", core_g, a, c)
    grad_c = torch.einsum("ijk,si,sj->sk", core_g, a, b)
    operators = []
    for z, dz in ((a[:, 1:], grad_a[:, 1:]), (b[:, 1:], grad_b[:, 1:]),
                  (c[:, 1:], grad_c[:, 1:])):
        raw = -(z.T @ dz)
        operators.append((raw + raw.T) / 2)
    triplet = None
    if keep_triplets:
        moment = torch.einsum("si,sj,sk->ijk", a, b, c)
        triplet = -(moment * core_g)
    direct = -torch.dot(g, holder["fitted_u"][row, query].float())
    return operators, triplet, direct


def collect_response(model, rows, masks, scale, tags, block_state, block, *, keep_triplets):
    operator_sums = [torch.zeros(
        2, len(SOURCES), len(MASK_TYPES), len(tags), dim, dim, dtype=torch.float64)
        for dim in MODE_DIMS]
    triplet_sums = torch.zeros(
        2, len(SOURCES), len(MASK_TYPES), len(tags), *AUG_DIMS, dtype=torch.float64
    ) if keep_triplets else None
    counts = torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    backwards, reconstruction = 0, 0.0
    identity_num = identity_den = trace_num = trace_den = 0.0
    forwards = 0
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    device = next(model.parameters()).device
    for start in range(0, DISCOVERY_STOP, BATCH):
        stop = min(start + BATCH, DISCOVERY_STOP)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        if keep_triplets:
            with torch.no_grad():
                native, _, _audit, _ = source_parent.run_forward(model, tokens, arm="native")
                replayed, _, _audit, error = source_parent.run_forward(model, tokens, arm="replay")
            difference = replayed - native
            replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
            replay["relative_squared"] = max(
                replay["relative_squared"],
                float(difference.square().sum()) / max(float(native.square().sum()), 1e-30))
            reconstruction = max(reconstruction, error)
            forwards += 2
        active = []
        for hi, (half_start, half_stop) in enumerate(HALVES):
            for ci, tag in enumerate(tags):
                for ki, mask_type in enumerate(MASK_TYPES):
                    selected = circuit_parent._half_batch_mask(
                        masks[tag][mask_type], start, stop, half_start, half_stop).to(device)
                    positions = selected.nonzero(as_tuple=False)
                    counts[hi, ki, ci] += len(positions)
                    active.extend((hi, ki, ci, int(row), int(query)) for row, query in positions)
        for si, source in enumerate(SOURCES):
            with torch.enable_grad():
                logits, holder, _audit, error = _forward_fitted(
                    model, tokens, source_parent.SOURCE_ARMS[source], scale,
                    block_state["interface"], block_state["all_payload"], block)
                forwards += 1
                reconstruction = max(reconstruction, error)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), batch_rows[:, 1:].to(device).reshape(-1),
                    reduction="none").view(len(batch_rows), -1)
                for ai, (hi, ki, ci, row, query) in enumerate(active):
                    gradient, = torch.autograd.grad(
                        nll[row, query], holder["leaf"], retain_graph=ai + 1 < len(active))
                    backwards += 1
                    operators, triplet, direct = _target_contributions(
                        holder, gradient, row, query, keep_triplets)
                    for mi, operator in enumerate(operators):
                        operator_sums[mi][hi, si, ki, ci] += operator.double().cpu()
                    if keep_triplets:
                        triplet_sums[hi, si, ki, ci] += triplet.double().cpu()
                        mismatch = triplet.sum() - direct
                        identity_num += float(mismatch.square())
                        identity_den += float(direct.square())
                        varying = triplet[1:].sum()
                        trace = torch.trace(operators[0])
                        trace_num += float((varying - trace).square())
                        trace_den += float(varying.square())
                del logits, holder, nll
    return {
        "operator_sums": operator_sums, "triplet_sums": triplet_sums,
        "counts": counts, "forwards": forwards, "backwards": backwards,
        "reconstruction": reconstruction, "native_replay": replay,
        "triplet_identity_relative_squared": identity_num / max(identity_den, 1e-30),
        "operator_trace_identity_relative_squared": trace_num / max(trace_den, 1e-30),
    }


def _means(collection, mode):
    sums = collection["operator_sums"][mode]
    counts = collection["counts"][:, None, :, :, None, None].clamp_min(1)
    means = sums / counts
    return means, means[:, :, 0] - means[:, :, 1]


def _projector(operators):
    gram = torch.einsum("cab,cdb->ad", operators.double(), operators.double())
    values, vectors = torch.linalg.eigh((gram + gram.T) / 2)
    vector = vectors[:, -1]
    return torch.outer(vector, vector), values


def _profile(operators, projector):
    value = torch.einsum("ab,cba->c", projector.double(), operators.double())
    return value - value.mean()


def _mode_report(main, refit, mode, tags, activation_vector):
    main_means, contrast = _means(main, mode)
    _ref_means, ref_contrast = _means(refit, mode)
    projector, spectrum = _projector(contrast[0, 0])
    ref_projector, _ = _projector(ref_contrast[0, 0])
    wrong_map = torch.roll(torch.eye(MODE_DIMS[mode], dtype=torch.float64), 1, dims=0)
    wrong_ref_projector = wrong_map.T @ ref_projector @ wrong_map
    half_projectors = [_projector(contrast[hi, 0])[0] for hi in range(2)]
    activation_projector = torch.outer(activation_vector, activation_vector)
    profiles, complements = {}, {}
    eye = torch.eye(MODE_DIMS[mode], dtype=torch.float64)
    for hi in range(2):
        for si, source in enumerate(SOURCES):
            key = f"half{hi}_{source}"
            profiles[key] = _profile(contrast[hi, si], projector)
            complements[key] = _profile(contrast[hi, si], eye - projector)
    cross = []
    for source in SOURCES:
        cross.append(_cosine(profiles[f"half0_{source}"], profiles[f"half1_{source}"]))
    for hi in range(2):
        cross.append(_cosine(profiles[f"half{hi}_{SOURCES[0]}"],
                              profiles[f"half{hi}_{SOURCES[1]}" ]))
    member_control = []
    for hi in range(2):
        for si in range(len(SOURCES)):
            member = torch.einsum("ab,cba->c", projector, main_means[hi, si, 0])
            control = torch.einsum("ab,cba->c", projector, main_means[hi, si, 1])
            member_control.append(float(torch.linalg.vector_norm(member) /
                                        torch.linalg.vector_norm(control).clamp_min(1e-30)))
    controls = []
    for seed in CONTROL_SEEDS:
        generator = torch.Generator().manual_seed(seed + mode * 100)
        local = []
        base = profiles[f"half0_{SOURCES[0]}"]
        for key, value in profiles.items():
            if key == f"half0_{SOURCES[0]}":
                continue
            permutation = torch.randperm(len(tags), generator=generator)
            local.append(_cosine(base, value[permutation]))
        controls.append(min(local))
    complement_cosines = [_cosine(profiles[key], complements[key]) for key in profiles]
    opposite = []
    for hi in range(2):
        slab = profiles[f"half{hi}_{SOURCES[0]}"]
        comp = complements[f"half{hi}_{SOURCES[0]}"]
        opposite.append(int((slab * comp < 0).sum()))
    eigengap = float(spectrum[-1] / spectrum[-2].clamp_min(1e-30))
    stability = [
        _projector_overlap(projector, ref_projector),
        *[_projector_overlap(projector, value) for value in half_projectors],
    ]
    activation_overlap = _projector_overlap(projector, activation_projector)
    return {
        "mode": MODE_NAMES[mode], "projector": projector, "profiles": profiles,
        "complement_profiles": complements, "spectrum": spectrum.tolist(),
        "leading_over_second": eigengap, "refit_and_half_overlaps": stability,
        "minimum_refit_and_half_overlap": min(stability),
        "wrong_refit_map_overlap": _projector_overlap(projector, wrong_ref_projector),
        "activation_only_overlap": activation_overlap,
        "minimum_cross_view_profile_cosine": min(cross),
        "cross_view_profile_cosines": cross,
        "member_control_norm_ratios": member_control,
        "control_minimum_cosines": controls,
        "control_95pct": float(torch.quantile(torch.tensor(controls), .95,
                                               interpolation="higher")),
        "slab_complement_cosines": complement_cosines,
        "opposite_sign_counts_native_by_half": opposite,
    }


def _activation_vectors(fit_edges, all_payload, block):
    score1 = fit_edges["score1"]
    score2 = fit_edges["score2"]
    payload = all_payload[fit_edges["source"]]
    output = []
    for value, mean_key, basis_key in (
        (score1, "mean1", "basis1"), (score2, "mean2", "basis2"),
        (payload.reshape(len(payload), -1), "meanv", "basisv")):
        coordinates = (value.float() - block[mean_key].float()) @ block[basis_key].float()
        covariance = coordinates.T @ coordinates / len(coordinates)
        _values, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
        output.append(vectors[:, -1].double().cpu())
    return output


def _rotation_identity():
    generator = torch.Generator().manual_seed(480)
    worst = 0.0
    for dim in MODE_DIMS:
        z = torch.randn(31, dim, generator=generator, dtype=torch.float64)
        dz = torch.randn(31, dim, generator=generator, dtype=torch.float64)
        q = torch.linalg.qr(torch.randn(dim, dim, generator=generator,
                                       dtype=torch.float64)).Q
        raw = z.T @ dz
        a = (raw + raw.T) / 2
        zr, dzr = z @ q, dz @ q
        rawr = zr.T @ dzr
        ar = (rawr + rawr.T) / 2
        expected = q.T @ a @ q
        worst = max(worst, float((ar - expected).square().sum() /
                                 expected.square().sum().clamp_min(1e-30)))
    return worst


def _restricted_mode_score(collection, mode, keep):
    _means_local, contrast = _means(collection, mode)
    projector, _ = _projector(contrast[0, 0, keep])
    profiles = []
    for hi in range(2):
        for si in range(len(SOURCES)):
            profiles.append(_profile(contrast[hi, si, keep], projector))
    base = profiles[0]
    return min(_cosine(base, value) for value in profiles[1:])


def _public_report(report):
    return {key: value for key, value in report.items()
            if key not in ("projector", "profiles", "complement_profiles")}


def main():
    started = time.time()
    rows, _positive, masks, scale, tags, validation_tags, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 480, "model_loaded": False,
            "attention_response_outcomes_opened": False,
            "validation_family_outcomes_opened": False, "sealed_opened": False,
            "varying_ranks": MODE_DIMS, "affine_triplets": math.prod(AUG_DIMS),
            "response_operator_dimensions": MODE_DIMS,
            "expected_backwards_two_fits": 2 * metadata["expected_backwards_per_fit"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung480 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block_state = rebuild_block(model)
    main = collect_response(
        model, rows, masks, scale, tags, block_state, block_state["main"],
        keep_triplets=True)
    refit = collect_response(
        model, rows, masks, scale, tags, block_state, block_state["refit"],
        keep_triplets=False)
    activation = _activation_vectors(
        block_state["fit_edges"], block_state["all_payload"], block_state["main"])
    reports = [_mode_report(main, refit, mode, tags, activation[mode])
               for mode in range(3)]
    reports.sort(key=lambda row: (-row["minimum_cross_view_profile_cosine"], row["mode"]))
    winner = reports[0]
    passing_b = [row for row in reports if
                 row["leading_over_second"] >= 1.50
                 and row["minimum_refit_and_half_overlap"] >= .80
                 and row["minimum_refit_and_half_overlap"] >= row["activation_only_overlap"] + .10]
    roots = sorted(set(int(tag.split(".")[1]) for tag in tags))
    leave_one = []
    for root in roots:
        keep = torch.tensor([int(tag.split(".")[1]) != root for tag in tags])
        candidates = [(float(_restricted_mode_score(main, mode, keep)), mode)
                      for mode in range(3)]
        minimum, local_mode = max(candidates, key=lambda row: (row[0], -row[1]))
        leave_one.append({"omitted_root": root, "minimum_cross_view_cosine": minimum,
                          "winning_mode": MODE_NAMES[local_mode],
                          "same_winning_mode": MODE_NAMES[local_mode] == winner["mode"]})
    exact_r424 = json.loads(R424_RESULT.read_text())
    exact_r425 = json.loads(R425_RESULT.read_text())
    select_ref = exact_r424["select_edge_metrics"]["joint"]["summed_relative_mse"]
    fresh_ref = exact_r425["fresh_edge_metrics"]["joint"]["summed_relative_mse"]
    rotation_error = _rotation_identity()
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and block_state["selected_index"] == 0 and block_state["payload_error"] <= 1e-10
        and abs(block_state["select_metric"]["summed_relative_mse"] - select_ref)
            / select_ref <= 5e-3
        and abs(block_state["fresh_metric"]["summed_relative_mse"] - fresh_ref)
            / fresh_ref <= 5e-3
        and abs(block_state["select_routed"]
                - exact_r424["select_transport"]["routed_u16_r2"]["joint"]) <= 5e-3
        and abs(block_state["fresh_routed"]
                - exact_r425["fresh_transport"]["routed_u16_r2"]["joint"]) <= 5e-3
        and main["triplet_identity_relative_squared"] <= 1e-8
        and main["operator_trace_identity_relative_squared"] <= 1e-8
        and rotation_error <= 1e-8
        and main["native_replay"]["relative_squared"] <= 1e-12
        and main["reconstruction"] <= 1e-10 and refit["reconstruction"] <= 1e-10
        and torch.equal(main["counts"], refit["counts"])
        and int(main["counts"][:, 0].min()) >= 39
        and int(main["counts"][:, 1].min()) >= 439
        and main["backwards"] == metadata["expected_backwards_per_fit"]
        and refit["backwards"] == metadata["expected_backwards_per_fit"]
        and main["forwards"] == metadata["expected_fit_forwards"] + 2 * math.ceil(
            DISCOVERY_STOP / BATCH)
        and refit["forwards"] == metadata["expected_fit_forwards"]
        and len(validation_tags) == 30)
    pred_b = len(passing_b) >= 2
    pred_c = bool(
        winner["minimum_cross_view_profile_cosine"] >= .70
        and winner["minimum_cross_view_profile_cosine"] >= winner["control_95pct"] + .15
        and min(winner["member_control_norm_ratios"]) >= 1.5)
    pred_d = sum(row["minimum_cross_view_cosine"] >= .60 and row["same_winning_mode"]
                 for row in leave_one) >= 5
    pred_e = bool(
        max(abs(value) for value in winner["slab_complement_cosines"]) <= .70
        and min(winner["opposite_sign_counts_native_by_half"]) >= 10)
    strong_null = bool(not pred_a or not pred_b or not pred_c)
    torch.save({
        "schema": "rung480_affine_triplets_and_gauge_covariant_response_v1",
        "main_operator_sums": main["operator_sums"],
        "refit_operator_sums_aligned": refit["operator_sums"],
        "response_counts": main["counts"], "main_affine_triplet_response_sums": main["triplet_sums"],
        "projectors": {row["mode"]: row["projector"] for row in reports},
        "profiles": {row["mode"]: row["profiles"] for row in reports},
        "complement_profiles": {row["mode"]: row["complement_profiles"] for row in reports},
        "varying_ranks": MODE_DIMS, "augmented_dimensions": AUG_DIMS,
        "discovery_tags": tags, "validation_tags_or_responses_included": False,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 480,
        "claim_level": "discovery_only_gauge_covariant_downstream_projector_screen",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "block_rebuild": {
            "selected_restart": block_state["selected_index"],
            "varying_ranks": MODE_DIMS, "augmented_dimensions": AUG_DIMS,
            "affine_triplet_count": math.prod(AUG_DIMS),
            "payload_fold_max_abs": block_state["payload_error"],
            "fit_half_projector_overlap": block_state["half_overlap"],
            "select_joint_summed_relative_mse": block_state["select_metric"]["summed_relative_mse"],
            "fresh_joint_summed_relative_mse": block_state["fresh_metric"]["summed_relative_mse"],
            "select_joint_routed_u16_r2": block_state["select_routed"],
            "fresh_joint_routed_u16_r2": block_state["fresh_routed"],
        },
        "instrument": {
            "triplet_sum_relative_squared": main["triplet_identity_relative_squared"],
            "operator_trace_relative_squared": main["operator_trace_identity_relative_squared"],
            "synthetic_rotation_conjugation_relative_squared": rotation_error,
            "factor_reconstruction_relative_squared_max": max(
                main["reconstruction"], refit["reconstruction"]),
            "native_replay": main["native_replay"],
            "member_support_min": int(main["counts"][:, 0].min()),
            "control_support_min": int(main["counts"][:, 1].min()),
        },
        "mode_reports": [_public_report(row) for row in reports],
        "passing_b_modes": [row["mode"] for row in passing_b],
        "winning_slab": winner["mode"], "leave_one_family": leave_one,
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_downstream_projectors': pred_b,
        'pred_c_stable_circuit_labelled_slab': pred_c,
        'pred_d_leave_family_stable': pred_d,
        'pred_e_distinct_downstream_uses': pred_e,
        "strong_null": strong_null,
        "validation_family_outcomes_opened": False,
        "sealed_attention0_consequences_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        "execution_price": {
            "response_forwards": main["forwards"] + refit["forwards"],
            "response_backwards": main["backwards"] + refit["backwards"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "runtime_s": time.time() - started,
        "next_step": ("exact_odd_family_attention0_slab_removal"
                      if all((pred_a, pred_b, pred_c, pred_d, pred_e))
                      else "mlp0_token_token_context_context_functional_decomposition"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": 480,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "winning_slab": winner["mode"],
        "passing_b_modes": result["passing_b_modes"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
