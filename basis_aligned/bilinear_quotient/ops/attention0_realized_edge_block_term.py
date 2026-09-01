"""RUNG 422 -- REALIZED CONTINUOUS ATTENTION0 QK1 x QK2 x OV BLOCK TERM.

On exact natural edges, compare equal-parameter marginal PCA projectors with
the same score-half6/score-half6/payload32 bases jointly optimized for the
complete downstream-metric edge product.  A fixed head derangement preserves
mode spectra while breaking the registered contraction.  This is a realized
identification screen: native generators remain, so no compression is claimed.
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
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_realized_edge_block_term_results.json"
BASE = OPS / "attention0_ov_downstream_codebook.py"
PARENT = BQ / "attention0_ov_downstream_codebook_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
D = 1152
N_HEAD = 9
HD = 128
U_RANK = 16
SCORE_RANK = 6
PAYLOAD_RANK = 32
PAYLOAD_DIM = N_HEAD * U_RANK
POSITIONS = tuple(range(16, 241, 16))
DOC_BATCH = 4
TRAIN_STEPS = 512
TRAIN_BATCH = 4096
TRAIN_LR = .02
SCREEN_VALUES = D * U_RANK + 2 * (N_HEAD + N_HEAD * SCORE_RANK) + (
    PAYLOAD_DIM + PAYLOAD_DIM * PAYLOAD_RANK)
DERANGEMENT = tuple((head + 4) % N_HEAD for head in range(N_HEAD))
ARMS = ("marginal", "joint", "deranged")
CONSUMERS = ("mlp0", "q1", "k1", "q2", "k2", "fresh_v")


def _score_halves(block, state, rope_tables, apply_rot):
    batch, length, _ = state.shape
    cos, sin = rope_tables(length, HD, state.device, torch.float32, "bf16")
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def qk(layer):
        value = F.rms_norm(layer(state).view(batch, length, N_HEAD, HD), (HD,))
        return apply_rot(value, cos, sin)

    q1, k1 = qk(block.attn.c_q), qk(block.attn.c_k)
    q2, k2 = qk(block.attn.c_q2), qk(block.attn.c_k2)
    score1 = torch.einsum("bqhd,bkhd->bhqk", q1, k1) / HD
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HD
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    return score1.masked_fill(~mask, 0), score2.masked_fill(~mask, 0)


def _collect_edges(model, rows, rope_tables, apply_rot):
    block0 = model.transformer.h[0]
    score1_parts, score2_parts, source_parts, document_parts = [], [], [], []
    for start in range(0, len(rows), DOC_BATCH):
        tokens = rows[start:start + DOC_BATCH, :-1].to("cuda")
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        state = F.rms_norm(token_base, (D,))
        score1, score2 = _score_halves(block0, state, rope_tables, apply_rot)
        for local in range(len(tokens)):
            for query in POSITIONS:
                count = query + 1
                score1_parts.append(score1[local, :, query, :count].T)
                score2_parts.append(score2[local, :, query, :count].T)
                source_parts.append(tokens[local, :count])
                document_parts.append(torch.full(
                    (count,), start + local, device=tokens.device, dtype=torch.long))
    return {
        "score1": torch.cat(score1_parts).float(),
        "score2": torch.cat(score2_parts).float(),
        "source": torch.cat(source_parts),
        "document": torch.cat(document_parts),
    }


def _pca(value, rank):
    mean = value.mean(0, keepdim=True)
    centered = value - mean
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    basis = eigenvectors[:, -rank:]
    return mean, basis, eigenvalues


def _project(value, mean, basis):
    return mean + ((value - mean) @ basis) @ basis.T


def _edge_values(score1, score2, payload):
    return score1[:, :, None] * score2[:, :, None] * payload


def _metric_square(value, metric_factor):
    return (value @ metric_factor.T).square().sum()


def _projector_overlap(left, right):
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    return float(singular.square().mean())


def _orth_error(basis):
    identity = torch.eye(basis.shape[1], device=basis.device)
    return float((basis.T @ basis - identity).abs().max())


def _fit_initial(edges, all_payload, mask=None):
    if mask is None:
        score1 = edges["score1"]
        score2 = edges["score2"]
        source = edges["source"]
    else:
        score1 = edges["score1"][mask]
        score2 = edges["score2"][mask]
        source = edges["source"][mask]
    payload = all_payload[source].reshape(-1, PAYLOAD_DIM)
    mean1, basis1, spectrum1 = _pca(score1, SCORE_RANK)
    mean2, basis2, spectrum2 = _pca(score2, SCORE_RANK)
    meanv, basisv, spectrumv = _pca(payload, PAYLOAD_RANK)
    return {
        "mean1": mean1, "basis1": basis1,
        "mean2": mean2, "basis2": basis2,
        "meanv": meanv, "basisv": basisv,
        "spectrum1": spectrum1, "spectrum2": spectrum2, "spectrumv": spectrumv,
    }


def _denominators(edges, all_payload, metric_factor, mask=None):
    indices = torch.arange(len(edges["source"]), device="cuda") if mask is None else mask.nonzero().flatten()
    individual = 0.0
    summed = 0.0
    for start in range(0, len(indices), 16_384):
        chosen = indices[start:start + 16_384]
        payload = all_payload[edges["source"][chosen]]
        target = _edge_values(edges["score1"][chosen], edges["score2"][chosen], payload)
        individual += float(_metric_square(target, metric_factor))
        summed += float(_metric_square(target.sum(1), metric_factor))
    return individual, summed


def _objective(indices, edges, all_payload, params, metric_factor, denominators):
    basis1 = torch.linalg.qr(params["raw1"], mode="reduced").Q
    basis2 = torch.linalg.qr(params["raw2"], mode="reduced").Q
    basisv = torch.linalg.qr(params["rawv"], mode="reduced").Q
    score1 = edges["score1"][indices]
    score2 = edges["score2"][indices]
    payload = all_payload[edges["source"][indices]]
    rec1 = _project(score1, params["mean1"], basis1)
    rec2 = _project(score2, params["mean2"], basis2)
    flat = payload.reshape(-1, PAYLOAD_DIM)
    recv = _project(flat, params["meanv"], basisv).reshape(-1, N_HEAD, U_RANK)
    target = _edge_values(score1, score2, payload)
    predicted = _edge_values(rec1, rec2, recv)
    individual = _metric_square(target - predicted, metric_factor)
    summed = _metric_square(target.sum(1) - predicted.sum(1), metric_factor)
    batch_fraction = len(indices) / denominators[2]
    individual_den = denominators[0] * batch_fraction
    summed_den = denominators[1] * batch_fraction
    loss = .5 * individual / max(individual_den, 1e-30) + .5 * summed / max(summed_den, 1e-30)
    return loss, (basis1, basis2, basisv)


def _optimize(edges, all_payload, initial, metric_factor, seed, mask=None):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    eligible = torch.arange(len(edges["source"]), device="cuda") if mask is None else mask.nonzero().flatten()
    den_individual, den_summed = _denominators(edges, all_payload, metric_factor, mask)
    params = {
        "raw1": torch.nn.Parameter(initial["basis1"] + .005 * torch.randn(
            initial["basis1"].shape, generator=generator, device="cpu").to("cuda")),
        "raw2": torch.nn.Parameter(initial["basis2"] + .005 * torch.randn(
            initial["basis2"].shape, generator=generator, device="cpu").to("cuda")),
        "rawv": torch.nn.Parameter(initial["basisv"] + .005 * torch.randn(
            initial["basisv"].shape, generator=generator, device="cpu").to("cuda")),
        "mean1": initial["mean1"], "mean2": initial["mean2"], "meanv": initial["meanv"],
    }
    optimizer = torch.optim.Adam([params["raw1"], params["raw2"], params["rawv"]], lr=TRAIN_LR)
    gradient_max = {"score1": 0.0, "score2": 0.0, "payload": 0.0}
    for step in range(TRAIN_STEPS):
        pick = torch.randint(
            len(eligible), (min(TRAIN_BATCH, len(eligible)),), generator=generator,
            device="cpu").to("cuda")
        indices = eligible[pick]
        loss, _bases = _objective(
            indices, edges, all_payload, params, metric_factor,
            (den_individual, den_summed, len(eligible)))
        optimizer.zero_grad()
        loss.backward()
        for name, parameter in (("score1", params["raw1"]),
                                ("score2", params["raw2"]),
                                ("payload", params["rawv"])):
            gradient_max[name] = max(gradient_max[name], float(parameter.grad.abs().max()))
        optimizer.step()
    with torch.no_grad():
        bases = tuple(torch.linalg.qr(params[name], mode="reduced").Q
                      for name in ("raw1", "raw2", "rawv"))
        # Exact full-FIT objective for restart selection.
        total_individual = total_summed = 0.0
        for start in range(0, len(eligible), 16_384):
            chosen = eligible[start:start + 16_384]
            score1 = edges["score1"][chosen]
            score2 = edges["score2"][chosen]
            payload = all_payload[edges["source"][chosen]]
            rec1 = _project(score1, params["mean1"], bases[0])
            rec2 = _project(score2, params["mean2"], bases[1])
            recv = _project(payload.reshape(-1, PAYLOAD_DIM), params["meanv"], bases[2]).reshape(
                -1, N_HEAD, U_RANK)
            target = _edge_values(score1, score2, payload)
            predicted = _edge_values(rec1, rec2, recv)
            total_individual += float(_metric_square(target - predicted, metric_factor))
            total_summed += float(_metric_square(
                target.sum(1) - predicted.sum(1), metric_factor))
        exact_loss = .5 * total_individual / den_individual + .5 * total_summed / den_summed
    return {
        "mean1": params["mean1"], "basis1": bases[0],
        "mean2": params["mean2"], "basis2": bases[1],
        "meanv": params["meanv"], "basisv": bases[2],
        "fit_objective": exact_loss,
        "gradient_max": gradient_max,
        "movement_from_pca": {
            "score1": 1 - _projector_overlap(initial["basis1"], bases[0]),
            "score2": 1 - _projector_overlap(initial["basis2"], bases[1]),
            "payload": 1 - _projector_overlap(initial["basisv"], bases[2]),
        },
    }


def _edge_metrics(edges, all_payload, model, metric_factor, arm):
    individual_num = summed_num = individual_den = summed_den = 0.0
    derangement = torch.tensor(DERANGEMENT, device="cuda")
    for start in range(0, len(edges["source"]), 16_384):
        sl = slice(start, start + 16_384)
        score1 = edges["score1"][sl]
        score2 = edges["score2"][sl]
        payload = all_payload[edges["source"][sl]]
        rec1 = _project(score1, model["mean1"], model["basis1"])
        rec2 = _project(score2, model["mean2"], model["basis2"])
        recv = _project(payload.reshape(-1, PAYLOAD_DIM), model["meanv"], model["basisv"]).reshape(
            -1, N_HEAD, U_RANK)
        if arm == "deranged":
            rec2 = rec2[:, derangement]
        target = _edge_values(score1, score2, payload)
        predicted = _edge_values(rec1, rec2, recv)
        individual_num += float(_metric_square(target - predicted, metric_factor))
        summed_num += float(_metric_square(target.sum(1) - predicted.sum(1), metric_factor))
        individual_den += float(_metric_square(target, metric_factor))
        summed_den += float(_metric_square(target.sum(1), metric_factor))
    return {
        "individual_relative_mse": individual_num / individual_den,
        "summed_relative_mse": summed_num / summed_den,
        "summed_r2_zero_origin": 1 - summed_num / summed_den,
    }


def _reconstruct_u(score1, score2, tokens, all_payload, model, arm):
    # scores arrive [batch,head,query,key]; projection acts on the head coordinate.
    shape = score1.shape
    flat1 = score1.permute(0, 2, 3, 1).reshape(-1, N_HEAD)
    flat2 = score2.permute(0, 2, 3, 1).reshape(-1, N_HEAD)
    rec1 = _project(flat1, model["mean1"], model["basis1"]).reshape(
        shape[0], shape[2], shape[3], N_HEAD).permute(0, 3, 1, 2)
    rec2 = _project(flat2, model["mean2"], model["basis2"]).reshape(
        shape[0], shape[2], shape[3], N_HEAD).permute(0, 3, 1, 2)
    payload = all_payload[tokens]
    recv = _project(payload.reshape(-1, PAYLOAD_DIM), model["meanv"], model["basisv"]).reshape(
        *payload.shape)
    if arm == "deranged":
        order = torch.tensor(DERANGEMENT, device=score1.device)
        rec2 = rec2[:, order]
    return torch.einsum("bhqk,bkhc->bqc", rec1 * rec2, recv)


def _suffix_logits(model, tokens, x0, token_base, attention0, first_value):
    x = token_base + attention0
    block0 = model.transformer.h[0]
    x = x + block0.mlp(F.rms_norm(x, (D,)))
    for block in model.transformer.h[1:]:
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first_value = block.attn(F.rms_norm(x, (D,)), first_value)
        x = x + attention
        x = x + block.mlp(F.rms_norm(x, (D,)))
    return 30 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30)


def _document_transport(model, rows, interface, all_payload, models, rope_tables, apply_rot, base, scoring):
    routed_sse = {arm: 0.0 for arm in ARMS}
    consumer_sse = {arm: {name: 0.0 for name in CONSUMERS} for arm in ARMS}
    target_routed = 0.0
    target_consumer = {name: 0.0 for name in CONSUMERS}
    ce = {"native": [], **{arm: [] for arm in ARMS}}
    replay_before_max = replay_after_max = replay_num = replay_den = 0.0
    block0, block1 = model.transformer.h[:2]
    for start in range(0, len(rows), DOC_BATCH):
        batch = rows[start:start + DOC_BATCH].to("cuda")
        tokens = batch[:, :-1]
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        state = F.rms_norm(token_base, (D,))
        native_attention, first_value = block0.attn(state, None)
        native_u = native_attention.float() @ interface.float()
        score1, score2 = _score_halves(block0, state, rope_tables, apply_rot)
        edge_u = torch.einsum(
            "bhqk,bkhc->bqc", score1 * score2, all_payload[tokens].to(score1.dtype)).float()
        remainder = native_u - edge_u
        replay_before_max = max(replay_before_max, float(remainder.abs().max()))
        replay_after_max = max(replay_after_max, float((native_u - (edge_u + remainder)).abs().max()))
        replay_num += float(remainder.double().square().sum())
        replay_den += float(native_u.double().square().sum())

        native_fields = base._consumer_fields(block0, block1, x0, token_base, native_attention)
        tail_attention = native_attention - (
            native_u @ interface.float().T).to(native_attention.dtype)
        tail_fields = base._consumer_fields(block0, block1, x0, token_base, tail_attention)
        target_routed += float(native_u[:, POSITIONS].double().square().sum())
        for name in CONSUMERS:
            response = (native_fields[name].float().flatten(2)[:, POSITIONS]
                        - tail_fields[name].float().flatten(2)[:, POSITIONS])
            target_consumer[name] += float(response.double().square().sum())

        native_logits = _suffix_logits(
            model, tokens, x0, token_base, native_attention, first_value)
        for row in range(len(batch)):
            ce["native"].append(scoring.document_mean_ce(native_logits[row], batch[row, 1:]))
        for arm in ARMS:
            approx_u = _reconstruct_u(score1, score2, tokens, all_payload, models[arm], arm)
            changed = native_attention + ((approx_u.float() - edge_u) @ interface.float().T).to(
                native_attention.dtype)
            changed_u = changed.float() @ interface.float()
            routed_sse[arm] += float(
                (changed_u[:, POSITIONS].double() - native_u[:, POSITIONS].double()).square().sum())
            changed_fields = base._consumer_fields(block0, block1, x0, token_base, changed)
            for name in CONSUMERS:
                consumer_sse[arm][name] += float(
                    (changed_fields[name].float().flatten(2)[:, POSITIONS].double()
                     - native_fields[name].float().flatten(2)[:, POSITIONS].double()).square().sum())
            logits = _suffix_logits(model, tokens, x0, token_base, changed, first_value)
            for row in range(len(batch)):
                ce[arm].append(scoring.document_mean_ce(logits[row], batch[row, 1:]))

    ce_tensors = {name: torch.stack(values).double().cpu() for name, values in ce.items()}
    ce_public = {}
    for name, values in ce_tensors.items():
        ce_public[name] = {
            "mean": float(values.mean()),
            "damage": float(values.mean() - ce_tensors["native"].mean()),
            "wave_damage": [
                float(values[:48].mean() - ce_tensors["native"][:48].mean()),
                float(values[48:].mean() - ce_tensors["native"][48:].mean()),
            ],
        }
    routed_r2 = {arm: 1 - routed_sse[arm] / target_routed for arm in ARMS}
    consumer_r2 = {
        arm: {name: 1 - consumer_sse[arm][name] / target_consumer[name]
              for name in CONSUMERS} for arm in ARMS}
    return {
        "routed_u16_r2": routed_r2,
        "consumer_r2": consumer_r2,
        "mean_consumer_r2": {
            arm: sum(values.values()) / len(values) for arm, values in consumer_r2.items()},
        "ce": ce_public,
        "edge_replay_before_remainder_max_abs": replay_before_max,
        "edge_replay_after_remainder_max_abs": replay_after_max,
        "edge_replay_relative_squared_before_remainder": replay_num / replay_den,
        "edge_replay_relative_squared_after_remainder": 0.0,
    }


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert D == N_HEAD * HD and U_RANK == 16
        assert SCORE_RANK == 6 and PAYLOAD_RANK == 32
        assert SCREEN_VALUES == 23_310 and len(POSITIONS) == 15
        assert sorted(DERANGEMENT) == list(range(N_HEAD))
        assert all(DERANGEMENT[index] != index for index in range(N_HEAD))
        print("ATTENTION0 REALIZED EDGE BLOCK TERM | dry run: r6/r6/r32, equal-price controls")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(QK))
    spec = importlib.util.spec_from_file_location("edge_base", BASE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring
    from tier2_model import rope_tables, apply_rot

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block0 = model.transformer.h[0]

    # Rebuild the rung419 task interface and all frozen calibration objects.
    with torch.no_grad():
        captured = base._capture_cproj_input(model, fit_rows, device).to(device)
        weight = block0.attn.c_proj.weight.detach().float()
        a_factor, b_factor = base._asvd(weight, captured)
        interface = torch.linalg.qr(a_factor[:, :U_RANK].float(), mode="reduced").Q.to(device)
        rank16_weight = (a_factor[:, :U_RANK] @ b_factor[:U_RANK]).to(device)
        full_weight = (a_factor @ b_factor).to(device)
        haar_interfaces = []
        for seed in base.HAAR_SEEDS:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            haar_interfaces.append(torch.linalg.qr(
                torch.randn(D, U_RANK, generator=generator), mode="reduced").Q.to(device))
        calibration_ce = {"native": base._ce(model, select_rows, device, scoring)}
        calibration_ce["full"] = base._ce(
            model, select_rows, device, scoring, cproj_weight=full_weight)
        calibration_ce["rank16"] = base._ce(
            model, select_rows, device, scoring, cproj_weight=rank16_weight)
        for index, haar in enumerate(haar_interfaces):
            haar_weight = haar @ haar.T @ weight
            calibration_ce[f"haar_{index}"] = base._ce(
                model, select_rows, device, scoring, cproj_weight=haar_weight)
        calibration = {
            name: float(values.mean() - calibration_ce["native"].mean())
            for name, values in calibration_ce.items() if name != "native"}
        calibration["native_ce"] = float(calibration_ce["native"].mean())
        calibration["u16_orthogonality_max_abs"] = _orth_error(interface)
        calibration["local_forward_replay_max_abs"] = base._native_forward_replay(
            model, select_rows, facade, device)
        embedding = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))[:base.VOCAB]
        payload_fold_error = base._payload_exactness(model, embedding)
        all_payload = base._payload_codes(model, interface, embedding)
        write_samples = []
        for start in range(0, len(fit_rows), DOC_BATCH):
            tokens = fit_rows[start:start + DOC_BATCH, :-1].to(device)
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
            attention0, _ = block0.attn(F.rms_norm(token_base, (D,)), None)
            write_samples.append(attention0.float()[:, POSITIONS].reshape(-1, D))
        write_samples = torch.cat(write_samples)
        sigma = torch.sqrt((write_samples @ interface).double().square().mean(0)).float()
        fit_gram, normalizers, fit_live = base._response_metric(
            model, fit_rows, interface, sigma, None, device)
        metric_factor, _metric_eigenvalues = base._metric_factor(fit_gram)
        parent = json.loads(PARENT.read_text())
        parent_gram = torch.tensor(parent["response_metrics"]["task"]["fit_gram"], device=device)
        gram_relative_error = float(
            (fit_gram - parent_gram).double().norm() / parent_gram.double().norm())
        fit_edges = _collect_edges(model, fit_rows, rope_tables, apply_rot)

    initial = _fit_initial(fit_edges, all_payload)
    marginal = {
        "mean1": initial["mean1"], "basis1": initial["basis1"],
        "mean2": initial["mean2"], "basis2": initial["basis2"],
        "meanv": initial["meanv"], "basisv": initial["basisv"],
    }
    restarts = [
        _optimize(fit_edges, all_payload, initial, metric_factor, seed)
        for seed in (422, 423)]
    joint = min(restarts, key=lambda value: value["fit_objective"])
    # The deranged arm has identical parameters; only its contraction changes.
    models = {"marginal": marginal, "joint": joint, "deranged": joint}

    half_models = []
    for low, high, seed in ((0, 48, 422_048), (48, 96, 422_096)):
        mask = (fit_edges["document"] >= low) & (fit_edges["document"] < high)
        half_initial = _fit_initial(fit_edges, all_payload, mask)
        half_models.append(_optimize(
            fit_edges, all_payload, half_initial, metric_factor, seed, mask))
    half_overlap = {
        "score1": _projector_overlap(half_models[0]["basis1"], half_models[1]["basis1"]),
        "score2": _projector_overlap(half_models[0]["basis2"], half_models[1]["basis2"]),
        "payload": _projector_overlap(half_models[0]["basisv"], half_models[1]["basisv"]),
    }

    with torch.no_grad():
        select_edges = _collect_edges(model, select_rows, rope_tables, apply_rot)
        edge_metrics = {
            arm: _edge_metrics(select_edges, all_payload, models[arm], metric_factor, arm)
            for arm in ARMS}
        transport = _document_transport(
            model, select_rows, interface, all_payload, models,
            rope_tables, apply_rot, base, scoring)

    marginal_summed = edge_metrics["marginal"]["summed_relative_mse"]
    joint_summed = edge_metrics["joint"]["summed_relative_mse"]
    deranged_summed = edge_metrics["deranged"]["summed_relative_mse"]
    joint_gain = (marginal_summed - joint_summed) / marginal_summed
    deranged_gain = (deranged_summed - joint_summed) / deranged_summed
    routed = transport["routed_u16_r2"]
    consumer = transport["mean_consumer_r2"]
    ce = transport["ce"]
    gradient_live = all(value > 0 for restart in restarts for value in restart["gradient_max"].values())
    movement_live = all(value > 1e-8 for value in joint["movement_from_pca"].values())

    pred_a = (
        calibration["u16_orthogonality_max_abs"] <= 2e-5
        and abs(calibration["full"]) < 1e-3
        and calibration["rank16"] <= .12
        and all(calibration[f"haar_{index}"] >= .80 for index in range(3))
        and calibration["local_forward_replay_max_abs"] <= 2e-5
        and payload_fold_error <= 1e-10
        and gram_relative_error <= 1e-6
        and transport["edge_replay_after_remainder_max_abs"] <= 2e-5
        and transport["edge_replay_relative_squared_after_remainder"] <= 1e-12
        and all(_orth_error(model_arm[key]) <= 2e-5
                for model_arm in models.values()
                for key in ("basis1", "basis2", "basisv"))
        and SCREEN_VALUES == 23_310
        and len(fit_edges["source"]) == len(select_edges["source"])
        and not torch.equal(fit_rows, select_rows))
    pred_b = (
        joint_summed <= .45
        and joint_gain >= .20
        and deranged_gain >= .20
        and edge_metrics["joint"]["individual_relative_mse"]
            <= edge_metrics["marginal"]["individual_relative_mse"] + .05)
    pred_c = (
        routed["joint"] >= .60
        and min(transport["consumer_r2"]["joint"].values()) >= .50
        and routed["joint"] - routed["marginal"] >= .05
        and consumer["joint"] - consumer["marginal"] >= .05
        and all(ce["joint"]["wave_damage"][wave]
                <= ce["marginal"]["wave_damage"][wave] + .005
                for wave in range(2)))
    pred_d = (
        half_overlap["score1"] >= .70
        and half_overlap["score2"] >= .70
        and half_overlap["payload"] >= .50
        and routed["joint"] - routed["deranged"] >= .10
        and consumer["joint"] - consumer["deranged"] >= .02
        and ce["deranged"]["damage"] - ce["joint"]["damage"] >= .005
        and all(ce["deranged"]["wave_damage"][wave]
                - ce["joint"]["wave_damage"][wave] >= .005 for wave in range(2))
        and gradient_live and movement_live)
    strong_null = (
        not pred_a
        or routed["joint"] <= .30
        or joint_gain < .02
        or abs(routed["joint"] - routed["deranged"]) <= .02
        or all(ce["joint"]["wave_damage"][wave]
               > ce["marginal"]["wave_damage"][wave] for wave in range(2)))

    public_restarts = [{
        "fit_objective": value["fit_objective"],
        "gradient_max": value["gradient_max"],
        "movement_from_pca": value["movement_from_pca"],
    } for value in restarts]
    result = {
        "status": "attention0_realized_edge_block_term_complete",
        "rung": 422,
        "claim_level": "realized_continuous_edge_identification_screen_not_compression_or_adoption",
        "definition": {
            "edge": "e[n,h,c]=score1[n,h]*score2[n,h]*U16_payload[n,h,c]",
            "marginal": "affine PCA ranks score1=6,score2=6,flattened_head_payload=32",
            "joint": "same shapes optimized for normalized individual plus head-summed response-metric edge error",
            "deranged": "joint reconstruction with score2 head h mapped to (h+4) mod 9 before product",
        },
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows), "FINAL_opened": 0},
        "edges": {"FIT": len(fit_edges["source"]), "SELECT": len(select_edges["source"]),
                  "query_positions": list(POSITIONS), "all_causal_sources": True},
        "instrument": {
            "calibration": calibration,
            "payload_fold_max_abs": payload_fold_error,
            "response_gram_parent_relative_error": gram_relative_error,
            "fit_consumer_live_max_abs": fit_live,
            "edge_replay_before_remainder_max_abs":
                transport["edge_replay_before_remainder_max_abs"],
            "edge_replay_after_remainder_max_abs":
                transport["edge_replay_after_remainder_max_abs"],
            "edge_replay_relative_squared_before_remainder":
                transport["edge_replay_relative_squared_before_remainder"],
            "edge_replay_relative_squared_after_remainder":
                transport["edge_replay_relative_squared_after_remainder"],
        },
        "ranks": {"score1": SCORE_RANK, "score2": SCORE_RANK, "payload": PAYLOAD_RANK},
        "literal_screen_values": SCREEN_VALUES,
        "net_model_saving_values": 0,
        "native_generators_retained": True,
        "fit_restarts": public_restarts,
        "selected_restart": min(range(len(restarts)), key=lambda index: restarts[index]["fit_objective"]),
        "fit_half_projector_overlap": half_overlap,
        "select_edge_metrics": edge_metrics,
        "select_joint_gain_over_marginal": joint_gain,
        "select_joint_gain_over_deranged": deranged_gain,
        "select_transport": {
            "routed_u16_r2": routed,
            "consumer_r2": transport["consumer_r2"],
            "mean_consumer_r2": consumer,
            "ce": ce,
        },
        'pred_a_instrument_valid': bool(pred_a),
        'pred_b_complete_edge_joint_block': bool(pred_b),
        'pred_c_downstream_transport': bool(pred_c),
        'pred_d_reproducible_specific_coupling': bool(pred_d),
        "strong_null_no_low_rank_realized_head_service_coupling": bool(strong_null),
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "next_step": (
            "physical_latent_score_payload_generator_factorization"
            if pred_a and pred_b and pred_c and pred_d and not strong_null
            else "geometric_edge_block_without_downstream_claim"
            if pred_a and pred_b and not pred_c and not strong_null
            else "predictive_projection_without_coupling_identification"
            if pred_a and pred_c and not pred_d and not strong_null
            else "mlp0_nonlinear_Q_or_attention1_copy_response"
            if pred_a else "instrument_repair_only"),
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "instrument": result["instrument"],
        "fit_half_overlap": half_overlap, "edge_metrics": edge_metrics,
        "joint_gain_over_marginal": joint_gain,
        "joint_gain_over_deranged": deranged_gain,
        "transport": result["select_transport"],
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c, "pred_d": pred_d,
        "strong_null": strong_null, "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ATTENTION0 REALIZED EDGE BLOCK TERM DONE", flush=True)


if __name__ == "__main__":
    main()
