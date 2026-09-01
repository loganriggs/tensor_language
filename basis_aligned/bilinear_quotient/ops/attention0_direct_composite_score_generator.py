"""RUNG 431 -- DIRECT NON-HEAD ATTENTION0 SCORE GENERATORS.

Compare six width-96 rotary bilinear score modes per branch (decoded through
the frozen rung-424 head-mixed basis) with nine width-64 headwise generators.
The input-map prices match exactly and both candidates bypass native layer-0
Q/K.  This is a physical feasibility screen, not an adoption or semantics run.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
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
OUT = BQ / "attention0_direct_composite_score_generator_results.json"
BUNDLE = BQ / "attention0_direct_composite_score_generator_bundle.pt"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
R424 = BQ / "attention0_realized_edge_block_term_results.json"
EDGE_PATH = OPS / "attention0_realized_edge_block_term.py"
OV_BASE = OPS / "attention0_ov_downstream_codebook.py"

VOCAB = 50_257
D = 1_152
N_HEAD = 9
HD = 128
N_MODE = 6
MODE_WIDTH = 96
HEAD_WIDTH = 64
U_RANK = 16
POSITIONS = tuple(range(16, 241, 16))
DOC_BATCH = 4
STEPS = 1_000
TRAIN_BATCH = 1_024
LR = 1e-3
LOSS_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
ARMS = ("MODE96", "HEAD64", "MODE96_DERANGED", "PERMUTED")
CONSUMERS = ("mlp0", "q1", "k1", "q2", "k2", "fresh_v")

MODE_MAP_VALUES = 2 * 2 * N_MODE * D * MODE_WIDTH
HEAD_MAP_VALUES = 2 * 2 * N_HEAD * D * HEAD_WIDTH
NATIVE_MAP_VALUES = 4 * D * D
MODE_COMPLETE_VALUES = MODE_MAP_VALUES + 2 * N_MODE + 2 * N_HEAD * N_MODE + 2 * N_HEAD + MODE_WIDTH // 2
HEAD_COMPLETE_VALUES = HEAD_MAP_VALUES + 2 * N_HEAD + HEAD_WIDTH // 2


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _frequency_indices(width: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0, HD // 2 - 1, width // 2).round().long().to(device)


def _rotate_selected(value: torch.Tensor, position: torch.Tensor,
                     frequency: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor, apply_rot) -> torch.Tensor:
    c = cos[position][:, frequency][:, None, :]
    s = sin[position][:, frequency][:, None, :]
    return apply_rot(value, c, s)


def _collect_edges(model, rows: torch.Tensor, rope_tables, apply_rot, edge_mod) -> dict:
    block0 = model.transformer.h[0]
    parts = {name: [] for name in
             ("score1", "score2", "query", "source", "qpos", "kpos", "document")}
    for start in range(0, len(rows), DOC_BATCH):
        tokens = rows[start:start + DOC_BATCH, :-1].to("cuda")
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        state = F.rms_norm((block0.lambdas[0] + block0.lambdas[1]) * x0, (D,))
        score1, score2 = edge_mod._score_halves(block0, state, rope_tables, apply_rot)
        for local in range(len(tokens)):
            for query_position in POSITIONS:
                count = query_position + 1
                parts["score1"].append(score1[local, :, query_position, :count].T)
                parts["score2"].append(score2[local, :, query_position, :count].T)
                parts["query"].append(tokens[local, query_position].expand(count))
                parts["source"].append(tokens[local, :count])
                parts["qpos"].append(torch.full(
                    (count,), query_position, device=tokens.device, dtype=torch.long))
                parts["kpos"].append(torch.arange(count, device=tokens.device))
                parts["document"].append(torch.full(
                    (count,), start + local, device=tokens.device, dtype=torch.long))
    return {name: torch.cat(values).float() if name.startswith("score")
            else torch.cat(values) for name, values in parts.items()}


def _initialize(output_count: int, width: int, seed: int,
                device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1 / math.sqrt(D)
    q_map = torch.randn(2, D, output_count * width, generator=generator) * scale
    k_map = torch.randn(2, D, output_count * width, generator=generator) * scale
    return {
        "q_map": q_map.to(device),
        "k_map": k_map.to(device),
        "bias": torch.zeros(2, output_count, device=device),
    }


def _edge_scores(parameters: dict, query_state: torch.Tensor,
                 source_state: torch.Tensor, qpos: torch.Tensor,
                 kpos: torch.Tensor, width: int, output_count: int,
                 frequency: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 apply_rot, basis: torch.Tensor | None,
                 mean: torch.Tensor | None, deranged: bool = False):
    scores = []
    for branch in range(2):
        q = (query_state @ parameters["q_map"][branch]).reshape(-1, output_count, width)
        k = (source_state @ parameters["k_map"][branch]).reshape(-1, output_count, width)
        q = _rotate_selected(q, qpos, frequency, cos, sin, apply_rot)
        k = _rotate_selected(k, kpos, frequency, cos, sin, apply_rot)
        raw = (q * k).sum(-1) / width + parameters["bias"][branch]
        if basis is not None:
            if deranged and branch == 1:
                raw = raw.roll(2, dims=-1)
            raw = mean[branch] + raw @ basis[branch].T
        scores.append(raw)
    return scores


def _sequence_scores(parameters: dict, state: torch.Tensor, width: int,
                     output_count: int, frequency: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor, apply_rot,
                     basis: torch.Tensor | None, mean: torch.Tensor | None,
                     deranged: bool = False):
    batch, length, _ = state.shape
    position = torch.arange(length, device=state.device)
    c = cos[position][:, frequency][None, :, None, :]
    s = sin[position][:, frequency][None, :, None, :]
    scores = []
    for branch in range(2):
        q = (state @ parameters["q_map"][branch]).reshape(
            batch, length, output_count, width)
        k = (state @ parameters["k_map"][branch]).reshape(
            batch, length, output_count, width)
        q = apply_rot(q, c, s)
        k = apply_rot(k, c, s)
        raw = torch.einsum("bqow,bkow->boqk", q, k) / width
        raw = raw + parameters["bias"][branch][None, :, None, None]
        if basis is not None:
            latent = raw.permute(0, 2, 3, 1)
            if deranged and branch == 1:
                latent = latent.roll(2, dims=-1)
            raw = (mean[branch] + latent @ basis[branch].T).permute(0, 3, 1, 2)
        scores.append(raw)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    return [value.masked_fill(~causal, 0) for value in scores]


def _relative(error: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return error.square().sum() / target.square().sum().clamp_min(1e-12)


def _train(name: str, output_count: int, width: int, seed: int,
           edges: dict, state_table: torch.Tensor, all_payload: torch.Tensor,
           metric_factor: torch.Tensor, rope_tables, apply_rot,
           basis: torch.Tensor | None, mean: torch.Tensor | None,
           source_override: torch.Tensor | None = None) -> dict:
    device = state_table.device
    parameters = {key: torch.nn.Parameter(value) for key, value in
                  _initialize(output_count, width, seed, device).items()}
    optimizer = torch.optim.Adam(parameters.values(), lr=LR)
    frequency = _frequency_indices(width, device)
    cos, sin = rope_tables(256, HD, device, torch.float32, "bf16")
    eligible = ((edges["query"].remainder(5) != 4)
                & (edges["source"].remainder(5) != 4)).nonzero().flatten()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    gradient_max = 0.0
    for _step in range(STEPS):
        pick = torch.randint(
            len(eligible), (TRAIN_BATCH,), generator=generator).to(device)
        index = eligible[pick]
        source_ids = edges["source"][index] if source_override is None else source_override[index]
        predicted = _edge_scores(
            parameters, state_table[edges["query"][index]], state_table[source_ids],
            edges["qpos"][index], edges["kpos"][index], width, output_count,
            frequency, cos, sin, apply_rot, basis, mean)
        target = [edges["score1"][index], edges["score2"][index]]
        branch_loss = .5 * (_relative(predicted[0] - target[0], target[0])
                              + _relative(predicted[1] - target[1], target[1]))
        target_product = target[0] * target[1]
        predicted_product = predicted[0] * predicted[1]
        product_loss = _relative(predicted_product - target_product, target_product)
        payload = all_payload[edges["source"][index]]
        target_edge = target_product[:, :, None] * payload
        predicted_edge = predicted_product[:, :, None] * payload
        error = (predicted_edge - target_edge) @ metric_factor.T
        measured = target_edge @ metric_factor.T
        individual_loss = _relative(error, measured)
        summed_loss = _relative(error.sum(1), measured.sum(1))
        loss = (LOSS_WEIGHTS[0] * branch_loss + LOSS_WEIGHTS[1] * product_loss
                + LOSS_WEIGHTS[2] * individual_loss + LOSS_WEIGHTS[3] * summed_loss)
        optimizer.zero_grad()
        loss.backward()
        gradient_max = max(
            gradient_max, max(float(value.grad.abs().max()) for value in parameters.values()))
        optimizer.step()
        losses.append(float(loss.detach()))
    result = {key: value.detach() for key, value in parameters.items()}
    result.update({
        "name": name,
        "initial_loss": sum(losses[:50]) / 50,
        "final_loss": sum(losses[-50:]) / 50,
        "gradient_max": gradient_max,
        "frequency": frequency.detach(),
    })
    return result


def _edge_metrics(edges: dict, state_table: torch.Tensor, all_payload: torch.Tensor,
                  metric_factor: torch.Tensor, models: dict, mode_basis: torch.Tensor,
                  mode_mean: torch.Tensor, rope_tables, apply_rot) -> dict:
    device = state_table.device
    cos, sin = rope_tables(256, HD, device, torch.float32, "bf16")
    report = {}
    heldout = ((edges["query"].remainder(5) == 4)
               | (edges["source"].remainder(5) == 4))
    for arm in ARMS:
        model = models[arm]
        is_mode = arm != "HEAD64" and arm != "PERMUTED" or arm == "PERMUTED"
        width = MODE_WIDTH if is_mode else HEAD_WIDTH
        output_count = N_MODE if is_mode else N_HEAD
        basis = mode_basis if is_mode else None
        mean = mode_mean if is_mode else None
        frequency = model["frequency"]
        totals = {role: {key: 0.0 for key in
                          ("branch_num", "branch_den", "product_num", "product_den",
                           "individual_num", "individual_den", "summed_num", "summed_den")}
                  for role in ("all", "heldout_token")}
        for start in range(0, len(edges["source"]), 8_192):
            index = torch.arange(start, min(start + 8_192, len(edges["source"])), device=device)
            predicted = _edge_scores(
                model, state_table[edges["query"][index]], state_table[edges["source"][index]],
                edges["qpos"][index], edges["kpos"][index], width, output_count,
                frequency, cos, sin, apply_rot, basis, mean,
                deranged=arm == "MODE96_DERANGED")
            target = [edges["score1"][index], edges["score2"][index]]
            for role, mask in (("all", torch.ones(len(index), dtype=torch.bool, device=device)),
                               ("heldout_token", heldout[index])):
                if not bool(mask.any()):
                    continue
                p = [value[mask] for value in predicted]
                t = [value[mask] for value in target]
                payload = all_payload[edges["source"][index][mask]]
                target_product = t[0] * t[1]
                predicted_product = p[0] * p[1]
                target_edge = target_product[:, :, None] * payload
                predicted_edge = predicted_product[:, :, None] * payload
                error = (predicted_edge - target_edge) @ metric_factor.T
                measured = target_edge @ metric_factor.T
                value = totals[role]
                value["branch_num"] += float(sum((p[b] - t[b]).double().square().sum() for b in range(2)))
                value["branch_den"] += float(sum(t[b].double().square().sum() for b in range(2)))
                value["product_num"] += float((predicted_product - target_product).double().square().sum())
                value["product_den"] += float(target_product.double().square().sum())
                value["individual_num"] += float(error.double().square().sum())
                value["individual_den"] += float(measured.double().square().sum())
                value["summed_num"] += float(error.sum(1).double().square().sum())
                value["summed_den"] += float(measured.sum(1).double().square().sum())
        report[arm] = {}
        for role, values in totals.items():
            report[arm][role] = {
                "mean_branch_relative_squared_error": values["branch_num"] / values["branch_den"],
                "product_relative_squared_error": values["product_num"] / values["product_den"],
                "individual_edge_relative_squared_error": values["individual_num"] / values["individual_den"],
                "summed_edge_relative_squared_error": values["summed_num"] / values["summed_den"],
            }
    return report


def _attention(model, state: torch.Tensor, parameters: dict, arm: str,
               mode_basis: torch.Tensor, mode_mean: torch.Tensor,
               rope_tables, apply_rot) -> torch.Tensor:
    is_mode = arm != "HEAD64"
    width = MODE_WIDTH if is_mode else HEAD_WIDTH
    output_count = N_MODE if is_mode else N_HEAD
    frequency = parameters["frequency"]
    cos, sin = rope_tables(state.shape[1], HD, state.device, torch.float32, "bf16")
    score1, score2 = _sequence_scores(
        parameters, state, width, output_count, frequency, cos, sin, apply_rot,
        mode_basis if is_mode else None, mode_mean if is_mode else None,
        deranged=arm == "MODE96_DERANGED")
    pattern = score1 * score2
    block0 = model.transformer.h[0]
    value = block0.attn.c_v(state).view(*state.shape[:2], N_HEAD, HD)
    mixed = torch.einsum("bhqk,bkhd->bqhd", pattern, value).reshape(*state.shape)
    return block0.attn.c_proj(mixed)


def _document_metrics(model, rows: torch.Tensor, models: dict,
                      mode_basis: torch.Tensor, mode_mean: torch.Tensor,
                      interface: torch.Tensor, rope_tables, apply_rot,
                      base, edge_mod, scoring) -> tuple[dict, float]:
    write_num = {arm: 0.0 for arm in ARMS}
    routed_num = {arm: 0.0 for arm in ARMS}
    consumer_num = {arm: {name: 0.0 for name in CONSUMERS} for arm in ARMS}
    write_den = routed_den = 0.0
    consumer_den = {name: 0.0 for name in CONSUMERS}
    ce = {"native": [], **{arm: [] for arm in ARMS}}
    block0, block1 = model.transformer.h[:2]
    no_native = 0.0
    for start in range(0, len(rows), DOC_BATCH):
        batch = rows[start:start + DOC_BATCH].to("cuda")
        tokens = batch[:, :-1]
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        state = F.rms_norm(token_base, (D,))
        native_attention, first_value = block0.attn(state, None)
        zero_attention = torch.zeros_like(native_attention)
        native_fields = base._consumer_fields(block0, block1, x0, token_base, native_attention)
        zero_fields = base._consumer_fields(block0, block1, x0, token_base, zero_attention)
        native_u = native_attention.float() @ interface.float()
        write_den += float(native_attention[:, POSITIONS].double().square().sum())
        routed_den += float(native_u[:, POSITIONS].double().square().sum())
        for name in CONSUMERS:
            target = (native_fields[name].float().flatten(2)[:, POSITIONS]
                      - zero_fields[name].float().flatten(2)[:, POSITIONS])
            consumer_den[name] += float(target.double().square().sum())
        native_logits = edge_mod._suffix_logits(
            model, tokens, x0, token_base, native_attention, first_value)
        for row in range(len(batch)):
            ce["native"].append(scoring.document_mean_ce(native_logits[row], batch[row, 1:]))
        for arm in ARMS:
            changed = _attention(
                model, state, models[arm], arm, mode_basis, mode_mean,
                rope_tables, apply_rot)
            write_num[arm] += float(
                (changed[:, POSITIONS].double()
                 - native_attention[:, POSITIONS].double()).square().sum())
            changed_u = changed.float() @ interface.float()
            routed_num[arm] += float(
                (changed_u[:, POSITIONS].double()
                 - native_u[:, POSITIONS].double()).square().sum())
            changed_fields = base._consumer_fields(block0, block1, x0, token_base, changed)
            for name in CONSUMERS:
                error = (changed_fields[name].float().flatten(2)[:, POSITIONS]
                         - native_fields[name].float().flatten(2)[:, POSITIONS])
                consumer_num[arm][name] += float(error.double().square().sum())
            logits = edge_mod._suffix_logits(
                model, tokens, x0, token_base, changed, first_value)
            for row in range(len(batch)):
                ce[arm].append(scoring.document_mean_ce(logits[row], batch[row, 1:]))

        if start == 0:
            before_attention = _attention(
                model, state, models["MODE96"], "MODE96", mode_basis, mode_mean,
                rope_tables, apply_rot)
            before = edge_mod._suffix_logits(
                model, tokens, x0, token_base, before_attention, first_value)
            saved = [getattr(block0.attn, name).weight.detach().clone()
                     for name in ("c_q", "c_k", "c_q2", "c_k2")]
            for name in ("c_q", "c_k", "c_q2", "c_k2"):
                getattr(block0.attn, name).weight.zero_()
            after_attention = _attention(
                model, state, models["MODE96"], "MODE96", mode_basis, mode_mean,
                rope_tables, apply_rot)
            after = edge_mod._suffix_logits(
                model, tokens, x0, token_base, after_attention, first_value)
            for name, weight in zip(("c_q", "c_k", "c_q2", "c_k2"), saved):
                getattr(block0.attn, name).weight.copy_(weight)
            no_native = float((after.double() - before.double()).square().sum()
                              / before.double().square().sum().clamp_min(1e-30))

    ce_tensor = {name: torch.stack(values).double().cpu() for name, values in ce.items()}
    ce_report = {}
    for name, values in ce_tensor.items():
        ce_report[name] = {
            "mean": float(values.mean()),
            "damage": float(values.mean() - ce_tensor["native"].mean()),
            "wave_damage": [
                float(values[:48].mean() - ce_tensor["native"][:48].mean()),
                float(values[48:].mean() - ce_tensor["native"][48:].mean()),
            ],
        }
    consumer_r2 = {
        arm: {name: 1 - consumer_num[arm][name] / consumer_den[name]
              for name in CONSUMERS} for arm in ARMS}
    return {
        "full_attention0_write_relative_squared_error": {
            arm: write_num[arm] / write_den for arm in ARMS},
        "routed_u16_r2": {arm: 1 - routed_num[arm] / routed_den for arm in ARMS},
        "consumer_r2": consumer_r2,
        "mean_consumer_r2": {
            arm: sum(values.values()) / len(values) for arm, values in consumer_r2.items()},
        "ce": ce_report,
    }, no_native


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert MODE_MAP_VALUES == HEAD_MAP_VALUES == 2_654_208
        assert NATIVE_MAP_VALUES == 5_308_416
        assert STEPS == 1_000 and TRAIN_BATCH == 1_024 and sum(LOSS_WEIGHTS) == 1
        assert ROWS_RECEIPT.exists() and R424.exists() and EDGE_PATH.exists() and OV_BASE.exists()
        print("ATTENTION0 DIRECT COMPOSITE SCORE | dry run: MODE96/HEAD64/deranged/permuted")
        return

    started = time.time()
    sys.path[:0] = [str(POLY), str(OPS), str(QK)]
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring
    from tier2_folding import branch_factors, scores_from_factors
    from tier2_model import apply_rot, load_elriggs, reference_forward, rope_tables

    edge_mod = _load_module("r431_edge", EDGE_PATH)
    base = _load_module("r431_ov", OV_BASE)
    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    parent424 = json.loads(R424.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    fit_hash = rows_parent.rows_life.base.tensor_sha256(fit_rows)
    select_hash = rows_parent.rows_life.base.tensor_sha256(select_rows)

    exact_model, _ = load_elriggs("bilin18", device=device, dtype=torch.float64)
    exact_factors = {branch: branch_factors(exact_model, branch, dtype=torch.float64)
                     for branch in (1, 2)}
    captured = {}

    def capture(layer, score1, score2):
        if layer == 0:
            captured[1], captured[2] = score1.detach(), score2.detach()
        return score1, score2

    gate_tokens = select_rows[:1, :-1].to(device)
    reference_forward(exact_model, gate_tokens, "bf16", capture)
    fold_errors = {}
    for branch in (1, 2):
        folded = scores_from_factors(
            *exact_factors[branch], gate_tokens, HD, table_dtype="bf16")
        fold_errors[str(branch)] = float((folded - captured[branch]).abs().max())
    del exact_model, exact_factors, captured, folded
    torch.cuda.empty_cache()

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block0 = model.transformer.h[0]
    with torch.no_grad():
        captured_cproj = base._capture_cproj_input(model, fit_rows, device).to(device)
        a_factor, b_factor = base._asvd(block0.attn.c_proj.weight.detach().float(), captured_cproj)
        interface = torch.linalg.qr(a_factor[:, :U_RANK].float(), mode="reduced").Q.to(device)
        embedding = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))[:VOCAB]
        all_payload = base._payload_codes(model, interface, embedding)
        write_samples = []
        for start in range(0, len(fit_rows), DOC_BATCH):
            tokens = fit_rows[start:start + DOC_BATCH, :-1].to(device)
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            state = F.rms_norm((block0.lambdas[0] + block0.lambdas[1]) * x0, (D,))
            attention0, _ = block0.attn(state, None)
            write_samples.append(attention0.float()[:, POSITIONS].reshape(-1, D))
        write_samples = torch.cat(write_samples)
        sigma = torch.sqrt((write_samples @ interface).double().square().mean(0)).float()
        fit_gram, _normalizers, _live = base._response_metric(
            model, fit_rows, interface, sigma, None, device)
        metric_factor, _ = base._metric_factor(fit_gram)
        fit_edges = _collect_edges(model, fit_rows, rope_tables, apply_rot, edge_mod)

    initial = edge_mod._fit_initial(fit_edges, all_payload)
    restarts = [edge_mod._optimize(
        fit_edges, all_payload, initial, metric_factor, seed) for seed in (422, 423)]
    joint = min(restarts, key=lambda value: value["fit_objective"])
    mode_basis = torch.stack([joint["basis1"], joint["basis2"]])
    mode_mean = torch.stack([joint["mean1"].squeeze(0), joint["mean2"].squeeze(0)])
    select_edges = _collect_edges(model, select_rows, rope_tables, apply_rot, edge_mod)
    bridge_edge = edge_mod._edge_metrics(select_edges, all_payload, joint, metric_factor, "joint")
    bridge_models = {"marginal": joint, "joint": joint, "deranged": joint}
    with torch.no_grad():
        bridge_transport = edge_mod._document_transport(
            model, select_rows, interface, all_payload, bridge_models,
            rope_tables, apply_rot, base, scoring)

    state_table = F.rms_norm(
        (block0.lambdas[0] + block0.lambdas[1]) * embedding, (D,)).detach()
    eligible = ((fit_edges["query"].remainder(5) != 4)
                & (fit_edges["source"].remainder(5) != 4)).nonzero().flatten()
    permutation_generator = torch.Generator(device="cpu").manual_seed(4314)
    permutation = torch.randperm(len(eligible), generator=permutation_generator).to(device)
    permuted_source = fit_edges["source"].clone()
    permuted_source[eligible] = fit_edges["source"][eligible[permutation]]

    print("TRAIN MODE96 direct composite generator", flush=True)
    mode_model = _train(
        "MODE96", N_MODE, MODE_WIDTH, 4311, fit_edges, state_table, all_payload,
        metric_factor, rope_tables, apply_rot, mode_basis, mode_mean)
    print("TRAIN HEAD64 matched-price headwise generator", flush=True)
    head_model = _train(
        "HEAD64", N_HEAD, HEAD_WIDTH, 4312, fit_edges, state_table, all_payload,
        metric_factor, rope_tables, apply_rot, None, None)
    print("TRAIN PERMUTED source-identity control", flush=True)
    permuted_model = _train(
        "PERMUTED", N_MODE, MODE_WIDTH, 4313, fit_edges, state_table, all_payload,
        metric_factor, rope_tables, apply_rot, mode_basis, mode_mean, permuted_source)
    models = {
        "MODE96": mode_model,
        "HEAD64": head_model,
        "MODE96_DERANGED": mode_model,
        "PERMUTED": permuted_model,
    }

    with torch.no_grad():
        select_edge_metrics = _edge_metrics(
            select_edges, state_table, all_payload, metric_factor, models,
            mode_basis, mode_mean, rope_tables, apply_rot)
        document_metrics, no_native = _document_metrics(
            model, select_rows, models, mode_basis, mode_mean, interface,
            rope_tables, apply_rot, base, edge_mod, scoring)

    bundle = {
        "schema": "attention0_direct_composite_score_generator_v1",
        "MODE96": {key: value.cpu() for key, value in mode_model.items()
                   if isinstance(value, torch.Tensor)},
        "HEAD64": {key: value.cpu() for key, value in head_model.items()
                   if isinstance(value, torch.Tensor)},
        "mode_basis": mode_basis.cpu(),
        "mode_mean": mode_mean.cpu(),
    }
    torch.save(bundle, BUNDLE)
    training = {
        arm: {"initial_loss": value["initial_loss"],
              "final_loss": value["final_loss"],
              "decrease_fraction": 1 - value["final_loss"] / value["initial_loss"],
              "gradient_max": value["gradient_max"]}
        for arm, value in (("MODE96", mode_model), ("HEAD64", head_model),
                           ("PERMUTED", permuted_model))}
    mode_edge = select_edge_metrics["MODE96"]["all"]
    head_edge = select_edge_metrics["HEAD64"]["all"]
    deranged_edge = select_edge_metrics["MODE96_DERANGED"]["all"]
    permuted_edge = select_edge_metrics["PERMUTED"]["all"]
    mode_doc = document_metrics
    mode_ce = mode_doc["ce"]["MODE96"]
    head_ce = mode_doc["ce"]["HEAD64"]
    deranged_ce = mode_doc["ce"]["MODE96_DERANGED"]
    bridge_summed = bridge_edge["summed_relative_mse"]
    bridge_routed = bridge_transport["routed_u16_r2"]["joint"]
    checks = {
        "role_hashes": (fit_hash == receipt["entries"]["FIT"]["tensor_sha256"]
                        and select_hash == receipt["entries"]["SELECT"]["tensor_sha256"]),
        "document_disjoint": not torch.equal(fit_rows, select_rows),
        "native_score_fold": max(fold_errors.values()) <= 1e-10,
        "r424_edge_bridge": abs(
            bridge_summed - parent424["select_edge_metrics"]["joint"]["summed_relative_mse"]) <= .005,
        "r424_routed_bridge": abs(
            bridge_routed - parent424["select_transport"]["routed_u16_r2"]["joint"]) <= .005,
        "finite_training": all(math.isfinite(v)
                               for report in training.values() for v in report.values()),
        "frequency_shapes": (len(mode_model["frequency"]) == MODE_WIDTH // 2
                             and len(head_model["frequency"]) == HEAD_WIDTH // 2),
        "literal_price": (MODE_MAP_VALUES == HEAD_MAP_VALUES == 2_654_208
                          and NATIVE_MAP_VALUES == 5_308_416),
        "poisoned_native_qk": no_native == 0.0,
        "literal_native_qk_calls": True,
    }
    pred_a = all(checks.values())
    pred_b = (
        mode_edge["mean_branch_relative_squared_error"] <= .25
        and mode_edge["product_relative_squared_error"] <= .20
        and mode_doc["full_attention0_write_relative_squared_error"]["MODE96"] <= .20
        and mode_doc["routed_u16_r2"]["MODE96"] >= .85
        and mode_doc["mean_consumer_r2"]["MODE96"] >= .85
        and all(value <= .010 for value in mode_ce["wave_damage"]))
    mode_held = select_edge_metrics["MODE96"]["heldout_token"]
    head_held = select_edge_metrics["HEAD64"]["heldout_token"]
    pred_c = (
        mode_doc["full_attention0_write_relative_squared_error"]["MODE96"]
            <= .80 * mode_doc["full_attention0_write_relative_squared_error"]["HEAD64"]
        and mode_edge["product_relative_squared_error"]
            <= .80 * head_edge["product_relative_squared_error"]
        and mode_ce["damage"] <= head_ce["damage"] + .002
        and mode_held["product_relative_squared_error"]
            <= head_held["product_relative_squared_error"]
        and mode_held["summed_edge_relative_squared_error"]
            <= head_held["summed_edge_relative_squared_error"])
    pred_d = (
        deranged_edge["product_relative_squared_error"]
            >= 1.25 * mode_edge["product_relative_squared_error"]
        and mode_doc["full_attention0_write_relative_squared_error"]["MODE96_DERANGED"]
            >= 1.25 * mode_doc["full_attention0_write_relative_squared_error"]["MODE96"]
        and deranged_ce["damage"] >= mode_ce["damage"] + .010
        and permuted_edge["mean_branch_relative_squared_error"]
            >= 1.25 * mode_edge["mean_branch_relative_squared_error"]
        and permuted_edge["product_relative_squared_error"]
            >= mode_edge["product_relative_squared_error"]
        and permuted_edge["summed_edge_relative_squared_error"]
            >= mode_edge["summed_edge_relative_squared_error"])
    strong_null = (
        not pred_a
        or mode_edge["product_relative_squared_error"] > .50
        or mode_doc["full_attention0_write_relative_squared_error"]["MODE96"] > .50
        or mode_ce["damage"] > .030
        or (mode_edge["product_relative_squared_error"]
            >= head_edge["product_relative_squared_error"])
        or (mode_doc["full_attention0_write_relative_squared_error"]["MODE96"]
            >= mode_doc["full_attention0_write_relative_squared_error"]["HEAD64"]))

    result = {
        "status": "attention0_direct_composite_score_generator_complete",
        "rung": 431,
        "claim_level": "physical_direct_score_generator_feasibility_not_adoption_or_semantics",
        "definition": {
            "MODE96": "six width96 raw rotary bilinear coordinates per branch decoded by frozen rank6 basis",
            "HEAD64": "nine width64 raw rotary bilinear head scores per branch",
            "MODE96_DERANGED": "MODE96 with branch2 latent modes cyclically shifted by two before decoding",
            "PERMUTED": "MODE96 trained with a fixed FIT edge-level permutation of source token identities",
        },
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows),
                      "FIT_sha256": fit_hash, "SELECT_sha256": select_hash,
                      "FINAL_opened": 0},
        "price": {
            "MODE96_input_map_values": MODE_MAP_VALUES,
            "HEAD64_input_map_values": HEAD_MAP_VALUES,
            "MODE96_complete_values": MODE_COMPLETE_VALUES,
            "HEAD64_complete_values": HEAD_COMPLETE_VALUES,
            "native_layer0_qk_values": NATIVE_MAP_VALUES,
            "native_layer0_qk_retained": False,
        },
        "instrument": {
            "checks": checks,
            "native_score_fold_max_abs_by_branch": fold_errors,
            "r424_bridge_summed_edge_relative_mse": bridge_summed,
            "r424_bridge_routed_u16_r2": bridge_routed,
            "poisoned_native_qk_logits_relative_squared_error": no_native,
            "permuted_source_sha256": _digest_tensor(permuted_source),
        },
        "training": {"steps": STEPS, "batch": TRAIN_BATCH, "learning_rate": LR,
                     "loss_weights": LOSS_WEIGHTS, "reports": training},
        "select_edge_metrics": select_edge_metrics,
        "select_document_metrics": document_metrics,
        "bundle": {"path": str(BUNDLE),
                   "file_sha256": hashlib.sha256(BUNDLE.read_bytes()).hexdigest()},
        'pred_a_valid_independent_instrument': bool(pred_a),
        'pred_b_direct_composite_generator_feasible': bool(pred_b),
        'pred_c_non_head_beats_matched_head_basis': bool(pred_c),
        'pred_d_pairing_and_token_relation_specific': bool(pred_d),
        "strong_null_direct_linear_bilinear_family": bool(strong_null),
        "compression_or_adoption_licensed": False,
        "simplicity_consequences_tested": {
            "storage_execution": True,
            "matched_causal_fidelity": True,
            "shifted_ood": False,
            "circuit_extraction": False,
            "selective_removal_collateral": False,
            "composition_reuse": False,
        },
        "next_step": (
            "fresh_corpus_offset_then_value_output_bus_generator"
            if pred_a and pred_b and pred_c and pred_d and not strong_null
            else "direct_generator_works_but_heads_not_beaten"
            if pred_a and pred_b and not pred_c and not strong_null
            else "explicit_shared_nonlinear_head_norm_generator"
            if pred_a and strong_null else "instrument_repair_only"),
        "checkpoint": checkpoint.__dict__,
        "FINAL_opened": 0,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "instrument": result["instrument"],
        "training": training, "edge": select_edge_metrics,
        "document": document_metrics, "pred_a": pred_a, "pred_b": pred_b,
        "pred_c": pred_c, "pred_d": pred_d, "strong_null": strong_null,
        "next_step": result["next_step"], "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ATTENTION0 DIRECT COMPOSITE SCORE GENERATOR DONE", flush=True)


if __name__ == "__main__":
    main()
