"""RUNG 444 -- ARCHETYPAL SAE IN ATTENTION0'S DOWNSTREAM-RESPONSE METRIC.

The modeled row is the exact U16 contribution of one realized causal edge,
sum_h(score1 * score2 * OV-payload).  Unlike rung439, the loss is measured
through rung424's downstream-response Gram.  Native generators remain live.
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
from scipy.optimize import linear_sum_assignment


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_response_archetypal_sae_results.json"
BUNDLE = BQ / "attention0_response_archetypal_sae_bundle.pt"
R424_SOURCE = OPS / "attention0_realized_edge_block_term.py"
R424_RESULT = BQ / "attention0_realized_edge_block_term_results.json"
BASE_SOURCE = OPS / "attention0_ov_downstream_codebook.py"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"

D = 1152
N_HEAD = 9
U_RANK = 16
N_ATOMS = 32
TOPK = 4
SUPPORT_POOL = 2048
TRAIN_STEPS = 512
TRAIN_BATCH = 4096
TRAIN_LR = .01
DOC_BATCH = 4
POSITIONS = tuple(range(16, 241, 16))
ARMS = ("U32", "A32", "P32")
CONSUMERS = ("mlp0", "q1", "k1", "q2", "k2", "fresh_v")
SCREEN_VALUES = N_ATOMS * U_RANK + U_RANK * N_ATOMS + N_ATOMS
SCREEN_BYTES = 4 * SCREEN_VALUES


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    header = f"{cpu.dtype}|{tuple(cpu.shape)}|".encode()
    return hashlib.sha256(header + cpu.numpy().tobytes()).hexdigest()


def _response_rows(edges: dict, all_payload: torch.Tensor,
                   source_override: torch.Tensor | None = None) -> torch.Tensor:
    source = edges["source"] if source_override is None else source_override
    parts = []
    for start in range(0, len(source), 16_384):
        sl = slice(start, start + 16_384)
        payload = all_payload[source[sl]]
        parts.append((edges["score1"][sl, :, None] *
                      edges["score2"][sl, :, None] * payload).sum(1))
    return torch.cat(parts).float()


class ResponseSAE(torch.nn.Module):
    def __init__(self, pool: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor,
                 seed: int, constrained: bool):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer("pool", pool)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.constrained = constrained
        self.encoder_weight = torch.nn.Parameter(
            .02 * torch.randn(U_RANK, N_ATOMS, generator=generator, device="cpu").to(pool.device))
        self.encoder_bias = torch.nn.Parameter(torch.zeros(N_ATOMS, device=pool.device))
        selected = torch.randperm(len(pool), generator=generator)[:N_ATOMS].to(pool.device)
        if constrained:
            logits = torch.full((N_ATOMS, len(pool)), -4.0, device=pool.device)
            logits[torch.arange(N_ATOMS, device=pool.device), selected] = 4.0
            self.support_logits = torch.nn.Parameter(logits)
            self.register_parameter("free_decoder", None)
        else:
            self.free_decoder = torch.nn.Parameter(pool[selected].clone())
            self.register_parameter("support_logits", None)

    def decoder(self) -> torch.Tensor:
        if self.constrained:
            return torch.softmax(self.support_logits, dim=1) @ self.pool
        return self.free_decoder

    def code(self, value: torch.Tensor) -> torch.Tensor:
        code = F.relu(((value - self.mean) / self.scale) @ self.encoder_weight + self.encoder_bias)
        top_values, top_indices = torch.topk(code, TOPK, dim=1)
        sparse = torch.zeros_like(code)
        sparse.scatter_(1, top_indices, top_values)
        return sparse

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.code(value) @ self.decoder()


def _metric_loss(target: torch.Tensor, predicted: torch.Tensor,
                 metric_factor: torch.Tensor) -> torch.Tensor:
    numerator = ((target - predicted) @ metric_factor.T).square().sum()
    denominator = (target @ metric_factor.T).square().sum().clamp_min(1e-30)
    return numerator / denominator


def _exact_loss(model: ResponseSAE, values: torch.Tensor,
                metric_factor: torch.Tensor) -> float:
    numerator = denominator = 0.0
    with torch.no_grad():
        for start in range(0, len(values), 16_384):
            target = values[start:start + 16_384]
            predicted = model(target)
            numerator += float(((target - predicted) @ metric_factor.T).double().square().sum())
            denominator += float((target @ metric_factor.T).double().square().sum())
    return numerator / denominator


def _pool(values: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(values), generator=generator)[:SUPPORT_POOL]
    return values[indices.to(values.device)].clone(), indices


def _fit(values: torch.Tensor, pool_values: torch.Tensor, metric_factor: torch.Tensor,
         seed: int, constrained: bool) -> tuple[ResponseSAE, dict]:
    mean = values.mean(0)
    scale = values.std(0).clamp_min(1e-6)
    model = ResponseSAE(pool_values, mean, scale, seed, constrained).to(values.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    initial_loss = _exact_loss(model, values, metric_factor)
    gradient_max = {name: 0.0 for name, _ in model.named_parameters()}
    for step in range(TRAIN_STEPS):
        indices = torch.randint(
            len(values), (min(TRAIN_BATCH, len(values)),), generator=generator,
            device="cpu").to(values.device)
        target = values[indices]
        loss = _metric_loss(target, model(target), metric_factor)
        optimizer.zero_grad()
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                gradient_max[name] = max(gradient_max[name], float(parameter.grad.abs().max()))
        optimizer.step()
    final_loss = _exact_loss(model, values, metric_factor)
    weights = torch.softmax(model.support_logits, dim=1).detach() if constrained else None
    certificate = None
    if constrained:
        replay = weights @ pool_values
        certificate = {
            "weight_min": float(weights.min()),
            "weight_sum_max_abs_error": float((weights.sum(1) - 1).abs().max()),
            "atom_replay_max_abs": float((replay - model.decoder()).abs().max()),
        }
    return model, {
        "initial_exact_fit_loss": initial_loss,
        "final_exact_fit_loss": final_loss,
        "loss_decrease_fraction": (initial_loss - final_loss) / max(initial_loss, 1e-30),
        "gradient_max": gradient_max,
        "certificate": certificate,
    }


def _metric_atoms(model: ResponseSAE, metric_factor: torch.Tensor) -> torch.Tensor:
    value = model.decoder().detach() @ metric_factor.T
    return F.normalize(value, dim=1)


def _matched_cosine(left: ResponseSAE, right: ResponseSAE,
                    metric_factor: torch.Tensor) -> dict:
    cosine = _metric_atoms(left, metric_factor) @ _metric_atoms(right, metric_factor).T
    rows, cols = linear_sum_assignment((-cosine).detach().cpu().numpy())
    matched = cosine[torch.tensor(rows, device=cosine.device),
                     torch.tensor(cols, device=cosine.device)]
    return {
        "median": float(matched.median()),
        "mean": float(matched.mean()),
        "minimum": float(matched.min()),
    }


def _serialize_model(model: ResponseSAE, pool_indices: torch.Tensor) -> dict:
    value = {
        "encoder_weight": model.encoder_weight.detach().half().cpu(),
        "encoder_bias": model.encoder_bias.detach().half().cpu(),
        "decoder": model.decoder().detach().half().cpu(),
        "mean": model.mean.detach().half().cpu(),
        "scale": model.scale.detach().half().cpu(),
        "pool_indices": pool_indices.cpu(),
    }
    if model.constrained:
        value["support_weights"] = torch.softmax(
            model.support_logits.detach(), dim=1).half().cpu()
    return value


def _consumer_transport(model, rows, interface, all_payload, saes, r424, base,
                        rope_tables, apply_rot, scoring):
    routed_sse = {arm: 0.0 for arm in ARMS}
    consumer_sse = {arm: {name: 0.0 for name in CONSUMERS} for arm in ARMS}
    target_routed = 0.0
    target_consumer = {name: 0.0 for name in CONSUMERS}
    ce = {"native": [], **{arm: [] for arm in ARMS}}
    edge_identity_num = edge_identity_den = 0.0
    block0, block1 = model.transformer.h[:2]
    for start in range(0, len(rows), DOC_BATCH):
        batch = rows[start:start + DOC_BATCH].to("cuda")
        tokens = batch[:, :-1]
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        state = F.rms_norm(token_base, (D,))
        native_attention, first_value = block0.attn(state, None)
        native_u = native_attention.float() @ interface.float()
        score1, score2 = r424._score_halves(block0, state, rope_tables, apply_rot)
        product = score1 * score2
        payload = all_payload[tokens]
        edge_values = torch.einsum("bhqk,bkhu->bqku", product, payload)
        causal = torch.tril(torch.ones(
            edge_values.shape[1], edge_values.shape[2], dtype=torch.bool,
            device=edge_values.device))[None].expand(len(tokens), -1, -1)
        exact_u = edge_values.sum(2)
        reference_u = torch.einsum("bhqk,bkhu->bqu", product, payload)
        edge_identity_num += float((exact_u.double() - reference_u.double()).square().sum())
        edge_identity_den += float(reference_u.double().square().sum())
        remainder = native_u - exact_u

        native_fields = base._consumer_fields(block0, block1, x0, token_base, native_attention)
        tail_attention = native_attention - (native_u @ interface.float().T).to(native_attention.dtype)
        tail_fields = base._consumer_fields(block0, block1, x0, token_base, tail_attention)
        target_routed += float(native_u[:, POSITIONS].double().square().sum())
        for name in CONSUMERS:
            response = (native_fields[name].float().flatten(2)[:, POSITIONS]
                        - tail_fields[name].float().flatten(2)[:, POSITIONS])
            target_consumer[name] += float(response.double().square().sum())
        native_logits = r424._suffix_logits(
            model, tokens, x0, token_base, native_attention, first_value)
        for row in range(len(batch)):
            ce["native"].append(scoring.document_mean_ce(native_logits[row], batch[row, 1:]))

        flat_edges = edge_values[causal]
        for arm, sae in saes.items():
            reconstructed = []
            for low in range(0, len(flat_edges), 16_384):
                reconstructed.append(sae(flat_edges[low:low + 16_384]))
            rec_tensor = torch.zeros_like(edge_values)
            rec_tensor[causal] = torch.cat(reconstructed)
            approx_u = rec_tensor.sum(2)
            changed = native_attention + ((approx_u + remainder - native_u) @
                                            interface.float().T).to(native_attention.dtype)
            changed_u = changed.float() @ interface.float()
            routed_sse[arm] += float(
                (changed_u[:, POSITIONS].double() - native_u[:, POSITIONS].double()).square().sum())
            changed_fields = base._consumer_fields(block0, block1, x0, token_base, changed)
            for name in CONSUMERS:
                consumer_sse[arm][name] += float(
                    (changed_fields[name].float().flatten(2)[:, POSITIONS].double()
                     - native_fields[name].float().flatten(2)[:, POSITIONS].double()).square().sum())
            logits = r424._suffix_logits(model, tokens, x0, token_base, changed, first_value)
            for row in range(len(batch)):
                ce[arm].append(scoring.document_mean_ce(logits[row], batch[row, 1:]))

    ce_tensors = {name: torch.stack(values).double().cpu() for name, values in ce.items()}
    ce_public = {
        name: {
            "mean": float(values.mean()),
            "damage": float(values.mean() - ce_tensors["native"].mean()),
            "wave_damage": [
                float(values[:48].mean() - ce_tensors["native"][:48].mean()),
                float(values[48:].mean() - ce_tensors["native"][48:].mean()),
            ],
        }
        for name, values in ce_tensors.items()
    }
    routed = {arm: 1 - routed_sse[arm] / target_routed for arm in ARMS}
    consumers = {
        arm: {name: 1 - consumer_sse[arm][name] / target_consumer[name]
              for name in CONSUMERS} for arm in ARMS}
    return {
        "routed_u16_r2": routed,
        "consumer_r2": consumers,
        "mean_consumer_r2": {
            arm: sum(values.values()) / len(values) for arm, values in consumers.items()},
        "ce": ce_public,
        "native_edge_sum_relative_squared_error": edge_identity_num / edge_identity_den,
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert N_ATOMS == 32 and TOPK == 4 and SUPPORT_POOL == 2048
        assert TRAIN_STEPS == 512 and TRAIN_BATCH == 4096 and TRAIN_LR == .01
        assert SCREEN_VALUES == 1056 and SCREEN_BYTES == 4224
        assert R424_SOURCE.exists() and R424_RESULT.exists() and ROWS_RECEIPT.exists()
        print("ATTENTION0 RESPONSE ARCHETYPAL SAE | dry run: U32/A32/P32, FIT halves, SELECT consequences")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(QK))
    r424 = _load_module("response_archetype_r424", R424_SOURCE)
    base = _load_module("response_archetype_base", BASE_SOURCE)
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring
    from tier2_model import rope_tables, apply_rot

    receipt = json.loads(ROWS_RECEIPT.read_text())
    parent = json.loads(R424_RESULT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    fit_hash = rows_parent.rows_life.base.tensor_sha256(fit_rows)
    select_hash = rows_parent.rows_life.base.tensor_sha256(select_rows)
    model, checkpoint = facade.load_bilin18(device=torch.device("cuda"), dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block0 = model.transformer.h[0]

    with torch.no_grad():
        captured = base._capture_cproj_input(model, fit_rows, torch.device("cuda")).cuda()
        weight = block0.attn.c_proj.weight.detach().float()
        a_factor, b_factor = base._asvd(weight, captured)
        interface = torch.linalg.qr(a_factor[:, :U_RANK].float(), mode="reduced").Q.cuda()
        embedding = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))[:base.VOCAB]
        payload_fold_error = base._payload_exactness(model, embedding)
        all_payload = base._payload_codes(model, interface, embedding)
        write_samples = []
        for start in range(0, len(fit_rows), DOC_BATCH):
            tokens = fit_rows[start:start + DOC_BATCH, :-1].cuda()
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
            attention0, _ = block0.attn(F.rms_norm(token_base, (D,)), None)
            write_samples.append(attention0.float()[:, POSITIONS].reshape(-1, D))
        write_samples = torch.cat(write_samples)
        sigma = torch.sqrt((write_samples @ interface).double().square().mean(0)).float()
        fit_gram, _normalizers, fit_live = base._response_metric(
            model, fit_rows, interface, sigma, None, torch.device("cuda"))
        metric_factor, metric_eigenvalues = base._metric_factor(fit_gram)
        parent_gram = torch.tensor(
            json.loads((BQ / "attention0_ov_downstream_codebook_results.json").read_text())
            ["response_metrics"]["task"]["fit_gram"], device="cuda")
        gram_relative_error = float(
            (fit_gram - parent_gram).double().norm() / parent_gram.double().norm())
        fit_edges = r424._collect_edges(model, fit_rows, rope_tables, apply_rot)
        select_edges = r424._collect_edges(model, select_rows, rope_tables, apply_rot)
        fit_values = _response_rows(fit_edges, all_payload)
        select_values = _response_rows(select_edges, all_payload)

    permutation_generator = torch.Generator(device="cpu").manual_seed(444_999)
    permutation = torch.randperm(len(fit_values), generator=permutation_generator)
    permuted_sources = fit_edges["source"][permutation.to("cuda")]
    permuted_values = _response_rows(fit_edges, all_payload, permuted_sources)
    real_pool, real_pool_indices = _pool(fit_values, 444_2048)
    perm_pool, perm_pool_indices = _pool(permuted_values, 444_2048)

    u444, u444_fit = _fit(fit_values, real_pool, metric_factor, 444, False)
    u445, u445_fit = _fit(fit_values, real_pool, metric_factor, 445, False)
    a444, a444_fit = _fit(fit_values, real_pool, metric_factor, 444, True)
    a445, a445_fit = _fit(fit_values, real_pool, metric_factor, 445, True)
    p444, p444_fit = _fit(fit_values, perm_pool, metric_factor, 444, True)

    half_models = []
    half_fits = []
    half_pool_indices = []
    for low, high, seed in ((0, 48, 444_048), (48, 96, 444_096)):
        mask = (fit_edges["document"] >= low) & (fit_edges["document"] < high)
        values = fit_values[mask]
        pool_values, pool_indices = _pool(values, seed + 2048)
        fitted, fit_info = _fit(values, pool_values, metric_factor, seed, True)
        half_models.append(fitted)
        half_fits.append(fit_info)
        half_pool_indices.append(pool_indices)

    selected_models = {"U32": u444, "A32": a444, "P32": p444}
    select_errors = {
        arm: _exact_loss(sae, select_values, metric_factor)
        for arm, sae in selected_models.items()}
    restart_stability = {
        "U32": _matched_cosine(u444, u445, metric_factor),
        "A32": _matched_cosine(a444, a445, metric_factor),
    }
    half_stability = _matched_cosine(half_models[0], half_models[1], metric_factor)
    with torch.no_grad():
        transport = _consumer_transport(
            model, select_rows, interface, all_payload, selected_models, r424, base,
            rope_tables, apply_rot, scoring)

    training = {
        "U32_seed444": u444_fit, "U32_seed445": u445_fit,
        "A32_seed444": a444_fit, "A32_seed445": a445_fit,
        "P32_seed444": p444_fit,
        "A32_half0": half_fits[0], "A32_half1": half_fits[1],
    }
    gradients_live = all(
        all(value > 0 for value in info["gradient_max"].values())
        for info in training.values())
    losses_decrease = all(info["loss_decrease_fraction"] > 0 for info in training.values())
    certificates_valid = all(
        info["certificate"] is None or (
            info["certificate"]["weight_min"] >= 0
            and info["certificate"]["weight_sum_max_abs_error"] <= 1e-6
            and info["certificate"]["atom_replay_max_abs"] <= 1e-6)
        for info in training.values())

    artifact = {
        "schema": "attention0_response_archetypal_sae_bundle_v1",
        "parent_r424_result_sha256": _sha256(R424_RESULT),
        "models": {
            "U32_seed444": _serialize_model(u444, real_pool_indices),
            "U32_seed445": _serialize_model(u445, real_pool_indices),
            "A32_seed444": _serialize_model(a444, real_pool_indices),
            "A32_seed445": _serialize_model(a445, real_pool_indices),
            "P32_seed444": _serialize_model(p444, perm_pool_indices),
            "A32_half0": _serialize_model(half_models[0], half_pool_indices[0]),
            "A32_half1": _serialize_model(half_models[1], half_pool_indices[1]),
        },
        "metric_factor": metric_factor.half().cpu(),
        "metric_eigenvalues": metric_eigenvalues.cpu(),
        "permuted_source_sha256": _tensor_sha256(permuted_sources),
        "screen_values": SCREEN_VALUES,
        "screen_bytes": SCREEN_BYTES,
        "native_generators_retained": True,
    }
    torch.save(artifact, BUNDLE)

    instrument_checks = {
        "fit_hash": fit_hash == receipt["entries"]["FIT"]["tensor_sha256"],
        "select_hash": select_hash == receipt["entries"]["SELECT"]["tensor_sha256"],
        "role_disjoint": not torch.equal(fit_rows, select_rows),
        "edge_count_fit": len(fit_values) == 185_760,
        "edge_count_select": len(select_values) == 185_760,
        "response_gram_reproduction": gram_relative_error <= 1e-6,
        "payload_fold": payload_fold_error <= 1e-10,
        "native_edge_sum_identity": transport["native_edge_sum_relative_squared_error"] <= 1e-10,
        "convex_certificates": certificates_valid,
        "gradients_live": gradients_live,
        "losses_decrease": losses_decrease,
        "price": SCREEN_VALUES == 1056 and SCREEN_BYTES == 4224,
        "final_unopened": not bool(parent["FINAL_opened"]),
        "fit_consumer_live": max(fit_live.values()) > 0,
    }
    pred_a = all(instrument_checks.values())
    pred_b = (
        select_errors["A32"] <= .15
        and select_errors["A32"] <= .85 * select_errors["P32"]
        and select_errors["U32"] <= .12)
    pred_c = (
        restart_stability["A32"]["median"] >= .70
        and restart_stability["A32"]["median"]
            >= restart_stability["U32"]["median"] + .15
        and half_stability["median"] >= .60)
    a_mean = transport["mean_consumer_r2"]["A32"]
    u_mean = transport["mean_consumer_r2"]["U32"]
    a_ce = transport["ce"]["A32"]["damage"]
    u_ce = transport["ce"]["U32"]["damage"]
    pred_d = (
        transport["routed_u16_r2"]["A32"] >= .90
        and min(transport["consumer_r2"]["A32"].values()) >= .80
        and a_ce <= .005
        and a_mean >= u_mean - .10
        and a_ce <= u_ce + .003)
    strong_null = (
        not pred_a
        or select_errors["A32"] >= .98 * select_errors["P32"]
        or restart_stability["A32"]["median"] <= restart_stability["U32"]["median"]
        or transport["routed_u16_r2"]["A32"] < .60
        or a_ce > .020)
    result = {
        "status": "complete" if pred_a else "instrument_invalid",
        "rung": 444,
        "claim_level": "response_metric_identification_screen",
        "definition": "one realized edge -> sum_h(score1*score2*U16 payload), downstream-response Gram loss",
        "documents": {
            "FIT_sha256": fit_hash, "SELECT_sha256": select_hash,
            "FIT_rows": len(fit_rows), "SELECT_rows": len(select_rows),
            "FINAL_opened": False,
        },
        "instrument": {
            "checks": instrument_checks,
            "response_gram_relative_error": gram_relative_error,
            "payload_fold_max_abs": payload_fold_error,
            "native_edge_sum_relative_squared_error": transport["native_edge_sum_relative_squared_error"],
            "fit_consumer_live_max_abs": max(fit_live.values()),
            "permuted_source_sha256": _tensor_sha256(permuted_sources),
        },
        "config": {
            "atoms": N_ATOMS, "topk": TOPK, "support_pool": SUPPORT_POOL,
            "steps": TRAIN_STEPS, "batch": TRAIN_BATCH, "lr": TRAIN_LR,
            "binding_seed": 444,
        },
        "training": training,
        "select_response_metric_relative_squared_error": select_errors,
        "restart_metric_atom_cosine": restart_stability,
        "fit_half_metric_atom_cosine": half_stability,
        "select_transport": transport,
        "literal_screen_values": SCREEN_VALUES,
        "literal_screen_bytes": SCREEN_BYTES,
        "native_generators_retained": True,
        "net_model_saving_values": 0,
        "bundle": {"path": str(BUNDLE), "file_sha256": _sha256(BUNDLE)},
        'pred_a_instrument_valid': bool(pred_a),
        'pred_b_causal_response_convex_geometry': bool(pred_b),
        'pred_c_archetypal_identifiability': bool(pred_c),
        'pred_d_downstream_consequence': bool(pred_d),
        "strong_null_no_response_archetypal_state": bool(strong_null),
        "compression_or_adoption_licensed": False,
        "next_step": (
            "fresh_response_state_removal_and_composition_family"
            if pred_a and pred_b and pred_c and pred_d and not strong_null
            else "retain_continuous_r424_quotient_close_k32_top4_convex_response_atoms"
        ),
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
