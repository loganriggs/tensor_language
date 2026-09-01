"""RUNG 433 -- ATTENTION0 Q/K NORMALIZER NECESSITY.

Keep native numerators and change only the 36 token/head RMS denominators to
FIT-token constants.  Exact-table replay is the instrument.  Separately test
whether the complete log-denominator table has a stable low-dimensional token
vocabulary above independently token-permuted controls.

Frozen registration:
  polynomial_causal/ATTENTION0_QK_NORMALIZER_NECESSITY_PREREGISTRATION.md
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

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
OUT = BQ / "attention0_qk_normalizer_necessity_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
EDGE_PATH = OPS / "attention0_realized_edge_block_term.py"
OV_BASE = OPS / "attention0_ov_downstream_codebook.py"
PREREG = POLY / "ATTENTION0_QK_NORMALIZER_NECESSITY_PREREGISTRATION.md"

VOCAB = 50_257
D = 1_152
N_HEAD = 9
HD = 128
U_RANK = 16
POSITIONS = tuple(range(16, 241, 16))
DOC_BATCH = 4
MAPS = ("c_q", "c_k", "c_q2", "c_k2")
CONSUMERS = ("mlp0", "q1", "k1", "q2", "k2", "fresh_v")
RANKS = (1, 4, 8, 16, 36)
PERMUTATION_SEEDS = tuple(433_100 + index for index in range(64))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _digest_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    return float(singular.square().mean())


@torch.no_grad()
def _denominator_tables(model) -> tuple[torch.Tensor, torch.Tensor, float, dict]:
    block0 = model.transformer.h[0]
    embedding = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))[:VOCAB]
    state = F.rms_norm((block0.lambdas[0] + block0.lambdas[1]) * embedding, (D,))
    tables = []
    reconstruction_errors = {}
    epsilon = float(torch.finfo(state.dtype).eps)
    for name in MAPS:
        raw = getattr(block0.attn, name)(state).view(VOCAB, N_HEAD, HD).float()
        denominator = (raw.square().mean(-1) + epsilon).sqrt()
        reconstructed = raw / denominator[..., None]
        native = F.rms_norm(raw, (HD,))
        reconstruction_errors[name] = {
            "max_absolute": float((reconstructed - native).abs().max()),
            "relative_squared": float(
                (reconstructed.double() - native.double()).square().sum()
                / native.double().square().sum().clamp_min(1e-30)),
        }
        tables.append(denominator)
    return state, torch.stack(tables), epsilon, reconstruction_errors


@torch.no_grad()
def _scores(block0, state: torch.Tensor, tokens: torch.Tensor,
            denominators: torch.Tensor, constants: torch.Tensor | None,
            rope_tables, apply_rot) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length, _ = state.shape
    cos, sin = rope_tables(length, HD, state.device, torch.float32, "bf16")
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    factors = []
    for map_index, name in enumerate(MAPS):
        raw = getattr(block0.attn, name)(state).view(batch, length, N_HEAD, HD).float()
        if constants is None:
            scale = denominators[map_index, tokens].reciprocal()
        else:
            scale = constants[map_index].reciprocal()[None, None, :]
        factors.append(apply_rot(raw * scale[..., None], cos, sin))
    q1, k1, q2, k2 = factors
    score1 = torch.einsum("bqhd,bkhd->bhqk", q1, k1) / HD
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HD
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    return score1.masked_fill(~mask, 0), score2.masked_fill(~mask, 0)


@torch.no_grad()
def _attention_from_scores(block0, state: torch.Tensor,
                           score1: torch.Tensor, score2: torch.Tensor) -> torch.Tensor:
    value = block0.attn.c_v(state).view(*state.shape[:2], N_HEAD, HD)
    mixed = torch.einsum("bhqk,bkhd->bqhd", score1 * score2, value).reshape(*state.shape)
    return block0.attn.c_proj(mixed)


@torch.no_grad()
def _table_structure(denominators: torch.Tensor) -> dict:
    # map-major/head-minor is the registered lexicographic ordering.
    matrix = denominators.permute(1, 0, 2).reshape(VOCAB, len(MAPS) * N_HEAD).log().double()
    ids = torch.arange(VOCAB, device=matrix.device)
    fit = ids.remainder(5) != 4
    select = ~fit
    mean = matrix[fit].mean(0, keepdim=True)
    fit_centered = matrix[fit] - mean
    select_centered = matrix[select] - mean
    _u, singular, vh = torch.linalg.svd(fit_centered, full_matrices=False)
    explained = {}
    select_energy = select_centered.square().sum().clamp_min(1e-30)
    for rank in RANKS:
        basis = vh[:rank].T
        residual = select_centered - (select_centered @ basis) @ basis.T
        explained[str(rank)] = 1 - float(residual.square().sum() / select_energy)

    even = torch.arange(matrix.shape[1], device=matrix.device).remainder(2) == 0
    odd = ~even
    ua = torch.linalg.svd(fit_centered[:, even], full_matrices=False).U[:, :8]
    ub = torch.linalg.svd(fit_centered[:, odd], full_matrices=False).U[:, :8]
    real_overlap = _overlap(ua, ub)

    controls = []
    nfit = int(fit.sum())
    left = fit_centered[:, even]
    right = fit_centered[:, odd]
    for seed in PERMUTATION_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        perm_left = torch.empty_like(left)
        perm_right = torch.empty_like(right)
        for column in range(left.shape[1]):
            order = torch.randperm(nfit, generator=generator, device="cpu").to(matrix.device)
            perm_left[:, column] = left[order, column]
        for column in range(right.shape[1]):
            order = torch.randperm(nfit, generator=generator, device="cpu").to(matrix.device)
            perm_right[:, column] = right[order, column]
        pua = torch.linalg.svd(perm_left, full_matrices=False).U[:, :8]
        pub = torch.linalg.svd(perm_right, full_matrices=False).U[:, :8]
        controls.append(_overlap(pua, pub))
    control_tensor = torch.tensor(controls, dtype=torch.float64)
    control_99 = float(torch.quantile(control_tensor, .99, interpolation="linear"))

    select_values = denominators[:, select].double()
    coefficient_of_variation = select_values.std(1, correction=0) / select_values.mean(1).clamp_min(1e-30)
    family_medians = {
        MAPS[index]: float(coefficient_of_variation[index].median())
        for index in range(len(MAPS))
    }
    return {
        "token_fit_count": int(fit.sum()),
        "token_select_count": int(select.sum()),
        "token_fit_sha256": _digest_tensor(ids[fit]),
        "token_select_sha256": _digest_tensor(ids[select]),
        "fit_singular_values": [float(value) for value in singular],
        "select_explained_by_fit_map_basis": explained,
        "map_half_token_subspace_overlap_rank8": real_overlap,
        "permuted_overlap_values": controls,
        "permuted_overlap_99th_percentile": control_99,
        "overlap_excess_over_permuted_99": real_overlap - control_99,
        "select_denominator_cv_by_map_head": coefficient_of_variation.cpu().tolist(),
        "select_denominator_cv_family_median": family_medians,
    }


@torch.no_grad()
def _physical(model, rows: torch.Tensor, state_table: torch.Tensor,
              denominators: torch.Tensor,
              constants: torch.Tensor, interface: torch.Tensor,
              rope_tables, apply_rot, base, edge_mod, scoring) -> dict:
    arms = ("EXACT_TABLE", "CONSTANT")
    block0, block1 = model.transformer.h[:2]
    score_sse = {arm: [0.0, 0.0] for arm in arms}
    score_den = [0.0, 0.0]
    product_sse = {arm: 0.0 for arm in arms}
    product_den = 0.0
    write_sse = {arm: 0.0 for arm in arms}
    write_den = 0.0
    routed_sse = {arm: 0.0 for arm in arms}
    routed_den = 0.0
    consumer_sse = {arm: {name: 0.0 for name in CONSUMERS} for arm in arms}
    consumer_den = {name: 0.0 for name in CONSUMERS}
    ce = {"NATIVE": [], **{arm: [] for arm in arms}}
    logit_max = {arm: 0.0 for arm in arms}
    state_max = 0.0
    direct_rescale_sse = 0.0
    direct_rescale_den = 0.0
    denominator_live_abs = 0.0
    denominator_live_rel_num = 0.0
    denominator_live_rel_den = 0.0
    ratio_max_deviation = 0.0

    for start in range(0, len(rows), DOC_BATCH):
        batch = rows[start:start + DOC_BATCH].to("cuda")
        tokens = batch[:, :-1]
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        state = F.rms_norm(token_base, (D,))
        state_max = max(state_max, float((state - state_table[tokens]).abs().max()))
        native_attention, first_value = block0.attn(state, None)
        native_score1, native_score2 = edge_mod._score_halves(
            block0, state, rope_tables, apply_rot)
        exact_score1, exact_score2 = _scores(
            block0, state, tokens, denominators, None, rope_tables, apply_rot)
        constant_score1, constant_score2 = _scores(
            block0, state, tokens, denominators, constants, rope_tables, apply_rot)
        scores = {
            "EXACT_TABLE": (exact_score1, exact_score2),
            "CONSTANT": (constant_score1, constant_score2),
        }

        # Live denominator check from the current sequence projections.
        for map_index, name in enumerate(MAPS):
            raw = getattr(block0.attn, name)(state).view(*state.shape[:2], N_HEAD, HD).float()
            live = (raw.square().mean(-1) + torch.finfo(raw.dtype).eps).sqrt()
            table = denominators[map_index, tokens]
            delta = live.double() - table.double()
            denominator_live_abs = max(denominator_live_abs, float(delta.abs().max()))
            denominator_live_rel_num += float(delta.square().sum())
            denominator_live_rel_den += float(live.double().square().sum())
            ratio = table / constants[map_index][None, None, :]
            ratio_max_deviation = max(ratio_max_deviation, float((ratio - 1).abs().max()))

        # Registered identity: direct constant score equals native score times norm ratios.
        q1r = denominators[0, tokens] / constants[0][None, None, :]
        k1r = denominators[1, tokens] / constants[1][None, None, :]
        q2r = denominators[2, tokens] / constants[2][None, None, :]
        k2r = denominators[3, tokens] / constants[3][None, None, :]
        rescaled1 = native_score1 * q1r.permute(0, 2, 1)[..., :, None] * k1r.permute(0, 2, 1)[:, :, None, :]
        rescaled2 = native_score2 * q2r.permute(0, 2, 1)[..., :, None] * k2r.permute(0, 2, 1)[:, :, None, :]
        direct_rescale_sse += float((rescaled1.double() - constant_score1.double()).square().sum())
        direct_rescale_sse += float((rescaled2.double() - constant_score2.double()).square().sum())
        direct_rescale_den += float(constant_score1.double().square().sum())
        direct_rescale_den += float(constant_score2.double().square().sum())

        native_product = native_score1 * native_score2
        for branch, native_score in enumerate((native_score1, native_score2)):
            score_den[branch] += float(native_score.double().square().sum())
        product_den += float(native_product.double().square().sum())
        write_den += float(native_attention[:, POSITIONS].double().square().sum())
        native_u = native_attention.float() @ interface.float()
        routed_den += float(native_u[:, POSITIONS].double().square().sum())
        zero_attention = torch.zeros_like(native_attention)
        native_fields = base._consumer_fields(block0, block1, x0, token_base, native_attention)
        zero_fields = base._consumer_fields(block0, block1, x0, token_base, zero_attention)
        for name in CONSUMERS:
            target = (native_fields[name].float().flatten(2)[:, POSITIONS]
                      - zero_fields[name].float().flatten(2)[:, POSITIONS])
            consumer_den[name] += float(target.double().square().sum())
        native_logits = edge_mod._suffix_logits(
            model, tokens, x0, token_base, native_attention, first_value)
        for row in range(len(batch)):
            ce["NATIVE"].append(scoring.document_mean_ce(native_logits[row], batch[row, 1:]))

        for arm, (score1, score2) in scores.items():
            changed = _attention_from_scores(block0, state, score1, score2)
            for branch, (score, native_score) in enumerate(
                    zip((score1, score2), (native_score1, native_score2))):
                score_sse[arm][branch] += float(
                    (score.double() - native_score.double()).square().sum())
            product_sse[arm] += float(
                ((score1.double() * score2.double()) - native_product.double()).square().sum())
            write_sse[arm] += float(
                (changed[:, POSITIONS].double()
                 - native_attention[:, POSITIONS].double()).square().sum())
            changed_u = changed.float() @ interface.float()
            routed_sse[arm] += float(
                (changed_u[:, POSITIONS].double() - native_u[:, POSITIONS].double()).square().sum())
            changed_fields = base._consumer_fields(block0, block1, x0, token_base, changed)
            for name in CONSUMERS:
                error = (changed_fields[name].float().flatten(2)[:, POSITIONS]
                         - native_fields[name].float().flatten(2)[:, POSITIONS])
                consumer_sse[arm][name] += float(error.double().square().sum())
            logits = edge_mod._suffix_logits(
                model, tokens, x0, token_base, changed, first_value)
            logit_max[arm] = max(logit_max[arm], float((logits - native_logits).abs().max()))
            for row in range(len(batch)):
                ce[arm].append(scoring.document_mean_ce(logits[row], batch[row, 1:]))

    ce_tensors = {name: torch.stack(values).double().cpu() for name, values in ce.items()}
    ce_report = {}
    for name, values in ce_tensors.items():
        ce_report[name] = {
            "mean": float(values.mean()),
            "damage": float(values.mean() - ce_tensors["NATIVE"].mean()),
            "wave_damage": [
                float(values[:48].mean() - ce_tensors["NATIVE"][:48].mean()),
                float(values[48:].mean() - ce_tensors["NATIVE"][48:].mean()),
            ],
        }
    return {
        "state_table_max_absolute_error": state_max,
        "live_denominator_max_absolute_error": denominator_live_abs,
        "live_denominator_relative_squared_error": (
            denominator_live_rel_num / max(denominator_live_rel_den, 1e-30)),
        "constant_ratio_max_deviation_from_one": ratio_max_deviation,
        "direct_constant_vs_rescaled_native_relative_squared_error": (
            direct_rescale_sse / max(direct_rescale_den, 1e-30)),
        "branch_score_relative_squared_error": {
            arm: [score_sse[arm][index] / max(score_den[index], 1e-30) for index in range(2)]
            for arm in arms},
        "product_relative_squared_error": {
            arm: product_sse[arm] / max(product_den, 1e-30) for arm in arms},
        "full_write_relative_squared_error": {
            arm: write_sse[arm] / max(write_den, 1e-30) for arm in arms},
        "routed_u16_r2": {
            arm: 1 - routed_sse[arm] / max(routed_den, 1e-30) for arm in arms},
        "consumer_r2": {
            arm: {name: 1 - consumer_sse[arm][name] / max(consumer_den[name], 1e-30)
                  for name in CONSUMERS} for arm in arms},
        "max_absolute_logit_difference": logit_max,
        "ce": ce_report,
    }


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert VOCAB == 50_257 and D == 1_152 and N_HEAD == 9 and HD == 128
        assert len(MAPS) * N_HEAD == 36 and len(PERMUTATION_SEEDS) == 64
        assert ROWS_RECEIPT.exists() and EDGE_PATH.exists() and OV_BASE.exists() and PREREG.exists()
        print("ATTENTION0 NORMALIZER NECESSITY | dry run: exact table, constant, spectrum, controls")
        return

    started = time.time()
    sys.path[:0] = [str(POLY), str(OPS), str(QK), str(BQ)]
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring
    from tier2_model import apply_rot, rope_tables
    import bilin18_observed_model_facade as facade

    edge_mod = _load_module("r433_edge", EDGE_PATH)
    base = _load_module("r433_ov", OV_BASE)
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    fit_hash = rows_parent.rows_life.base.tensor_sha256(fit_rows)
    select_hash = rows_parent.rows_life.base.tensor_sha256(select_rows)
    assert fit_hash == receipt["entries"]["FIT"]["tensor_sha256"]
    assert select_hash == receipt["entries"]["SELECT"]["tensor_sha256"]
    assert len(select_rows) == 96

    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    state_table, denominators, epsilon, reconstruction = _denominator_tables(model)
    ids = torch.arange(VOCAB, device="cuda")
    fit_mask = ids.remainder(5) != 4
    constants = denominators[:, fit_mask].double().square().mean(1).sqrt().float()
    structure = _table_structure(denominators)

    block0 = model.transformer.h[0]
    captured_cproj = base._capture_cproj_input(model, fit_rows, torch.device("cuda")).to("cuda")
    a_factor, _b_factor = base._asvd(block0.attn.c_proj.weight.detach().float(), captured_cproj)
    interface = torch.linalg.qr(a_factor[:, :U_RANK].float(), mode="reduced").Q
    physical = _physical(
        model, select_rows, state_table, denominators, constants, interface,
        rope_tables, apply_rot, base, edge_mod, scoring)

    max_table_rel = max(value["relative_squared"] for value in reconstruction.values())
    exact_ce = physical["ce"]["EXACT_TABLE"]
    constant_ce = physical["ce"]["CONSTANT"]
    exact_product = physical["product_relative_squared_error"]["EXACT_TABLE"]
    constant_product = physical["product_relative_squared_error"]["CONSTANT"]
    exact_write = physical["full_write_relative_squared_error"]["EXACT_TABLE"]
    constant_write = physical["full_write_relative_squared_error"]["CONSTANT"]
    constant_branch = physical["branch_score_relative_squared_error"]["CONSTANT"]
    family_cv = structure["select_denominator_cv_family_median"]

    pred_a = bool(
        physical["state_table_max_absolute_error"] <= 1e-6
        and max_table_rel <= 1e-6
        and physical["live_denominator_relative_squared_error"] <= 1e-6
        and physical["direct_constant_vs_rescaled_native_relative_squared_error"] <= 1e-10
        and exact_product <= 1e-10
        and exact_write <= 1e-10
        and physical["max_absolute_logit_difference"]["EXACT_TABLE"] <= 2e-5
        and abs(exact_ce["damage"]) <= 1e-6
        and physical["constant_ratio_max_deviation_from_one"] >= 1e-3)
    pred_b = bool(
        constant_product >= .20
        and constant_write >= .10
        and constant_ce["damage"] >= .010
        and all(value > 0 for value in constant_ce["wave_damage"]))
    pred_c = bool(
        structure["select_explained_by_fit_map_basis"]["8"] >= .90
        and structure["map_half_token_subspace_overlap_rank8"] >= .60
        and structure["overlap_excess_over_permuted_99"] >= .20)
    pred_d = bool(
        sum(value >= .02 for value in family_cv.values()) >= 3
        and all(value >= .15 for value in constant_branch))
    strong_null = bool(
        (constant_product <= .05 and constant_ce["damage"] <= .002)
        or (structure["select_explained_by_fit_map_basis"]["8"] <= .60
            and structure["overlap_excess_over_permuted_99"] <= .05))

    output = {
        "schema": "attention0_qk_normalizer_necessity_v1",
        "status": "complete" if pred_a else "instrument_invalid",
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint),
        "dimensions": {"vocab": VOCAB, "residual": D, "heads": N_HEAD,
                       "head_width": HD, "normalizer_functions": 36},
        "epsilon": epsilon,
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows),
                      "FIT_sha256": fit_hash, "SELECT_sha256": select_hash},
        "token_roles": {"FIT": structure["token_fit_count"],
                        "SELECT": structure["token_select_count"],
                        "FIT_sha256": structure["token_fit_sha256"],
                        "SELECT_sha256": structure["token_select_sha256"]},
        "prices_scalar_values": {
            "native_layer0_qk_maps": 4 * D * D,
            "exact_denominator_table_only": VOCAB * 36,
            "constants_only": 36,
            "descriptive_rank8_log_table_only": VOCAB * 8 + 36 * 8 + 36,
        },
        "table_reconstruction": reconstruction,
        "table_structure": structure,
        "physical": physical,
        "predictions": {
            'pred_a_exact_table_instrument': pred_a,
            'pred_b_token_dependent_denominators_causally_material': pred_b,
            'pred_c_shared_stable_token_vocabulary': pred_c,
            'pred_d_not_one_exceptional_map': pred_d,
            "strong_null": strong_null,
        },
        "scope": "diagnostic only; native numerators retained; no compression or semantics",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "predictions": output["predictions"],
        "constant_product_rel_sq": constant_product,
        "constant_write_rel_sq": constant_write,
        "constant_ce": constant_ce,
        "rank8_select_explained": structure["select_explained_by_fit_map_basis"]["8"],
        "split_overlap": structure["map_half_token_subspace_overlap_rank8"],
        "permuted_99": structure["permuted_overlap_99th_percentile"],
        "output": str(OUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
