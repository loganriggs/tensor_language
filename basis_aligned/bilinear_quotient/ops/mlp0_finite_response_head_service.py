"""RUNG 417 -- FINITE DOWNSTREAM-RESPONSE HEAD SERVICE AT MLP0.

Rung416 found stable, causal attention0-written geometry but rejected the
claim that its modes are shared by many heads.  This assay changes the object:
two head paths are equivalent only if named downstream consumers respond in
the same way.  Reuse rung402's exact centered-interaction split and record the
finite attention1 and MLP1 write changes when each I_h is (a) added to the
NUMERIC boundary and (b) removed from FULL.

For action and each of the four response cells, form heldout 9x9 head Grams and
head spectra.  Fit on FIT a scalar combination of the eight tail-head vectors
to reconstruct dominant head3, then evaluate on SELECT against raw-action and
seeded within-batch row-permutation controls.  All statistics are streamed;
no activation bank or deployed substitute is stored.

Frozen predictions
------------------
pred_a: exact I closure <=1e-8; native block1 direct replay max error<=2e-5;
    all response RMS>=1e-6; frozen roles are disjoint and auditable.
pred_b: in >=3/4 response cells, rank90 is >=2 below action OR top2 energy is
    >=action+.15, and FIT/SELECT normalized-Gram transport is >=.80.
pred_c: in all four response cells, heldout head3 response R2>=.50, exceeds
    action R2 by >=.20, and exceeds matched shuffled response by >=.30.
pred_d: all response regressions select the same largest tail coefficient,
    every coefficient-vector Spearman>=.75, and response R2 range<=.20.

Strong null: exactness fails; all response ranks are no smaller than action;
every response R2<=.20; any shuffle is within .05; or no response cell beats
both action and shuffle.  Identification only; no compression/adoption/FINAL.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "mlp0_finite_response_head_service_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
D = 1152
N_HEAD = 9
HEAD3 = 3
HEAD_DIM = 128
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
CELLS = ("single_attention1", "single_mlp1", "drop_attention1", "drop_mlp1")


def _spearman(left, right):
    a = torch.as_tensor(left, dtype=torch.float64)
    b = torch.as_tensor(right, dtype=torch.float64)
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = ra.norm() * rb.norm()
    return float((ra @ rb) / denominator) if float(denominator) else 0.0


def _row_digest(rows):
    return hashlib.sha256(rows.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _empty_moment(device):
    return {
        "sum": torch.zeros(N_HEAD, D, dtype=torch.float64, device=device),
        "raw_gram": torch.zeros(N_HEAD, N_HEAD, dtype=torch.float64, device=device),
        "shuffle_raw_gram": torch.zeros(N_HEAD, N_HEAD, dtype=torch.float64, device=device),
        "count": 0,
        "square": torch.zeros(N_HEAD, dtype=torch.float64, device=device),
    }


def _permuted(flat, role_offset, batch_index):
    # Permute complete document-position rows, preserving every 1152-vector.
    result = []
    rows = flat.shape[1]
    for head in range(N_HEAD):
        generator = torch.Generator(device="cpu").manual_seed(
            417_000 + role_offset + batch_index * 100 + head)
        permutation = torch.randperm(rows, generator=generator, device="cpu").to(flat.device)
        result.append(flat[head, permutation])
    return torch.stack(result)


def _accumulate(moment, values, role_offset, batch_index):
    # values: [head,batch,position,width]
    flat = values[:, :, SCORING, :].float().flatten(1, 2)
    moment["sum"] += flat.double().sum(1)
    moment["raw_gram"] += torch.einsum("hpd,kpd->hk", flat, flat).double()
    shuffled = _permuted(flat, role_offset, batch_index)
    moment["shuffle_raw_gram"] += torch.einsum("hpd,kpd->hk", shuffled, shuffled).double()
    moment["square"] += flat.double().square().sum((1, 2))
    moment["count"] += flat.shape[1]
    return flat, shuffled


def _centered(moment, *, shuffled=False):
    means = moment["sum"] / moment["count"]
    raw = moment["shuffle_raw_gram"] if shuffled else moment["raw_gram"]
    gram = raw - moment["count"] * (means @ means.T)
    gram = (gram + gram.T) / 2
    return means, gram


def _spectral_summary(moment):
    means, gram = _centered(moment)
    values = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = float(values.sum())
    fractions = values / max(total, 1e-30)
    rank90 = int(torch.searchsorted(fractions.cumsum(0), .90).item() + 1)
    norm = torch.diag(gram).clamp_min(1e-30).sqrt()
    correlation = gram / (norm[:, None] * norm[None, :])
    return {
        "mean": means,
        "gram": gram,
        "correlation": correlation,
        "eigenvalues": values,
        "rank90": rank90,
        "top2_energy_fraction": float(fractions[:2].sum()),
        "head_rms": torch.sqrt(moment["square"] / (moment["count"] * D)),
    }


def _fit_model(moment, *, shuffled=False):
    means, gram = _centered(moment, shuffled=shuffled)
    tail = [head for head in range(N_HEAD) if head != HEAD3]
    design = gram[tail][:, tail]
    scale = float(torch.trace(design)) / len(tail)
    ridge = max(scale * 1e-10, 1e-20)
    beta = torch.linalg.solve(
        design + ridge * torch.eye(len(tail), dtype=torch.float64, device=design.device),
        gram[tail, HEAD3])
    return {"means": means.float(), "beta": beta.float(), "tail": tail, "ridge": ridge}


def _eval_model(flat, shuffled, model, accum, *, use_shuffle=False):
    predictors = shuffled if use_shuffle else flat
    tail = model["tail"]
    means = model["means"]
    centered = predictors[tail] - means[tail, None, :]
    prediction = means[HEAD3][None, :] + torch.einsum("h,hpd->pd", model["beta"], centered)
    target = flat[HEAD3]
    accum[0] += float((target.double() - prediction.double()).square().sum())
    accum[1] += float((target.double() - means[HEAD3].double()).square().sum())


def _native_block1_from_mlp0(block1, x0, token_base, attention0, mlp0, first_value):
    post0 = token_base + attention0 + mlp0
    mixed1 = block1.lambdas[0] * post0 + block1.lambdas[1] * x0
    attention1, _ = block1.attn(F.rms_norm(mixed1, (D,)), first_value)
    mlp1 = block1.mlp(F.rms_norm(mixed1 + attention1, (D,)))
    return attention1, mlp1


@torch.no_grad()
def _facade_replay(model, tokens, expected_attention0, expected_first_value):
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade

    captured = {}

    def attention(event):
        if event.site == 0:
            return expected_attention0, expected_first_value
        value, bus = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            captured["attention1"] = value.detach()
        return value, bus

    def mlp(event):
        value = event.block.mlp(event.state)
        if event.site == 1:
            captured["mlp1"] = value.detach()
        return value

    facade.forward_with_dispatch(model, tokens, attention, mlp)
    return captured["attention1"], captured["mlp1"]


@torch.no_grad()
def _collect_role(model, rows, device, reference, head_means, epsilon_mean, base, carrier,
                  role_offset, eval_models=None):
    block0 = model.transformer.h[0]
    block1 = model.transformer.h[1]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    moments = {"action": _empty_moment(device), **{cell: _empty_moment(device) for cell in CELLS}}
    evaluations = {}
    if eval_models is not None:
        for name in ("action", *CELLS):
            evaluations[name] = {"real": [0.0, 0.0]}
            if name != "action":
                evaluations[name]["shuffle"] = [0.0, 0.0]
    interaction_num = 0.0
    interaction_den = 0.0
    replay = {"attention1_max_abs": 0.0, "mlp1_max_abs": 0.0}

    for batch_index, start in enumerate(range(0, len(rows), DOCUMENT_BATCH)):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        attention0, first_value, head_writes, epsilon = carrier._attention0_with_heads(
            block0, F.rms_norm(token_base, (D,)), None)
        normalized0 = F.rms_norm(token_base + attention0, (D,))
        native0 = block0.mlp(normalized0)
        _, branches, _, _, _ = carrier._exact_components(
            base, token_base, attention0, normalized0, reference, left, right, down)
        semantic, numerical, _, _ = carrier._split_interaction(
            base, token_base, head_writes, epsilon, reference, head_means,
            epsilon_mean, branches["I"], left, right, down)
        reconstructed = semantic.sum(0) + numerical
        interaction_num += float((reconstructed.double() - branches["I"].double()).square().sum())
        interaction_den += float(branches["I"].double().square().sum())

        numeric0 = native0 - semantic.sum(0).to(native0.dtype)
        full_a1, full_m1 = _native_block1_from_mlp0(
            block1, x0, token_base, attention0, native0, first_value)
        numeric_a1, numeric_m1 = _native_block1_from_mlp0(
            block1, x0, token_base, attention0, numeric0, first_value)
        response = {cell: [] for cell in CELLS}
        for head in range(N_HEAD):
            single0 = native0 - sum(
                (semantic[h] for h in range(N_HEAD) if h != head),
                start=torch.zeros_like(numerical)).to(native0.dtype)
            drop0 = native0 - semantic[head].to(native0.dtype)
            single_a1, single_m1 = _native_block1_from_mlp0(
                block1, x0, token_base, attention0, single0, first_value)
            drop_a1, drop_m1 = _native_block1_from_mlp0(
                block1, x0, token_base, attention0, drop0, first_value)
            response["single_attention1"].append(single_a1.float() - numeric_a1.float())
            response["single_mlp1"].append(single_m1.float() - numeric_m1.float())
            response["drop_attention1"].append(full_a1.float() - drop_a1.float())
            response["drop_mlp1"].append(full_m1.float() - drop_m1.float())

        action_flat, action_shuffle = _accumulate(
            moments["action"], semantic, role_offset, batch_index)
        if eval_models is not None:
            _eval_model(action_flat, action_shuffle, eval_models["action"]["real"],
                        evaluations["action"]["real"])
        for cell in CELLS:
            values = torch.stack(response[cell])
            flat, shuffled = _accumulate(moments[cell], values, role_offset, batch_index)
            if eval_models is not None:
                _eval_model(flat, shuffled, eval_models[cell]["real"],
                            evaluations[cell]["real"])
                _eval_model(flat, shuffled, eval_models[cell]["shuffle"],
                            evaluations[cell]["shuffle"], use_shuffle=True)

        facade_a1, facade_m1 = _facade_replay(model, tokens, attention0, first_value)
        replay["attention1_max_abs"] = max(
            replay["attention1_max_abs"], float((full_a1.float() - facade_a1.float()).abs().max()))
        replay["mlp1_max_abs"] = max(
            replay["mlp1_max_abs"], float((full_m1.float() - facade_m1.float()).abs().max()))

    summaries = {name: _spectral_summary(moment) for name, moment in moments.items()}
    public = {
        name: {
            "rank90": value["rank90"],
            "top2_energy_fraction": value["top2_energy_fraction"],
            "head_rms": value["head_rms"].cpu().tolist(),
            "normalized_head_gram": value["correlation"].cpu().tolist(),
            "head_eigenvalues": value["eigenvalues"].cpu().tolist(),
        }
        for name, value in summaries.items()
    }
    diagnostics = {
        "interaction_sum_relative_mse": interaction_num / max(interaction_den, 1e-30),
        "native_block1_replay": replay,
        "minimum_response_head_rms": min(
            min(public[cell]["head_rms"]) for cell in CELLS),
    }
    evaluated = None
    if eval_models is not None:
        evaluated = {}
        for name, controls in evaluations.items():
            evaluated[name] = {}
            for control, (sse, sst) in controls.items():
                evaluated[name][control] = {
                    "sse": sse, "sst": sst, "r2": 1.0 - sse / max(sst, 1e-30)}
    return moments, summaries, public, diagnostics, evaluated


def _gram_transport(fit, select):
    indices = torch.triu_indices(N_HEAD, N_HEAD, offset=1)
    a = fit[indices[0], indices[1]].double()
    b = select[indices[0], indices[1]].double()
    a -= a.mean()
    b -= b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert N_HEAD * HEAD_DIM == D and HEAD3 == 3 and len(CELLS) == 4
        assert SCORING.stop - SCORING.start == 192
        print("MLP0 FINITE RESPONSE HEAD SERVICE | dry run: 4 cells, controls frozen")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import mlp0_centered_context_anova_factorial as base
    import mlp0_centered_interaction_head_carriers as carrier

    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    reference = base._reference_moments(model, fit_rows, device)
    head_means, epsilon_mean, reference_diag = carrier._head_reference(
        model, fit_rows, reference, device)

    fit_moments, fit_summaries, fit_public, fit_diag, _ = _collect_role(
        model, fit_rows, device, reference, head_means, epsilon_mean, base, carrier,
        role_offset=0)
    models = {}
    for name in ("action", *CELLS):
        models[name] = {"real": _fit_model(fit_moments[name])}
        if name != "action":
            models[name]["shuffle"] = _fit_model(fit_moments[name], shuffled=True)
    _, select_summaries, select_public, select_diag, select_eval = _collect_role(
        model, select_rows, device, reference, head_means, epsilon_mean, base, carrier,
        role_offset=10_000, eval_models=models)

    action_r2 = select_eval["action"]["real"]["r2"]
    transport = {
        name: _gram_transport(
            fit_summaries[name]["correlation"], select_summaries[name]["correlation"])
        for name in ("action", *CELLS)
    }
    coefficients = {
        cell: models[cell]["real"]["beta"].cpu().tolist() for cell in CELLS}
    coefficient_spearman = {}
    for i, left_cell in enumerate(CELLS):
        for right_cell in CELLS[i + 1:]:
            coefficient_spearman[f"{left_cell}__{right_cell}"] = _spearman(
                coefficients[left_cell], coefficients[right_cell])

    cell_tests = {}
    for cell in CELLS:
        response = select_public[cell]
        action = select_public["action"]
        real_r2 = select_eval[cell]["real"]["r2"]
        shuffle_r2 = select_eval[cell]["shuffle"]["r2"]
        compressed = (
            response["rank90"] <= action["rank90"] - 2
            or response["top2_energy_fraction"] >= action["top2_energy_fraction"] + .15)
        cell_tests[cell] = {
            "compressed_vs_action": compressed,
            "gram_transport": transport[cell],
            "head3_response_r2": real_r2,
            "head3_action_r2": action_r2,
            "head3_shuffle_r2": shuffle_r2,
            "margin_over_action": real_r2 - action_r2,
            "margin_over_shuffle": real_r2 - shuffle_r2,
        }

    row_disjoint = _row_digest(fit_rows) != _row_digest(select_rows)
    pred_a = (
        row_disjoint
        and fit_diag["interaction_sum_relative_mse"] <= 1e-8
        and select_diag["interaction_sum_relative_mse"] <= 1e-8
        and max(fit_diag["native_block1_replay"].values()) <= 2e-5
        and max(select_diag["native_block1_replay"].values()) <= 2e-5
        and min(fit_diag["minimum_response_head_rms"],
                select_diag["minimum_response_head_rms"]) >= 1e-6)
    pred_b = sum(
        test["compressed_vs_action"] and test["gram_transport"] >= .80
        for test in cell_tests.values()) >= 3
    pred_c = all(
        test["head3_response_r2"] >= .50
        and test["margin_over_action"] >= .20
        and test["margin_over_shuffle"] >= .30
        for test in cell_tests.values())
    top_tail = [int(torch.tensor(coefficients[cell]).abs().argmax()) for cell in CELLS]
    # Indices above are within the frozen tail ordering [0,1,2,4,5,6,7,8].
    tail = models[CELLS[0]]["real"]["tail"]
    top_tail_heads = [tail[index] for index in top_tail]
    response_r2 = [cell_tests[cell]["head3_response_r2"] for cell in CELLS]
    pred_d = (
        len(set(top_tail_heads)) == 1
        and min(coefficient_spearman.values()) >= .75
        and max(response_r2) - min(response_r2) <= .20)
    all_response_ranks_not_smaller = all(
        select_public[cell]["rank90"] >= select_public["action"]["rank90"]
        for cell in CELLS)
    every_response_low = all(value <= .20 for value in response_r2)
    any_shuffle_close = any(
        test["margin_over_shuffle"] <= .05 for test in cell_tests.values())
    no_cell_beats_both = not any(
        test["head3_response_r2"] > action_r2
        and test["head3_response_r2"] > test["head3_shuffle_r2"]
        for test in cell_tests.values())
    strong_null = (
        not pred_a or all_response_ranks_not_smaller or every_response_low
        or any_shuffle_close or no_cell_beats_both)

    if pred_a and pred_b and pred_c and pred_d and not strong_null:
        next_step = "shared_response_basis_dual_reader_factorization_and_physical_group_validation"
    elif pred_a and pred_b and not strong_null:
        next_step = "subhead_response_directions_without_head_equivalence_claim"
    elif pred_a:
        next_step = "cross_head_double_qk_shared_half_factorization_then_attention1_response_test"
    else:
        next_step = "instrument_repair_only"

    result = {
        "status": "mlp0_finite_response_head_service_complete",
        "rung": 417,
        "claim_level": "finite_distribution_mechanism_identification_not_compression",
        "definition": {
            "action": "exact centered MLP0 token-by-context term I_h",
            "single_response": "consumer(NUMERIC+I_h)-consumer(NUMERIC)",
            "drop_response": "consumer(FULL)-consumer(FULL-I_h)",
            "consumer": "native 1152-dimensional attention1 or MLP1 write",
            "regression": "FIT scalar combination of other eight centered head vectors plus output-coordinate mean intercept; SELECT evaluation",
            "shuffle": "independent seeded within-batch document-position row permutation per predictor head",
        },
        "documents": {
            "FIT": len(fit_rows), "SELECT": len(select_rows), "FINAL_opened": 0,
            "FIT_sha256": _row_digest(fit_rows), "SELECT_sha256": _row_digest(select_rows),
            "roles_disjoint": row_disjoint,
        },
        "positions_per_document": SCORING.stop - SCORING.start,
        "diagnostic_price": {
            "native_parameters_reused": "all",
            "stored_activation_bank_values": 0,
            "stored_gram_values": 5 * N_HEAD * N_HEAD,
            "stored_regression_coefficients": len(CELLS) * 2 * (N_HEAD - 1),
            "compression_or_adoption_claim": False,
        },
        "permutation_seed_rule": "417000 + role_offset(0 FIT,10000 SELECT) + batch_index*100 + head",
        "reference_diagnostics": reference_diag,
        "FIT": fit_public,
        "SELECT": select_public,
        "transport": transport,
        "head3_regression": {
            "action_SELECT_r2": action_r2,
            "response_cells": cell_tests,
            "real_coefficients_tail_order_0_1_2_4_5_6_7_8": coefficients,
            "top_absolute_tail_head": dict(zip(CELLS, top_tail_heads)),
            "coefficient_pair_spearman": coefficient_spearman,
        },
        "diagnostics": {"FIT": fit_diag, "SELECT": select_diag},
        'pred_a_exact_live_disjoint': bool(pred_a),
        'pred_b_downstream_response_shared_head_service': bool(pred_b),
        'pred_c_redundant_head3_producer': bool(pred_c),
        'pred_d_equivalence_survives_consumer_and_background': bool(pred_d),
        "null_no_finite_block1_head_service": bool(strong_null),
        "next_step": next_step,
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 FINITE RESPONSE HEAD SERVICE DONE", flush=True)


if __name__ == "__main__":
    main()
