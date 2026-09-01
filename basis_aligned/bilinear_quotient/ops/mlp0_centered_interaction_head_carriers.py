"""RUNG 402 -- EXACT HEAD CARRIERS OF CENTERED MLP0 INTERACTION.

Rung401 identified centered token-by-context interaction I as MLP0's largest
causal role.  Attention0's write is a sum of nine output-projected head writes,
and I is linear in the centered context write.  Split I into nine semantic head
terms plus an always-retained BF16 head-sum remainder, then score 21 physical
arms with T/C/S fixed: FULL, ZERO_I, NUMERIC, every SINGLE_h, every DROP_h.

The 15:10 UTC pre-execution amendment added ZERO_I separately from NUMERIC so
the exact rung401 no-I boundary and the causal inertness of numerical remainder
are both tested.  Singleton benefit is CE(NUMERIC)-CE(SINGLE_h); removal benefit
is CE(DROP_h)-CE(FULL); endpoint average is their mean and is NOT a Shapley value.

Frozen predictions
------------------
pred_a: nine semantic I_h plus numerical I_eps reconstruct I at relMSE<=1e-8;
    FULL/ZERO_I reproduce rung401 parent CEs within1e-6; every arm has the live
    24/432/24/408 call census; BF16 head-write remainder relative squared energy
    <=1e-4; and |CE(NUMERIC)-CE(ZERO_I)|<=.002 in both roles.
pred_b: on SELECT, head3 has largest endpoint-average benefit and its Spearman
    correlation with the frozen historical direct-head costs is >=.50.
pred_c: positive endpoint-average top2 share>=.65 in both roles, same top head,
    and Spearman(FIT,SELECT)>=.75.
pred_d: on SELECT some singleton benefit>=.05, some removal benefit>=.02, and
    at least one head has both endpoint benefits positive.

Strong null: pred_a fails; FIT/SELECT or SELECT/direct Spearman<=0; both roles'
top2 shares<=.35; or every SELECT absolute endpoint average is <.01 nat.

All native MLP0/attention0 weights remain.  Added head means are diagnostic
reference statistics, not a compressor.  No FINAL, adoption, or compression.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

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
OUT = BQ / "mlp0_centered_interaction_head_carriers_results.json"
PARENT = BQ / "mlp0_centered_context_anova_exact_residual_results.json"
HEAD_MAP = BQ / "attn_head_map_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
D = 1152
N_HEAD = 9
HEAD_DIM = 128
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)


def _arms():
    result = [("FULL", "full", None), ("ZERO_I", "zero", None),
              ("NUMERIC", "numeric", None)]
    result.extend((f"SINGLE_{head}", "single", head) for head in range(N_HEAD))
    result.extend((f"DROP_{head}", "drop", head) for head in range(N_HEAD))
    return tuple(result)


ARMS = _arms()


def _rank(values: torch.Tensor):
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks


def _spearman(left, right):
    a = _rank(torch.as_tensor(left, dtype=torch.float64))
    b = _rank(torch.as_tensor(right, dtype=torch.float64))
    a -= a.mean()
    b -= b.mean()
    denominator = a.norm() * b.norm()
    return float((a @ b) / denominator) if float(denominator) > 0 else 0.0


def _top2_positive_share(values):
    positive = torch.as_tensor(values, dtype=torch.float64).clamp_min(0)
    total = float(positive.sum())
    return float(positive.topk(2).values.sum()) / total if total > 0 else 0.0


@torch.no_grad()
def _attention0_with_heads(block, state, first_value):
    captured = {}

    def capture(_module, args):
        captured["joined"] = args[0].detach()

    handle = block.attn.c_proj.register_forward_pre_hook(capture)
    try:
        write, next_value = block.attn(state, first_value)
    finally:
        handle.remove()
    joined = captured["joined"]
    weight = block.attn.c_proj.weight.detach().float()
    head_writes = []
    for head in range(N_HEAD):
        band = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        head_writes.append(F.linear(joined[..., band].float(), weight[:, band]))
    head_writes = torch.stack(head_writes)
    epsilon = write.float() - head_writes.sum(0)
    return write, next_value, head_writes, epsilon


@torch.no_grad()
def _head_reference(model, fit_rows, reference, device):
    block0 = model.transformer.h[0]
    head_sum = torch.zeros(N_HEAD, D, dtype=torch.float64)
    epsilon_sum = torch.zeros(D, dtype=torch.float64)
    epsilon_energy = 0.0
    write_energy = 0.0
    count = 0
    for start in range(0, len(fit_rows), DOCUMENT_BATCH):
        tokens = fit_rows[start:start + DOCUMENT_BATCH, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        state = F.rms_norm(token_base, (D,))
        write, _, heads, epsilon = _attention0_with_heads(block0, state, None)
        head_sum += heads.double().sum((1, 2)).cpu()
        epsilon_sum += epsilon.double().sum((0, 1)).cpu()
        epsilon_energy += float(epsilon.double().square().sum())
        write_energy += float(write.double().square().sum())
        count += heads.shape[1] * heads.shape[2]
    head_means = (head_sum / count).to(device=device, dtype=torch.float32)
    observed_epsilon_mean = (epsilon_sum / count).to(device=device, dtype=torch.float32)
    # Close against the exact parent context mean without assigning rounding to a semantic head.
    epsilon_mean = reference["context_mean"] - head_means.sum(0)
    mean_closure_relmse = float(
        (epsilon_mean - observed_epsilon_mean).double().square().sum()
        / reference["context_mean"].double().square().sum().clamp_min(1e-30))
    return head_means, epsilon_mean, {
        "fit_head_write_bf16_remainder_relative_squared_energy":
            epsilon_energy / max(write_energy, 1e-30),
        "fit_head_mean_closure_relative_mse": mean_closure_relmse,
        "fit_head_reference_positions": count,
    }


@torch.no_grad()
def _exact_components(base, token_base, attention_write, normalized,
                      reference, left, right, down):
    retained, branches, g, gain, collinearity = base._components(
        token_base, attention_write, normalized, reference, left, right, down)
    z = normalized.float()
    raw = token_base.float() + attention_write.float()
    denominator = raw.square().sum(-1, keepdim=True).clamp_min(1e-30)
    scale = (z * raw).sum(-1, keepdim=True) / denominator
    scaled_raw = scale * raw
    residual = z - scaled_raw
    explicit_residual = (
        base._T(scaled_raw, residual, left, right, down)
        + base._T(residual, scaled_raw, left, right, down)
        + base._T(residual, residual, left, right, down))
    return retained + explicit_residual, branches, g, gain, collinearity


@torch.no_grad()
def _split_interaction(base, token_base, head_writes, epsilon, reference,
                       head_means, epsilon_mean, parent_interaction,
                       left, right, down):
    token_delta = token_base.float() - reference["token_mean"]
    lt = F.linear(token_delta, left)
    rt = F.linear(token_delta, right)
    semantic = []
    for head in range(N_HEAD):
        context_delta = head_writes[head] - head_means[head]
        lc = F.linear(context_delta, left)
        rc = F.linear(context_delta, right)
        semantic.append(reference["gain_mean"] * F.linear(lt * rc + lc * rt, down))
    semantic = torch.stack(semantic)
    epsilon_delta = epsilon - epsilon_mean
    le = F.linear(epsilon_delta, left)
    re = F.linear(epsilon_delta, right)
    numerical_formula = reference["gain_mean"] * F.linear(lt * re + le * rt, down)
    preclose = semantic.sum(0) + numerical_formula
    arithmetic_closing = parent_interaction - preclose
    numerical = numerical_formula + arithmetic_closing
    return semantic, numerical, preclose, arithmetic_closing


def _included(mode, head):
    if mode == "full":
        return set(range(N_HEAD)), True
    if mode == "zero":
        return set(), False
    if mode == "numeric":
        return set(), True
    if mode == "single":
        return {head}, True
    if mode == "drop":
        return set(range(N_HEAD)) - {head}, True
    raise ValueError(mode)


@torch.no_grad()
def _score_role(model, rows, device, reference, head_means, epsilon_mean, base):
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade

    block0 = model.transformer.h[0]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    document_ce = {label: [] for label, _, _ in ARMS}
    calls = {label: {"forwards": 0, "attention": 0, "site0": 0, "other_mlp": 0}
             for label, _, _ in ARMS}
    diag = {
        "interaction_num": 0.0, "interaction_den": 0.0,
        "preclose_num": 0.0, "closing_num": 0.0,
        "head_epsilon_num": 0.0, "head_write_den": 0.0,
        "state_max_abs_error": 0.0,
    }

    for start in range(0, len(rows), DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        attention_state = F.rms_norm(token_base, (D,))
        attention_write, first_value, head_writes, epsilon = _attention0_with_heads(
            block0, attention_state, None)
        normalized = F.rms_norm(token_base + attention_write, (D,))
        native0 = block0.mlp(normalized)
        retained, branches, _, _, _ = _exact_components(
            base, token_base, attention_write, normalized, reference, left, right, down)
        semantic, numerical, preclose, closing = _split_interaction(
            base, token_base, head_writes, epsilon, reference, head_means,
            epsilon_mean, branches["I"], left, right, down)

        reconstructed = semantic.sum(0) + numerical
        diag["interaction_num"] += float(
            (reconstructed.double() - branches["I"].double()).square().sum())
        diag["interaction_den"] += float(branches["I"].double().square().sum())
        diag["preclose_num"] += float(
            (preclose.double() - branches["I"].double()).square().sum())
        diag["closing_num"] += float(closing.double().square().sum())
        diag["head_epsilon_num"] += float(epsilon.double().square().sum())
        diag["head_write_den"] += float(attention_write.double().square().sum())

        cached_writes = {}
        for label, mode, head in ARMS:
            if mode == "zero":
                # Match rung401's one-tensor BF16 subtraction exactly.  Summing the
                # ten float head/numerical pieces before casting changes rounding.
                cached_writes[label] = native0 - branches["I"].to(native0.dtype)
                continue
            semantic_set, keep_numerical = _included(mode, head)
            omitted = sum(
                (semantic[h] for h in range(N_HEAD) if h not in semantic_set),
                start=torch.zeros_like(numerical))
            if not keep_numerical:
                omitted = omitted + numerical
            cached_writes[label] = native0 - omitted.to(native0.dtype)

        for label, _, _ in ARMS:
            def attention(event, label=label):
                calls[label]["attention"] += 1
                if event.site == 0:
                    return attention_write, first_value
                return event.block.attn(event.state, event.first_value)

            def mlp(event, label=label):
                if event.site == 0:
                    calls[label]["site0"] += 1
                    diag["state_max_abs_error"] = max(
                        diag["state_max_abs_error"],
                        float((event.state.float() - normalized.float()).abs().max()))
                    return cached_writes[label]
                calls[label]["other_mlp"] += 1
                return event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            losses = F.cross_entropy(
                logits[:, SCORING].transpose(1, 2), targets[:, SCORING],
                reduction="none").mean(1)
            document_ce[label].extend(float(loss) for loss in losses)
            calls[label]["forwards"] += 1

    expected = len(rows) // DOCUMENT_BATCH
    wanted = {"forwards": expected, "attention": 18 * expected,
              "site0": expected, "other_mlp": 17 * expected}
    live_census = all(calls[label] == wanted and len(document_ce[label]) == len(rows)
                      for label in document_ce)
    pooled = {label: float(torch.tensor(values, dtype=torch.float64).mean())
              for label, values in document_ce.items()}
    singleton = [pooled["NUMERIC"] - pooled[f"SINGLE_{h}"] for h in range(N_HEAD)]
    removal = [pooled[f"DROP_{h}"] - pooled["FULL"] for h in range(N_HEAD)]
    average = [(singleton[h] + removal[h]) / 2 for h in range(N_HEAD)]
    denominator = max(diag["interaction_den"], 1e-30)
    return {
        "pooled_ce": pooled,
        "conditional_full_interaction_benefit": pooled["ZERO_I"] - pooled["FULL"],
        "numerical_interaction_ce_effect": pooled["ZERO_I"] - pooled["NUMERIC"],
        "singleton_benefit": singleton,
        "removal_benefit": removal,
        "endpoint_average_benefit": average,
        "top_head": int(torch.tensor(average).argmax()),
        "positive_top2_share": _top2_positive_share(average),
        "diagnostics": {
            "interaction_sum_relative_mse": diag["interaction_num"] / denominator,
            "interaction_preclosing_relative_mse": diag["preclose_num"] / denominator,
            "interaction_arithmetic_closing_relative_energy": diag["closing_num"] / denominator,
            "head_write_bf16_remainder_relative_squared_energy":
                diag["head_epsilon_num"] / max(diag["head_write_den"], 1e-30),
            "state_max_abs_error": diag["state_max_abs_error"],
            "live_census": live_census,
        },
        "calls": calls,
    }


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(ARMS) == 21 and len({label for label, _, _ in ARMS}) == 21
        assert N_HEAD * HEAD_DIM == D
        assert PARENT.name == "mlp0_centered_context_anova_exact_residual_results.json"
        print("MLP0 INTERACTION HEAD CARRIERS | dry run: 21 arms and bars valid")
        return

    started = time.time()
    parent = json.loads(PARENT.read_text())
    head_map = json.loads(HEAD_MAP.read_text())
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import mlp0_centered_context_anova_factorial as base

    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    reference = base._reference_moments(model, fit_rows, device)
    head_means, epsilon_mean, reference_diag = _head_reference(
        model, fit_rows, reference, device)
    roles = {
        "FIT": _score_role(model, fit_rows, device, reference, head_means,
                           epsilon_mean, base),
        "SELECT": _score_role(model, select_rows, device, reference, head_means,
                              epsilon_mean, base),
    }

    direct_cost = [float(row["cost"]) for row in head_map["heads"]["0"]]
    fit_average = roles["FIT"]["endpoint_average_benefit"]
    select_average = roles["SELECT"]["endpoint_average_benefit"]
    split_rho = _spearman(fit_average, select_average)
    direct_rho = _spearman(select_average, direct_cost)
    parent_differences = {}
    for role in ("FIT", "SELECT"):
        parent_ce = parent["roles"][role]["pooled_ce"]
        parent_differences[role] = {
            "FULL_vs_parent_full": abs(roles[role]["pooled_ce"]["FULL"]
                                       - parent_ce["T+C+I+S"]),
            "ZERO_I_vs_parent_no_I": abs(roles[role]["pooled_ce"]["ZERO_I"]
                                      - parent_ce["T+C+S"]),
        }

    pred_a = all(
        roles[role]["diagnostics"]["interaction_sum_relative_mse"] <= 1e-8
        and roles[role]["diagnostics"]["live_census"]
        and roles[role]["diagnostics"]["head_write_bf16_remainder_relative_squared_energy"] <= 1e-4
        and abs(roles[role]["numerical_interaction_ce_effect"]) <= .002
        and max(parent_differences[role].values()) <= 1e-6
        for role in ("FIT", "SELECT"))
    pred_b = roles["SELECT"]["top_head"] == 3 and direct_rho >= .50
    pred_c = (
        all(roles[role]["positive_top2_share"] >= .65 for role in ("FIT", "SELECT"))
        and roles["FIT"]["top_head"] == roles["SELECT"]["top_head"]
        and split_rho >= .75)
    select_singleton = roles["SELECT"]["singleton_benefit"]
    select_removal = roles["SELECT"]["removal_benefit"]
    pred_d = (
        max(select_singleton) >= .05
        and max(select_removal) >= .02
        and any(select_singleton[h] > 0 and select_removal[h] > 0
                for h in range(N_HEAD)))
    strong_null = (
        not pred_a
        or split_rho <= 0
        or direct_rho <= 0
        or all(roles[role]["positive_top2_share"] <= .35
               for role in ("FIT", "SELECT"))
        or max(abs(value) for value in select_average) < .01)

    if pred_a and pred_b and pred_c and pred_d and not strong_null:
        next_step = "source_position_resolution_of_fixed_top_heads"
    elif pred_a and pred_c and pred_d and not strong_null:
        next_step = "source_position_resolution_of_new_top_heads"
    elif pred_a and not strong_null:
        next_step = "branch_resolved_p448_context_projection_audit"
    else:
        next_step = "instrument_repair_only"

    result = {
        "status": "mlp0_centered_interaction_head_carriers_complete",
        "rung": 402,
        "claim_level": "head_resolved_interaction_identification_not_compression",
        "definition": {
            "semantic_head": "I_h from centered output-projected attention0 head write",
            "numerical": "BF16 head-sum and float arithmetic remainder, never assigned to a semantic head",
            "singleton_benefit": "CE(NUMERIC)-CE(SINGLE_h)",
            "removal_benefit": "CE(DROP_h)-CE(FULL)",
            "endpoint_average": "half the singleton plus removal benefits; not Shapley",
        },
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows), "FINAL_opened": 0},
        "positions_per_document": SCORING.stop - SCORING.start,
        "arms": [label for label, _, _ in ARMS],
        "literal_diagnostic_storage": {
            "native_mlp0_values_reused": 15_926_400,
            "head_mean_values_added": N_HEAD * D,
            "compression_claim": False,
        },
        "reference_diagnostics": reference_diag,
        "historical_direct_head_cost": direct_cost,
        "historical_direct_head_top2_share": head_map["top2_cost_frac_by_layer"]["0"],
        "roles": roles,
        "transport": {
            "fit_select_endpoint_spearman": split_rho,
            "select_vs_historical_direct_cost_spearman": direct_rho,
            "fit_top_head": roles["FIT"]["top_head"],
            "select_top_head": roles["SELECT"]["top_head"],
        },
        "parent_pooled_ce_absolute_differences": parent_differences,
        'pred_a_exact_live_parent_and_numerical_controls': bool(pred_a),
        'pred_b_old_head_map_predicts_isolated_interaction': bool(pred_b),
        'pred_c_sparse_split_stable_carrier': bool(pred_c),
        'pred_d_individually_material_head_route': bool(pred_d),
        "null_head_attribution_invalid_or_unstructured": bool(strong_null),
        "next_step": next_step,
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 INTERACTION HEAD CARRIERS DONE", flush=True)


if __name__ == "__main__":
    main()
