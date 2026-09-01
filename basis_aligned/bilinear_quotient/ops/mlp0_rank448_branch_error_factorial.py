"""RUNG 403 -- EXACT BRANCH ERRORS OF THE FIXED MLP0 p448 PROGRAM.

Reconstruct rung328's fit-A rank448 shared-input MLP0 program without changing
its rank or rows.  Resolve its deployed change from native into exact
product-reference delta-T/C/I/S factors plus A, an explicit auxiliary factor
containing the changed constant, exact vector-normalization residual, and BF16
closing arithmetic.  Score all 32 subsets on unchanged rung401 FIT/SELECT rows.

Frozen predictions
------------------
pred_a: native/compact analytical identities <=1e-8 both roles; EMPTY/FULL are
    exact native/compact BF16 states; endpoint CE differences <=1e-6; every arm
    has the live 24/432/24/408 census; rank/shapes/fit rows/9,954,432-value price
    are exact; covariance retained energy is within2e-6 of .9011108875.
pred_b: SELECT total compact damage is in [0,.030] nat; FIT has the same sign
    and total-damage magnitude differs by <=.015.
pred_c: T is the largest positive named Shapley damage on both roles; SELECT T
    >=.002; named FIT/SELECT rank Spearman >=.75.
pred_d: the largest positive named branch supplies >=40% of the positive named
    total on both roles; |auxiliary Shapley|<=.005 on both.

Strong null: pred_a fails; |SELECT total damage|<.001; named split Spearman<=0;
or |auxiliary Shapley| is at least the sum of absolute named Shapleys on either
role.  T dominance routes to an exact-token/private bypass; I to a distributed
interaction metric; C/S to their own preservation; auxiliary dominance to
constant/vector-residual/precision anatomy.  Diagnostic only: no rank tuning,
compression license, adoption, FINAL, or source-position expansion.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import itertools
import json
import math
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
OUT = BQ / "mlp0_rank448_branch_error_factorial_results.json"
PARENT = BQ / "mlp0_centered_context_anova_exact_residual_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
FIT_CACHE = BQ / ".rowcache/fineweb_n192_skip11000.pt"
FIT_SLICE = (0, 24)
RANK = 448
D = 1152
H = 4608
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
NAMED = ("T", "C", "I", "S")
FACTORS = NAMED + ("A",)
ARMS = tuple(
    frozenset(factor for index, factor in enumerate(FACTORS) if mask & (1 << index))
    for mask in range(1 << len(FACTORS))
)
NATIVE_VALUES = 3 * H * D + D
COMPACT_VALUES = D * RANK + 2 * H * RANK + H * D + D
SAVING_VALUES = NATIVE_VALUES - COMPACT_VALUES
RUNG328_RETAINED_ENERGY = 0.9011108875274658


def _arm_name(subset: frozenset[str]) -> str:
    return "+".join(factor for factor in FACTORS if factor in subset) or "EMPTY"


def _mobius(performance: dict[frozenset[str], float]):
    result = {}
    for subset in ARMS:
        value = 0.0
        items = tuple(subset)
        for size in range(len(items) + 1):
            for chosen in itertools.combinations(items, size):
                child = frozenset(chosen)
                value += (-1.0) ** (len(subset) - len(child)) * performance[child]
        result[_arm_name(subset)] = value
    return result


def _shapley(performance: dict[frozenset[str], float]):
    count = len(FACTORS)
    result = {}
    for factor in FACTORS:
        total = 0.0
        others = tuple(item for item in FACTORS if item != factor)
        for size in range(len(others) + 1):
            coefficient = (
                math.factorial(size) * math.factorial(count - size - 1)
                / math.factorial(count)
            )
            for chosen in itertools.combinations(others, size):
                subset = frozenset(chosen)
                total += coefficient * (
                    performance[subset | {factor}] - performance[subset]
                )
        result[factor] = total
    return result


def _rank(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks


def _spearman(left, right):
    a, b = _rank(left), _rank(right)
    a -= a.mean()
    b -= b.mean()
    denominator = a.norm() * b.norm()
    return float((a @ b) / denominator) if float(denominator) else 0.0


def _positive_top_share(values):
    positive = torch.as_tensor(values, dtype=torch.float64).clamp_min(0)
    total = float(positive.sum())
    return float(positive.max()) / total if total > 0 else 0.0


@torch.no_grad()
def _project_pair(values: torch.Tensor, factors: dict):
    values = values.float()
    encoder = factors.get("encoder")
    if encoder is not None:
        values = F.linear(values, encoder)
    return F.linear(values, factors["left"]), F.linear(values, factors["right"])


@torch.no_grad()
def _bilinear(left_input: torch.Tensor, right_input: torch.Tensor, factors: dict):
    left, _ = _project_pair(left_input, factors)
    _, right = _project_pair(right_input, factors)
    return F.linear(left * right, factors["down"])


@torch.no_grad()
def _reference_for_factors(token_cpu, context_cpu, gain_cpu, factors, device):
    token_mean = token_cpu.mean(0).to(device)
    context_mean = context_cpu.mean(0).to(device)
    gain_mean = float(gain_cpu.mean())
    token_self = torch.zeros(D, device=device)
    context_self = torch.zeros(D, device=device)
    count = 0
    for start in range(0, len(token_cpu), 256):
        token = token_cpu[start:start + 256].to(device) - token_mean
        context = context_cpu[start:start + 256].to(device) - context_mean
        token_self += _bilinear(token, token, factors).sum(0)
        context_self += _bilinear(context, context, factors).sum(0)
        count += len(token)
    token_self /= count
    context_self /= count
    total_mean = token_mean + context_mean
    mu_g = _bilinear(total_mean[None], total_mean[None], factors)[0]
    mu_g += token_self + context_self
    return {
        "token_mean": token_mean,
        "context_mean": context_mean,
        "token_self_mean": token_self,
        "context_self_mean": context_self,
        "mu_g": mu_g,
        "gain_mean": gain_mean,
        "fit_positions": count,
    }


@torch.no_grad()
def _exact_components(base, token_base, attention_write, normalized,
                      reference, factors):
    token_delta = token_base.float() - reference["token_mean"]
    context_delta = attention_write.float() - reference["context_mean"]
    total_mean = reference["token_mean"] + reference["context_mean"]
    lt, rt = _project_pair(token_delta, factors)
    lc, rc = _project_pair(context_delta, factors)
    lm, rm = _project_pair(total_mean, factors)
    down = factors["down"]
    token_main = F.linear(lt * rm + lm * rt + lt * rt, down)
    token_main -= reference["token_self_mean"]
    context_main = F.linear(lc * rm + lm * rc + lc * rc, down)
    context_main -= reference["context_self_mean"]
    interaction = F.linear(lt * rc + lc * rt, down)
    g = reference["mu_g"] + token_main + context_main + interaction
    gain, collinearity = base._gain(normalized, token_base, attention_write)
    gain_mean = reference["gain_mean"]
    branches = {
        "T": gain_mean * token_main,
        "C": gain_mean * context_main,
        "I": gain_mean * interaction,
        "S": (gain - gain_mean) * g,
    }
    retained = gain_mean * reference["mu_g"]
    z = normalized.float()
    raw = token_base.float() + attention_write.float()
    denominator = raw.square().sum(-1, keepdim=True).clamp_min(1e-30)
    scale = (z * raw).sum(-1, keepdim=True) / denominator
    scaled_raw = scale * raw
    residual = z - scaled_raw
    retained = retained + (
        _bilinear(scaled_raw, residual, factors)
        + _bilinear(residual, scaled_raw, factors)
        + _bilinear(residual, residual, factors)
    )
    return retained, branches, collinearity


@torch.no_grad()
def _compact_deployed(state, program, dtype):
    code = F.linear(state.float(), program["encoder"])
    hidden = F.linear(code, program["left"]) * F.linear(code, program["right"])
    return F.linear(hidden, program["down"], program["bias"]).to(dtype)


@torch.no_grad()
def _score_role(model, rows, device, base, native_factors, compact_factors,
                native_reference, compact_reference):
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade

    labels = {subset: _arm_name(subset) for subset in ARMS}
    document_ce = {label: [] for label in labels.values()}
    calls = {
        label: {"forwards": 0, "attention": 0, "site0": 0, "other_mlp": 0}
        for label in labels.values()
    }
    diagnostics = {
        "native_identity_num": 0.0,
        "native_identity_den": 0.0,
        "compact_identity_num": 0.0,
        "compact_identity_den": 0.0,
        "native_endpoint_max_abs": 0.0,
        "compact_endpoint_max_abs": 0.0,
        "state_replay_max_abs": 0.0,
        "factor_squared_energy": {name: 0.0 for name in FACTORS},
        "native_deployed_squared_energy": 0.0,
        "positions": 0,
    }

    for start in range(0, len(rows), DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        block0 = model.transformer.h[0]
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        cache = {}

        for subset in ARMS:
            label = labels[subset]

            def attention(event, label=label):
                calls[label]["attention"] += 1
                return event.block.attn(event.state, event.first_value)

            def mlp(event, subset=subset, label=label):
                if event.site != 0:
                    calls[label]["other_mlp"] += 1
                    return event.block.mlp(event.state)
                calls[label]["site0"] += 1
                native = event.block.mlp(event.state)
                if not cache:
                    retained_n, branches_n, _ = _exact_components(
                        base, token_base, event.attention_write, event.state,
                        native_reference, native_factors)
                    retained_p, branches_p, _ = _exact_components(
                        base, token_base, event.attention_write, event.state,
                        compact_reference, compact_factors)
                    direct_n = _bilinear(event.state, event.state, native_factors)
                    direct_p = _bilinear(event.state, event.state, compact_factors)
                    analytical_n = retained_n + sum(branches_n.values())
                    analytical_p = retained_p + sum(branches_p.values())
                    diagnostics["native_identity_num"] += float(
                        (analytical_n.double() - direct_n.double()).square().sum())
                    diagnostics["native_identity_den"] += float(direct_n.double().square().sum())
                    diagnostics["compact_identity_num"] += float(
                        (analytical_p.double() - direct_p.double()).square().sum())
                    diagnostics["compact_identity_den"] += float(direct_p.double().square().sum())
                    compact = _compact_deployed(event.state, compact_factors, native.dtype)
                    deltas = {name: branches_p[name] - branches_n[name] for name in NAMED}
                    named_sum = sum(deltas.values(), start=torch.zeros_like(deltas["T"]))
                    deltas["A"] = compact.float() - native.float() - named_sum
                    cache.update({
                        "state": event.state.detach().clone(),
                        "attention": event.attention_write.detach().clone(),
                        "native": native.detach().clone(),
                        "compact": compact.detach().clone(),
                        "deltas": {name: value.detach().clone() for name, value in deltas.items()},
                    })
                    scoring_positions = compact[:, SCORING]
                    for name in FACTORS:
                        diagnostics["factor_squared_energy"][name] += float(
                            deltas[name][:, SCORING].double().square().sum())
                    diagnostics["native_deployed_squared_energy"] += float(
                        native[:, SCORING].double().square().sum())
                    diagnostics["positions"] += scoring_positions.shape[0] * scoring_positions.shape[1]
                else:
                    diagnostics["state_replay_max_abs"] = max(
                        diagnostics["state_replay_max_abs"],
                        float((event.state - cache["state"]).abs().max()),
                        float((event.attention_write - cache["attention"]).abs().max()),
                    )

                if not subset:
                    result = native
                    diagnostics["native_endpoint_max_abs"] = max(
                        diagnostics["native_endpoint_max_abs"],
                        float((result - cache["native"]).abs().max()))
                elif len(subset) == len(FACTORS):
                    result = cache["compact"]
                    diagnostics["compact_endpoint_max_abs"] = max(
                        diagnostics["compact_endpoint_max_abs"],
                        float((result - cache["compact"]).abs().max()))
                else:
                    change = sum(
                        (cache["deltas"][name] for name in subset),
                        start=torch.zeros_like(cache["deltas"]["T"]),
                    )
                    result = (native.float() + change).to(native.dtype)
                return result

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            losses = F.cross_entropy(
                logits[:, SCORING].transpose(1, 2), targets[:, SCORING], reduction="none"
            ).mean(1)
            document_ce[label].extend(float(loss) for loss in losses)
            calls[label]["forwards"] += 1

    expected = len(rows) // DOCUMENT_BATCH
    wanted = {"forwards": expected, "attention": 18 * expected,
              "site0": expected, "other_mlp": 17 * expected}
    live_census = all(
        calls[label] == wanted and len(document_ce[label]) == len(rows)
        for label in document_ce
    )
    pooled_ce = {
        label: float(torch.tensor(values, dtype=torch.float64).mean())
        for label, values in document_ce.items()
    }
    performance = {subset: pooled_ce[labels[subset]] for subset in ARMS}
    shapley = _shapley(performance)
    named_values = [shapley[name] for name in NAMED]
    energy_denominator = max(diagnostics["native_deployed_squared_energy"], 1e-30)
    return {
        "pooled_ce": pooled_ce,
        "total_compact_ce_damage": pooled_ce["T+C+I+S+A"] - pooled_ce["EMPTY"],
        "shapley_ce_damage": shapley,
        "mobius_ce_damage": _mobius(performance),
        "named_positive_top_share": _positive_top_share(named_values),
        "named_rank_order": sorted(NAMED, key=lambda name: shapley[name], reverse=True),
        "factor_relative_squared_output_energy": {
            name: diagnostics["factor_squared_energy"][name] / energy_denominator
            for name in FACTORS
        },
        "diagnostics": {
            "native_analytical_identity_relative_mse": diagnostics["native_identity_num"]
            / max(diagnostics["native_identity_den"], 1e-30),
            "compact_analytical_identity_relative_mse": diagnostics["compact_identity_num"]
            / max(diagnostics["compact_identity_den"], 1e-30),
            "native_endpoint_state_max_abs_error": diagnostics["native_endpoint_max_abs"],
            "compact_endpoint_state_max_abs_error": diagnostics["compact_endpoint_max_abs"],
            "pre_mlp0_state_replay_max_abs_error": diagnostics["state_replay_max_abs"],
            "live_census": live_census,
            "scored_positions": diagnostics["positions"],
        },
        "calls": calls,
    }


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(ARMS) == 32 and len({_arm_name(arm) for arm in ARMS}) == 32
        assert FIT_CACHE.exists() and PARENT.exists() and ROWS_RECEIPT.exists()
        assert COMPACT_VALUES == 9_954_432 and SAVING_VALUES == 5_971_968
        print("MLP0 p448 BRANCH ERROR | dry run: 32 arms, rows, price, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(BQ.parent / "qk_mdl"))
    import bilin18_observed_model_facade as facade
    import mlp0_centered_context_anova_factorial as base
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)

    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    program_fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    covariance = _covariance(model, program_fit_rows, _manual_logits)
    program, _basis, program_diagnostic = _rrr_program(
        model.transformer.h[0].mlp, covariance, rank=RANK)
    del covariance, _basis
    torch.cuda.empty_cache()

    native_factors = {
        "left": model.transformer.h[0].mlp.Left.weight.detach().float(),
        "right": model.transformer.h[0].mlp.Right.weight.detach().float(),
        "down": model.transformer.h[0].mlp.Down.weight.detach().float(),
    }
    compact_factors = {name: value.detach().float() for name, value in program.items()}
    expected_shapes = {
        "encoder": (RANK, D), "left": (H, RANK), "right": (H, RANK),
        "down": (D, H), "bias": (D,),
    }
    observed_shapes = {name: tuple(value.shape) for name, value in compact_factors.items()}

    token_cpu, context_cpu, gain_cpu = base._capture_inputs(model, fit_rows, device)
    native_reference = _reference_for_factors(
        token_cpu, context_cpu, gain_cpu, native_factors, device)
    compact_reference = _reference_for_factors(
        token_cpu, context_cpu, gain_cpu, compact_factors, device)
    roles = {
        "FIT": _score_role(
            model, fit_rows, device, base, native_factors, compact_factors,
            native_reference, compact_reference),
        "SELECT": _score_role(
            model, select_rows, device, base, native_factors, compact_factors,
            native_reference, compact_reference),
    }

    parent = json.loads(PARENT.read_text())
    parent_native_ce_difference = {
        role: abs(roles[role]["pooled_ce"]["EMPTY"]
                  - parent["roles"][role]["pooled_ce"]["T+C+I+S"])
        for role in ("FIT", "SELECT")
    }
    exact_program = (
        observed_shapes == expected_shapes
        and int(program_diagnostic["rank"]) == RANK
        and abs(program_diagnostic["context_cov_retained_energy"]
                - RUNG328_RETAINED_ENERGY) <= 2e-6
    )
    exact_instrument = (
        exact_program
        and all(
            roles[role]["diagnostics"]["native_analytical_identity_relative_mse"] <= 1e-8
            and roles[role]["diagnostics"]["compact_analytical_identity_relative_mse"] <= 1e-8
            and roles[role]["diagnostics"]["native_endpoint_state_max_abs_error"] == 0.0
            and roles[role]["diagnostics"]["compact_endpoint_state_max_abs_error"] == 0.0
            and roles[role]["diagnostics"]["pre_mlp0_state_replay_max_abs_error"] == 0.0
            and roles[role]["diagnostics"]["live_census"]
            and parent_native_ce_difference[role] <= 1e-6
            for role in ("FIT", "SELECT")
        )
    )
    fit_total = roles["FIT"]["total_compact_ce_damage"]
    select_total = roles["SELECT"]["total_compact_ce_damage"]
    sign_match = (fit_total >= 0) == (select_total >= 0)
    pred_b = 0 <= select_total <= .030 and sign_match and abs(fit_total - select_total) <= .015
    fit_shapley = roles["FIT"]["shapley_ce_damage"]
    select_shapley = roles["SELECT"]["shapley_ce_damage"]
    named_spearman = _spearman(
        [fit_shapley[name] for name in NAMED],
        [select_shapley[name] for name in NAMED])
    pred_c = (
        max(NAMED, key=lambda name: fit_shapley[name]) == "T"
        and max(NAMED, key=lambda name: select_shapley[name]) == "T"
        and select_shapley["T"] >= .002
        and named_spearman >= .75
    )
    pred_d = (
        all(roles[role]["named_positive_top_share"] >= .40 for role in ("FIT", "SELECT"))
        and all(abs(roles[role]["shapley_ce_damage"]["A"]) <= .005
                for role in ("FIT", "SELECT"))
    )
    auxiliary_dominates = any(
        abs(roles[role]["shapley_ce_damage"]["A"])
        >= sum(abs(roles[role]["shapley_ce_damage"][name]) for name in NAMED)
        for role in ("FIT", "SELECT")
    )
    strong_null = (
        not exact_instrument or abs(select_total) < .001
        or named_spearman <= 0 or auxiliary_dominates
    )
    select_top = max(NAMED, key=lambda name: select_shapley[name])
    if strong_null:
        next_object = None
    else:
        next_object = {
            "T": "exact_token_or_token_private_bypass",
            "I": "distributed_interaction_weighted_projection",
            "C": "context_main_preserving_projection",
            "S": "normalization_gain_aware_projection",
        }[select_top]

    result = {
        "status": "mlp0_rank448_branch_error_factorial_complete",
        "rung": 403,
        "claim_level": "exact_fixed_compressor_branch_attribution_not_new_compression",
        "convention": "CE added above native; lower is better",
        "definition": {
            "T": "compact-minus-native fixed-gain token-main branch",
            "C": "compact-minus-native fixed-gain context-main branch",
            "I": "compact-minus-native fixed-gain centered token-context interaction",
            "S": "compact-minus-native normalization-gain modulation",
            "A": "changed constant plus explicit vector-normalization residual plus BF16 closure",
        },
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows), "FINAL_opened": 0},
        "positions_per_document": SCORING.stop - SCORING.start,
        "program_identity": {
            "fit_cache": str(FIT_CACHE),
            "fit_rows_half_open": list(FIT_SLICE),
            "rank": RANK,
            "shapes": {name: list(shape) for name, shape in observed_shapes.items()},
            "fit_diagnostic": program_diagnostic,
            "rung328_retained_energy_target": RUNG328_RETAINED_ENERGY,
            "literal_mlp0_values": COMPACT_VALUES,
            "native_mlp0_values": NATIVE_VALUES,
            "saving_values": SAVING_VALUES,
        },
        "parent_rung401_result": str(PARENT),
        "parent_native_ce_difference": parent_native_ce_difference,
        "roles": roles,
        "named_fit_select_spearman": named_spearman,
        "fit_total_damage": fit_total,
        "select_total_damage": select_total,
        "select_top_named_branch": select_top,
        "auxiliary_dominates_named_magnitude": auxiliary_dominates,
        'pred_a_exact_instrument_and_fixed_program_identity': bool(exact_instrument),
        'pred_b_fixed_p448_remains_useful_on_grammar_rows': bool(pred_b),
        'pred_c_token_main_is_leading_named_obstruction': bool(pred_c),
        'pred_d_named_grammar_localizes_error_not_auxiliary': bool(pred_d),
        "null_branch_audit_is_invalid_or_uninformative": bool(strong_null),
        "next_object": next_object,
        "rank_tuning_licensed": False,
        "compression_or_adoption_licensed": False,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 p448 BRANCH ERROR FACTORIAL DONE", flush=True)


if __name__ == "__main__":
    main()
