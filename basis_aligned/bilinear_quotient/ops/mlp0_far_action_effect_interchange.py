"""RUNG 398 -- FAR-ACTION DOWNSTREAM-EQUIVALENCE PHYSICAL INTERCHANGE.

Rebuild the exact M/L/Q length-one decomposition.  Token-id-mod5 fitting
tokens are donors and the untouched fifth are receivers.  For L and Q
separately, find donors whose component cosine to the receiver is <=.50 but
whose conditional attention1 or MLP1 effect is nearest.  Physically swap only
that donor component into the receiver's full M+L+Q frame.  Selection in one
consumer is validated in the other.  Raw-embedding nearest, component-action
nearest, and seed398 random donors are frozen controls.

Frozen predictions
------------------
pred_a: live full-action relative error <=1e-6, donor/receiver sets disjoint,
    >=99% far-candidate feasibility, observed far component cosine <=.5001,
    and every physical swap arm runs.
pred_b: both L cross-consumer conditional-effect cosines >=.60, at least one
    >=.75, and each beats random by >=.15 and raw-neighbor by >=.05.
pred_c: both Q cross-consumer cosines >=.50, at least one >=.70, with the same
    random/raw margins.
pred_d: some far-effect scheme preserves per-token cosine >=.80 in both
    consumers for >=5% of receivers, uses >=100 distinct donors, and has
    median selector-effect cosine >=.80.

Strong null: far feasibility <50%; or all four L/Q cross-consumer cosines
<=.30; or every far-effect route is within .03 of its random control.  A+B or
A+C+D supports downstream equivalence despite different storage directions.
Otherwise retain exact token identity and move to reader-weighted quadratic
spectra.  No context promotion or compression claim.

Literal diagnostic intervention-table price is 115,793,280 float values:
M(1,152) plus complete per-token L and Q tables (2*50,257*1,152).  This is a
causal assay, not a proposed shipped program; response arrays are measurements.
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


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_far_action_effect_interchange_results.json"
DEV = "cuda"
D = 1152
REAL_V = 50_257
EVAL_V = 10_052
FAR_COSINE = .50
DIAGNOSTIC_VALUES = 115_793_280


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float(1 - (target - prediction).square().sum()
                 / target.square().sum().clamp_min(1e-20))


def _metrics(target: torch.Tensor, prediction: torch.Tensor):
    a, b = target.flatten(), prediction.flatten()
    return {
        "r2": _r2(target, prediction),
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-20)),
        "norm_ratio": float(b.norm() / a.norm().clamp_min(1e-20)),
        "relative_error": float((a - b).norm() / a.norm().clamp_min(1e-20)),
    }


def _row_cosine(a: torch.Tensor, b: torch.Tensor):
    return F.cosine_similarity(a.float(), b.float(), dim=1, eps=1e-12)


@torch.no_grad()
def _base_outputs(model, token_ids: torch.Tensor, mean: torch.Tensor,
                  linear: torch.Tensor, quadratic: torch.Tensor):
    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    names = ("M", "ML", "MQ", "MLQ")
    outputs = {name: {"attention1": [], "mlp1": []} for name in names}
    live_num = live_den = 0.0
    batches = 0
    for start in range(0, len(token_ids), 256):
        ids = token_ids[start:start + 256]
        token = ids.to(DEV).view(-1, 1)
        raw = F.rms_norm(model.transformer.wte(token), (D,))
        remix = (block0.lambdas[0] + block0.lambdas[1]) * raw
        attention0, value0 = block0.attn(F.rms_norm(remix, (D,)), None)
        pre0 = remix + attention0
        z = F.rms_norm(pre0, (D,))
        write = block0.mlp(z)
        bias = block0.mlp.Down_bias.view(1, 1, D).to(write)
        native_action = write - bias
        arm_actions = {
            "M": mean[ids],
            "ML": mean[ids] + linear[ids],
            "MQ": mean[ids] + quadratic[ids],
            "MLQ": mean[ids] + linear[ids] + quadratic[ids],
        }
        supplied = arm_actions["MLQ"][:, None]
        live_num += float((native_action - supplied).float().square().sum())
        live_den += float(native_action.float().square().sum())
        for name, action in arm_actions.items():
            post0 = pre0 + bias + action[:, None]
            remixed1 = block1.lambdas[0] * post0 + block1.lambdas[1] * raw
            attention1, _ = block1.attn(F.rms_norm(remixed1, (D,)), value0)
            mlp1 = block1.mlp(F.rms_norm(remixed1 + attention1, (D,)))
            outputs[name]["attention1"].append(attention1[:, 0].float().cpu())
            outputs[name]["mlp1"].append(mlp1[:, 0].float().cpu())
        batches += 1
    return {
        name: {kind: torch.cat(parts).to(DEV) for kind, parts in by_kind.items()}
        for name, by_kind in outputs.items()
    }, {
        "base_batches": batches,
        "live_full_action_relative_error": (live_num / max(live_den, 1e-30)) ** .5,
    }


@torch.no_grad()
def _donor_maps(receiver_ids: torch.Tensor, donor_ids: torch.Tensor,
                raw: torch.Tensor, component: torch.Tensor,
                attention_effect: torch.Tensor, mlp_effect: torch.Tensor,
                seed: int):
    donor_component = F.normalize(component[donor_ids].float(), dim=1)
    donor_raw = F.normalize(raw[donor_ids].float(), dim=1)
    donor_attention = F.normalize(attention_effect[donor_ids].float(), dim=1)
    donor_mlp = F.normalize(mlp_effect[donor_ids].float(), dim=1)
    pieces = {key: [] for key in (
        "effect_attention_far", "effect_mlp_far", "action_nearest", "raw_nearest")}
    scores = {key: [] for key in (
        "attention_selector_cosine", "mlp_selector_cosine",
        "attention_far_action_cosine", "mlp_far_action_cosine")}
    feasible = []
    for start in range(0, len(receiver_ids), 192):
        ids = receiver_ids[start:start + 192]
        receiver_component = F.normalize(component[ids].float(), dim=1)
        action_similarity = receiver_component @ donor_component.T
        allowed = action_similarity <= FAR_COSINE
        feasible.append(allowed.any(1).cpu())

        receiver_attention = F.normalize(attention_effect[ids].float(), dim=1)
        attention_similarity = receiver_attention @ donor_attention.T
        attention_similarity.masked_fill_(~allowed, -torch.inf)
        attention_score, attention_position = attention_similarity.max(1)
        pieces["effect_attention_far"].append(donor_ids[attention_position].cpu())
        scores["attention_selector_cosine"].append(attention_score.cpu())
        scores["attention_far_action_cosine"].append(
            action_similarity.gather(1, attention_position[:, None])[:, 0].cpu())
        del attention_similarity

        receiver_mlp = F.normalize(mlp_effect[ids].float(), dim=1)
        mlp_similarity = receiver_mlp @ donor_mlp.T
        mlp_similarity.masked_fill_(~allowed, -torch.inf)
        mlp_score, mlp_position = mlp_similarity.max(1)
        pieces["effect_mlp_far"].append(donor_ids[mlp_position].cpu())
        scores["mlp_selector_cosine"].append(mlp_score.cpu())
        scores["mlp_far_action_cosine"].append(
            action_similarity.gather(1, mlp_position[:, None])[:, 0].cpu())
        pieces["action_nearest"].append(donor_ids[action_similarity.argmax(1)].cpu())

        receiver_raw = F.normalize(raw[ids].float(), dim=1)
        pieces["raw_nearest"].append(donor_ids[(receiver_raw @ donor_raw.T).argmax(1)].cpu())

    generator = torch.Generator().manual_seed(seed)
    random_positions = torch.randint(len(donor_ids), (len(receiver_ids),), generator=generator)
    pieces["random"] = [donor_ids.cpu()[random_positions]]
    return ({key: torch.cat(parts).to(DEV) for key, parts in pieces.items()},
            {key: torch.cat(parts) for key, parts in scores.items()},
            float(torch.cat(feasible).float().mean()))


@torch.no_grad()
def _swap_outputs(model, receiver_ids: torch.Tensor, mean: torch.Tensor,
                  linear: torch.Tensor, quadratic: torch.Tensor,
                  l_donors: dict[str, torch.Tensor], q_donors: dict[str, torch.Tensor]):
    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    names = ([f"L_{name}" for name in l_donors]
             + [f"Q_{name}" for name in q_donors])
    outputs = {name: {"attention1": [], "mlp1": []} for name in names}
    batches = 0
    for start in range(0, len(receiver_ids), 256):
        ids = receiver_ids[start:start + 256]
        token = ids.to(DEV).view(-1, 1)
        raw = F.rms_norm(model.transformer.wte(token), (D,))
        remix = (block0.lambdas[0] + block0.lambdas[1]) * raw
        attention0, value0 = block0.attn(F.rms_norm(remix, (D,)), None)
        pre0 = remix + attention0
        bias = block0.mlp.Down_bias.view(1, 1, D).to(raw)
        arms = {}
        for name, donors in l_donors.items():
            donor = donors[start:start + len(ids)]
            arms[f"L_{name}"] = mean[ids] + linear[donor] + quadratic[ids]
        for name, donors in q_donors.items():
            donor = donors[start:start + len(ids)]
            arms[f"Q_{name}"] = mean[ids] + linear[ids] + quadratic[donor]
        for name, action in arms.items():
            post0 = pre0 + bias + action[:, None]
            remixed1 = block1.lambdas[0] * post0 + block1.lambdas[1] * raw
            attention1, _ = block1.attn(F.rms_norm(remixed1, (D,)), value0)
            mlp1 = block1.mlp(F.rms_norm(remixed1 + attention1, (D,)))
            outputs[name]["attention1"].append(attention1[:, 0].float().cpu())
            outputs[name]["mlp1"].append(mlp1[:, 0].float().cpu())
        batches += 1
    return {
        name: {kind: torch.cat(parts).to(DEV) for kind, parts in by_kind.items()}
        for name, by_kind in outputs.items()
    }, batches


def _score_component(component: str, swap_outputs: dict, base_eval: dict,
                     schemes: tuple[str, ...]):
    without = "MQ" if component == "L" else "ML"
    target = {
        kind: base_eval["MLQ"][kind] - base_eval[without][kind]
        for kind in ("attention1", "mlp1")}
    result = {}
    row_scores = {}
    for scheme in schemes:
        name = f"{component}_{scheme}"
        prediction = {
            kind: swap_outputs[name][kind] - base_eval[without][kind]
            for kind in ("attention1", "mlp1")}
        result[scheme] = {kind: _metrics(target[kind], prediction[kind])
                          for kind in ("attention1", "mlp1")}
        row_scores[scheme] = {
            kind: _row_cosine(target[kind], prediction[kind]).cpu()
            for kind in ("attention1", "mlp1")}
        both = torch.minimum(row_scores[scheme]["attention1"], row_scores[scheme]["mlp1"])
        result[scheme]["fraction_both_consumer_cosine_ge_08"] = float((both >= .80).float().mean())
        result[scheme]["median_both_consumer_cosine"] = float(both.median())
    return result, row_scores


def _decode_examples(receiver_ids: torch.Tensor, donor_ids: torch.Tensor,
                     component: torch.Tensor, selector_scores: torch.Tensor,
                     row_scores: dict[str, torch.Tensor], limit: int = 12):
    import tiktoken
    encoder = tiktoken.get_encoding("gpt2")

    def decode(token: int):
        return encoder.decode_single_token_bytes(token).decode("utf-8", "backslashreplace")

    both = torch.minimum(row_scores["attention1"], row_scores["mlp1"])
    top = both.topk(min(limit, len(both))).indices
    examples = []
    for position in top.tolist():
        receiver, donor = int(receiver_ids[position]), int(donor_ids[position])
        examples.append({
            "receiver_id": receiver,
            "receiver_token": decode(receiver),
            "donor_id": donor,
            "donor_token": decode(donor),
            "component_cosine": float(_row_cosine(
                component[receiver:receiver + 1], component[donor:donor + 1])[0]),
            "selector_cosine": float(selector_scores[position]),
            "physical_attention1_cosine": float(row_scores["attention1"][position]),
            "physical_mlp1_cosine": float(row_scores["mlp1"][position]),
        })
    return examples


@torch.no_grad()
def main():
    schemes = ("effect_attention_far", "effect_mlp_far", "action_nearest",
               "raw_nearest", "random")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EVAL_V == sum(i % 5 == 0 for i in range(REAL_V))
        assert DIAGNOSTIC_VALUES == D + 2 * REAL_V * D
        assert FAR_COSINE == .50 and len(schemes) == 5
        print("MLP0 FAR-ACTION INTERCHANGE | dry run: split, donors, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, str(ROOT / "ops"))
    from tier2_model import load_elriggs
    from mlp0_mean_linear_quadratic_causal_factorial import (
        _capture, _complete_degree_one, _standardize)

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    raw_cpu, z_cpu, action_cpu = _capture(model)
    token_ids = torch.arange(REAL_V)
    fit_cpu, eval_cpu = token_ids % 5 != 0, token_ids % 5 == 0
    donor_ids = token_ids[fit_cpu].to(DEV)
    receiver_ids = token_ids[eval_cpu].to(DEV)
    assert not bool(torch.isin(receiver_ids, donor_ids).any())

    z, _z_mean, _z_scale = _standardize(z_cpu[fit_cpu], z_cpu)
    action, action_mean, action_scale = _standardize(action_cpu[fit_cpu], action_cpu)
    z, action = z.to(DEV), action.to(DEV)
    fit = fit_cpu.to(DEV)
    linear_standard, _coefficient, _info = _complete_degree_one(z[fit], z, action[fit])
    native = action_cpu.to(DEV)
    raw = raw_cpu.to(DEV)
    mean = action_mean.to(DEV).expand(REAL_V, -1)
    linear = linear_standard * action_scale
    quadratic = native - mean - linear
    exact_decomposition_error = float(
        (native - mean - linear - quadratic).norm() / native.norm().clamp_min(1e-20))

    base, instrument = _base_outputs(model, token_ids, mean, linear, quadratic)
    l_attention_effect = base["MLQ"]["attention1"] - base["MQ"]["attention1"]
    l_mlp_effect = base["MLQ"]["mlp1"] - base["MQ"]["mlp1"]
    q_attention_effect = base["MLQ"]["attention1"] - base["ML"]["attention1"]
    q_mlp_effect = base["MLQ"]["mlp1"] - base["ML"]["mlp1"]

    l_donors, l_selection, l_feasible = _donor_maps(
        receiver_ids, donor_ids, raw, linear,
        l_attention_effect, l_mlp_effect, 398)
    q_donors, q_selection, q_feasible = _donor_maps(
        receiver_ids, donor_ids, raw, quadratic,
        q_attention_effect, q_mlp_effect, 1398)
    swaps, swap_batches = _swap_outputs(
        model, receiver_ids, mean, linear, quadratic, l_donors, q_donors)
    del model

    eval_index = receiver_ids
    base_eval = {name: {kind: value[eval_index] for kind, value in by_kind.items()}
                 for name, by_kind in base.items()}
    l_scores, l_rows = _score_component("L", swaps, base_eval, schemes)
    q_scores, q_rows = _score_component("Q", swaps, base_eval, schemes)

    l_attn_cross = l_scores["effect_attention_far"]["mlp1"]["cosine"]
    l_mlp_cross = l_scores["effect_mlp_far"]["attention1"]["cosine"]
    q_attn_cross = q_scores["effect_attention_far"]["mlp1"]["cosine"]
    q_mlp_cross = q_scores["effect_mlp_far"]["attention1"]["cosine"]
    l_random_mlp = l_scores["random"]["mlp1"]["cosine"]
    l_random_attention = l_scores["random"]["attention1"]["cosine"]
    l_raw_mlp = l_scores["raw_nearest"]["mlp1"]["cosine"]
    l_raw_attention = l_scores["raw_nearest"]["attention1"]["cosine"]
    q_random_mlp = q_scores["random"]["mlp1"]["cosine"]
    q_random_attention = q_scores["random"]["attention1"]["cosine"]
    q_raw_mlp = q_scores["raw_nearest"]["mlp1"]["cosine"]
    q_raw_attention = q_scores["raw_nearest"]["attention1"]["cosine"]

    all_far_action_cosines = torch.cat((
        l_selection["attention_far_action_cosine"],
        l_selection["mlp_far_action_cosine"],
        q_selection["attention_far_action_cosine"],
        q_selection["mlp_far_action_cosine"],
    ))
    route_summaries = []
    for component, donors, selection, scores in (
            ("L", l_donors, l_selection, l_scores),
            ("Q", q_donors, q_selection, q_scores)):
        for selector, selector_key in (("effect_attention_far", "attention_selector_cosine"),
                                       ("effect_mlp_far", "mlp_selector_cosine")):
            route_summaries.append({
                "component": component,
                "selector": selector,
                "both_consumer_fraction_ge_08":
                    scores[selector]["fraction_both_consumer_cosine_ge_08"],
                "distinct_donors": int(donors[selector].unique().numel()),
                "median_selector_cosine": float(selection[selector_key].median()),
            })

    pred_a = (
        exact_decomposition_error <= 1e-6
        and instrument["live_full_action_relative_error"] <= 1e-6
        and min(l_feasible, q_feasible) >= .99
        and float(all_far_action_cosines.max()) <= .5001
        and swap_batches > 0
        and all(len(value) == EVAL_V for value in (*l_donors.values(), *q_donors.values())))
    pred_b = (
        min(l_attn_cross, l_mlp_cross) >= .60
        and max(l_attn_cross, l_mlp_cross) >= .75
        and l_attn_cross >= l_random_mlp + .15
        and l_mlp_cross >= l_random_attention + .15
        and l_attn_cross >= l_raw_mlp + .05
        and l_mlp_cross >= l_raw_attention + .05)
    pred_c = (
        min(q_attn_cross, q_mlp_cross) >= .50
        and max(q_attn_cross, q_mlp_cross) >= .70
        and q_attn_cross >= q_random_mlp + .15
        and q_mlp_cross >= q_random_attention + .15
        and q_attn_cross >= q_raw_mlp + .05
        and q_mlp_cross >= q_raw_attention + .05)
    pred_d = any(
        route["both_consumer_fraction_ge_08"] >= .05
        and route["distinct_donors"] >= 100
        and route["median_selector_cosine"] >= .80
        for route in route_summaries)
    cross_cosines = (l_attn_cross, l_mlp_cross, q_attn_cross, q_mlp_cross)
    all_near_random = (
        l_attn_cross <= l_random_mlp + .03
        and l_mlp_cross <= l_random_attention + .03
        and q_attn_cross <= q_random_mlp + .03
        and q_mlp_cross <= q_random_attention + .03)
    strong_null = (
        min(l_feasible, q_feasible) < .50
        or max(cross_cosines) <= .30
        or all_near_random
        or not pred_a)
    supported = bool(pred_a and ((pred_b) or (pred_c and pred_d)) and not strong_null)

    examples = {
        "L_attention_selected": _decode_examples(
            receiver_ids, l_donors["effect_attention_far"], linear,
            l_selection["attention_selector_cosine"], l_rows["effect_attention_far"]),
        "L_mlp_selected": _decode_examples(
            receiver_ids, l_donors["effect_mlp_far"], linear,
            l_selection["mlp_selector_cosine"], l_rows["effect_mlp_far"]),
        "Q_attention_selected": _decode_examples(
            receiver_ids, q_donors["effect_attention_far"], quadratic,
            q_selection["attention_selector_cosine"], q_rows["effect_attention_far"]),
        "Q_mlp_selected": _decode_examples(
            receiver_ids, q_donors["effect_mlp_far"], quadratic,
            q_selection["mlp_selector_cosine"], q_rows["effect_mlp_far"]),
    }
    result = {
        "status": "mlp0_far_action_effect_interchange_complete",
        "rung": 398,
        "claim_level": "token_only_cross_consumer_physical_interchange_not_compression",
        "population": {"real_tokens": REAL_V, "donor_tokens": int(fit_cpu.sum()),
                       "receiver_tokens": int(eval_cpu.sum()), "split": "token_id_mod5"},
        "literal_diagnostic_intervention_values": DIAGNOSTIC_VALUES,
        "executable_compression_claim": False,
        "far_component_cosine_ceiling": FAR_COSINE,
        "instrument": {
            "exact_decomposition_relative_error": exact_decomposition_error,
            **instrument,
            "swap_batches": swap_batches,
            "l_far_candidate_fraction": l_feasible,
            "q_far_candidate_fraction": q_feasible,
            "maximum_selected_far_action_cosine": float(all_far_action_cosines.max()),
        },
        "L_selection": {key: {"median": float(value.median()), "p05": float(value.quantile(.05)),
                               "p95": float(value.quantile(.95))}
                        for key, value in l_selection.items()},
        "Q_selection": {key: {"median": float(value.median()), "p05": float(value.quantile(.05)),
                               "p95": float(value.quantile(.95))}
                        for key, value in q_selection.items()},
        "L_physical_conditional_effect": l_scores,
        "Q_physical_conditional_effect": q_scores,
        "route_summaries": route_summaries,
        "decoded_best_preserved_examples": examples,
        'pred_a_exact_split_far_candidates_and_live_swaps': bool(pred_a),
        'pred_b_far_L_effect_equivalence_cross_validates': bool(pred_b),
        'pred_c_far_Q_effect_equivalence_cross_validates': bool(pred_c),
        'pred_d_equivalence_is_widespread_not_one_donor': bool(pred_d),
        "null_far_effect_equivalence_absent": bool(strong_null),
        "downstream_token_equivalence_supported": supported,
        "next_route": ("semantic_group_validation" if supported
                       else "consumer_aware_quadratic_spectrum"),
        "live_context_transfer_licensed": False,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 FAR-ACTION INTERCHANGE DONE", flush=True)


if __name__ == "__main__":
    main()
