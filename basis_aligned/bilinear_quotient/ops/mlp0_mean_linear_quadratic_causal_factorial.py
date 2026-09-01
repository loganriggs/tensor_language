"""RUNG 397 -- EXACT MLP0 MEAN/LINEAR/QUADRATIC TOKEN CAUSAL FACTORIAL.

Repair rung396's shared-mean confound without choosing another rank.  On the
unchanged exhaustive length-one population and token-id-mod5 split, decompose
the heldout bias-free MLP0 action exactly as

    F = M + L + Q,

where M is the training-action mean, L is the complete rank-1152 canonical
degree-one projection from the exact normalized MLP0 input z, and Q is the
exact heldout residual.  Inject all eight subsets through the same native
block1 frame and compute attention1/MLP1 vector Mobius terms.  Fixed heldout
permutations of L, Q, and the paired L+Q are wrong-token controls.

Frozen predictions
------------------
pred_a: decomposition and live-full action relative errors <=1e-6, full-arm
    response R2>=.99999, every component is non-inert, and maximum vector
    Mobius closure error <=1e-5.
pred_b: aligned L after M recovers >=.20 of joined and >=.35 of attention1
    error left by M, and exceeds shuffled-L recovery by >=.15 for both.
pred_c: aligned Q after M recovers >=.20 of joined error and exceeds shuffled-Q
    recovery by >=.15, OR the joined LQ+MLQ interaction norm is >=.20 of the
    native joined response.
pred_d: attention1 and MLP1 conditional recoveries differ by >=.10 for L or Q.

Strong null: the paired wrong-token full action reaches joined response
R2>=.95; or both aligned L and Q gains are within .05 of their shuffled gains
for both consumers; or an exactness/instrument tripwire fails.  This assay
cannot license live context.  B routes to consumer-effect token interchange;
Q/interaction dominance routes to consumer-aware quadratic spectra.

Literal diagnostic price if materialized: 12,909,314 float values = M(1,152)
+ dense L map(1,327,104) + z mean(1,152) + two scalar scales + heldout exact
Q table(10,052*1,152).  Q is heldout lookup, so this is causal attribution,
explicitly not an executable compression claim.
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
OUT = ROOT / "mlp0_mean_linear_quadratic_causal_factorial_results.json"
DEV = "cuda"
D = 1152
REAL_V = 50_257
EVAL_V = 10_052
DIAGNOSTIC_VALUES = 12_909_314
ARM_MASKS = {
    "none": 0,
    "M": 1,
    "L": 2,
    "ML": 3,
    "Q": 4,
    "MQ": 5,
    "LQ": 6,
    "MLQ": 7,
}


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float(1 - (target - prediction).square().sum()
                 / target.square().sum().clamp_min(1e-20))


def _relative_error(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float((target - prediction).norm() / target.norm().clamp_min(1e-20))


def _response_metrics(target: torch.Tensor, prediction: torch.Tensor):
    a, b = target.flatten(), prediction.flatten()
    return {
        "r2": _r2(target, prediction),
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-20)),
        "norm_ratio": float(b.norm() / a.norm().clamp_min(1e-20)),
        "relative_error": _relative_error(target, prediction),
    }


def _error_recovery(target: torch.Tensor, before: torch.Tensor,
                    after: torch.Tensor) -> float:
    before_sse = (target - before).square().sum().clamp_min(1e-20)
    after_sse = (target - after).square().sum()
    return float(1 - after_sse / before_sse)


def _standardize(train: torch.Tensor, full: torch.Tensor):
    mean = train.float().mean(0, keepdim=True)
    scale = float((train.float() - mean).square().mean().sqrt().clamp_min(1e-12))
    return (full.float() - mean) / scale, mean, scale


@torch.no_grad()
def _capture(model):
    block0 = model.transformer.h[0]
    raw_rows, z_rows, action_rows = [], [], []
    for start in range(0, REAL_V, 256):
        token = torch.arange(start, min(start + 256, REAL_V), device=DEV).view(-1, 1)
        raw = F.rms_norm(model.transformer.wte(token), (D,))
        remix = (block0.lambdas[0] + block0.lambdas[1]) * raw
        attention, _value = block0.attn(F.rms_norm(remix, (D,)), None)
        z = F.rms_norm(remix + attention, (D,))
        write = block0.mlp(z)
        action = write - block0.mlp.Down_bias.view(1, 1, D).to(write)
        raw_rows.append(raw[:, 0].float().cpu())
        z_rows.append(z[:, 0].float().cpu())
        action_rows.append(action[:, 0].float().cpu())
    return torch.cat(raw_rows), torch.cat(z_rows), torch.cat(action_rows)


@torch.no_grad()
def _complete_degree_one(train_z: torch.Tensor, eval_z: torch.Tensor,
                         train_action: torch.Tensor):
    covariance = train_z.T @ train_z / len(train_z)
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    floor = float(values[-1]) * 1e-6
    inverse = (vectors * values.clamp_min(floor).reciprocal()) @ vectors.T
    cross = train_z.T @ train_action / len(train_z)
    coefficient = inverse @ cross
    return eval_z @ coefficient, coefficient, {
        "covariance_min_eigenvalue": float(values[0]),
        "covariance_max_eigenvalue": float(values[-1]),
        "covariance_floor": floor,
        "effective_covariance_rank": int((values > floor).sum()),
    }


@torch.no_grad()
def _causal_outputs(model, token_ids: torch.Tensor,
                    actions: dict[str, torch.Tensor]):
    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    outputs = {name: {"attention1": [], "mlp1": []} for name in actions}
    live_full_numerator = live_full_denominator = 0.0
    batch_count = 0
    for start in range(0, len(token_ids), 256):
        token = token_ids[start:start + 256].to(DEV).view(-1, 1)
        raw = F.rms_norm(model.transformer.wte(token), (D,))
        remix = (block0.lambdas[0] + block0.lambdas[1]) * raw
        attention0, value0 = block0.attn(F.rms_norm(remix, (D,)), None)
        pre0 = remix + attention0
        z = F.rms_norm(pre0, (D,))
        write = block0.mlp(z)
        bias = block0.mlp.Down_bias.view(1, 1, D).to(write)
        native_action = write - bias
        supplied_full = actions["MLQ"][start:start + len(token)].to(DEV)[:, None]
        live_full_numerator += float((native_action - supplied_full).float().square().sum())
        live_full_denominator += float(native_action.float().square().sum())

        for name, full_action in actions.items():
            action = full_action[start:start + len(token)].to(DEV)[:, None]
            post0 = pre0 + bias + action
            remixed1 = block1.lambdas[0] * post0 + block1.lambdas[1] * raw
            attention1, _ = block1.attn(F.rms_norm(remixed1, (D,)), value0)
            mlp1 = block1.mlp(F.rms_norm(remixed1 + attention1, (D,)))
            outputs[name]["attention1"].append(attention1[:, 0].float().cpu())
            outputs[name]["mlp1"].append(mlp1[:, 0].float().cpu())
        batch_count += 1
    combined = {
        name: {kind: torch.cat(parts).to(DEV) for kind, parts in by_kind.items()}
        for name, by_kind in outputs.items()
    }
    return combined, {
        "batches": batch_count,
        "live_full_action_relative_error":
            (live_full_numerator / max(live_full_denominator, 1e-30)) ** .5,
    }


def _responses(outputs: dict[str, dict[str, torch.Tensor]]):
    result = {}
    for name, kinds in outputs.items():
        attention = kinds["attention1"] - outputs["none"]["attention1"]
        mlp = kinds["mlp1"] - outputs["none"]["mlp1"]
        result[name] = {
            "attention1": attention,
            "mlp1": mlp,
            "joint": torch.cat((attention, mlp), dim=1),
        }
    return result


def _mobius(responses: dict[str, dict[str, torch.Tensor]]):
    by_mask = {mask: responses[name] for name, mask in ARM_MASKS.items()}
    terms = {}
    for mask in range(1, 8):
        label = "".join(letter for bit, letter in ((1, "M"), (2, "L"), (4, "Q"))
                        if mask & bit)
        terms[label] = {}
        for kind in ("attention1", "mlp1", "joint"):
            value = torch.zeros_like(by_mask[0][kind])
            subset = mask
            while True:
                sign = -1 if ((mask.bit_count() - subset.bit_count()) % 2) else 1
                value = value + sign * by_mask[subset][kind]
                if subset == 0:
                    break
                subset = (subset - 1) & mask
            terms[label][kind] = value

    closure = {}
    for mask in range(8):
        for kind in ("attention1", "mlp1", "joint"):
            reconstructed = torch.zeros_like(by_mask[mask][kind])
            for term_mask in range(1, 8):
                if term_mask & mask == term_mask:
                    label = "".join(letter for bit, letter in ((1, "M"), (2, "L"), (4, "Q"))
                                    if term_mask & bit)
                    reconstructed += terms[label][kind]
            closure[f"{mask}_{kind}"] = _relative_error(
                by_mask[mask][kind], reconstructed) if mask else float(reconstructed.norm())
    return terms, closure


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert set(ARM_MASKS.values()) == set(range(8))
        assert EVAL_V == sum(i % 5 == 0 for i in range(REAL_V))
        assert DIAGNOSTIC_VALUES == D + D * D + D + 2 + EVAL_V * D
        print("MLP0 M/L/Q FACTORIAL | dry run: split, arms, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    _raw_cpu, z_cpu, action_cpu = _capture(model)
    token_ids = torch.arange(REAL_V)
    fit_cpu, eval_cpu = token_ids % 5 != 0, token_ids % 5 == 0
    assert int(eval_cpu.sum()) == EVAL_V

    z, z_mean, z_scale = _standardize(z_cpu[fit_cpu], z_cpu)
    action, action_mean_std, action_scale = _standardize(action_cpu[fit_cpu], action_cpu)
    z, action = z.to(DEV), action.to(DEV)
    fit, evaluate = fit_cpu.to(DEV), eval_cpu.to(DEV)
    l_standard, coefficient, degree1_info = _complete_degree_one(
        z[fit], z[evaluate], action[fit])

    native = action_cpu[eval_cpu].to(DEV)
    mean = action_mean_std.to(DEV).expand(EVAL_V, -1)
    linear = l_standard * action_scale
    quadratic = native - mean - linear
    reconstruction = mean + linear + quadratic
    factorization_error = _relative_error(native, reconstruction)

    generator = torch.Generator(device=DEV).manual_seed(397)
    permutation = torch.randperm(EVAL_V, generator=generator, device=DEV)
    linear_shuffled = linear[permutation]
    quadratic_shuffled = quadratic[permutation]
    zero = torch.zeros_like(native)
    actions = {
        "none": zero,
        "M": mean,
        "L": linear,
        "ML": mean + linear,
        "Q": quadratic,
        "MQ": mean + quadratic,
        "LQ": linear + quadratic,
        "MLQ": reconstruction,
        "M_Lsh": mean + linear_shuffled,
        "M_Qsh": mean + quadratic_shuffled,
        "M_LQsh": mean + linear_shuffled + quadratic_shuffled,
        "ML_Qsh": mean + linear + quadratic_shuffled,
        "MQ_Lsh": mean + quadratic + linear_shuffled,
    }
    outputs, instrument = _causal_outputs(model, token_ids[eval_cpu], actions)
    del model
    responses = _responses(outputs)
    target = responses["MLQ"]

    arm_metrics = {
        name: {kind: _response_metrics(target[kind], response[kind])
               for kind in ("attention1", "mlp1", "joint")}
        for name, response in responses.items()
    }
    conditional = {}
    for kind in ("attention1", "mlp1", "joint"):
        conditional[kind] = {
            "L_after_M": _error_recovery(target[kind], responses["M"][kind],
                                           responses["ML"][kind]),
            "shuffled_L_after_M": _error_recovery(
                target[kind], responses["M"][kind], responses["M_Lsh"][kind]),
            "Q_after_M": _error_recovery(target[kind], responses["M"][kind],
                                           responses["MQ"][kind]),
            "shuffled_Q_after_M": _error_recovery(
                target[kind], responses["M"][kind], responses["M_Qsh"][kind]),
        }
        conditional[kind]["L_alignment_margin"] = (
            conditional[kind]["L_after_M"] - conditional[kind]["shuffled_L_after_M"])
        conditional[kind]["Q_alignment_margin"] = (
            conditional[kind]["Q_after_M"] - conditional[kind]["shuffled_Q_after_M"])

    mobius_vectors, closure = _mobius(responses)
    mobius_metrics = {
        label: {kind: _response_metrics(target[kind], vector)
                for kind, vector in vectors.items()}
        for label, vectors in mobius_vectors.items()
    }
    joined_lq_interaction = (
        mobius_vectors["LQ"]["joint"] + mobius_vectors["MLQ"]["joint"])
    joined_lq_interaction_norm = float(
        joined_lq_interaction.norm() / target["joint"].norm().clamp_min(1e-20))

    action_rms = {
        "M": float(mean.square().mean().sqrt()),
        "L": float(linear.square().mean().sqrt()),
        "Q": float(quadratic.square().mean().sqrt()),
    }
    response_rms = {
        name: {kind: float(value.square().mean().sqrt())
               for kind, value in responses[name].items()}
        for name in ("M", "L", "Q")
    }
    non_inert = all(value > 1e-5 for value in action_rms.values()) and all(
        value > 1e-5 for by_kind in response_rms.values() for value in by_kind.values())
    max_closure_error = max(closure.values())
    full_response_r2 = arm_metrics["MLQ"]["joint"]["r2"]

    pred_a = (
        factorization_error <= 1e-6
        and instrument["live_full_action_relative_error"] <= 1e-6
        and full_response_r2 >= .99999
        and non_inert
        and max_closure_error <= 1e-5)
    pred_b = (
        conditional["joint"]["L_after_M"] >= .20
        and conditional["attention1"]["L_after_M"] >= .35
        and conditional["joint"]["L_alignment_margin"] >= .15
        and conditional["attention1"]["L_alignment_margin"] >= .15)
    pred_c = (
        (conditional["joint"]["Q_after_M"] >= .20
         and conditional["joint"]["Q_alignment_margin"] >= .15)
        or joined_lq_interaction_norm >= .20)
    pred_d = (
        abs(conditional["attention1"]["L_after_M"]
            - conditional["mlp1"]["L_after_M"]) >= .10
        or abs(conditional["attention1"]["Q_after_M"]
               - conditional["mlp1"]["Q_after_M"]) >= .10)
    no_aligned_specificity = all(
        conditional[kind][f"{component}_alignment_margin"] <= .05
        for kind in ("attention1", "mlp1") for component in ("L", "Q"))
    exactness_failure = not pred_a
    strong_null = (
        arm_metrics["M_LQsh"]["joint"]["r2"] >= .95
        or no_aligned_specificity
        or exactness_failure)

    result = {
        "status": "mlp0_mean_linear_quadratic_causal_factorial_complete",
        "rung": 397,
        "claim_level": "exact_token_only_causal_attribution_not_compression",
        "population": {
            "real_tokens": REAL_V,
            "fit_tokens": int(fit_cpu.sum()),
            "heldout_tokens": int(eval_cpu.sum()),
            "split": "token_id_mod5",
        },
        "definition": "F=M+L+Q; M=train mean, L=complete degree-one z projection, Q=exact heldout residual",
        "literal_diagnostic_values_if_materialized": DIAGNOSTIC_VALUES,
        "executable_compression_claim": False,
        "fit": {
            "z_mean_shape": list(z_mean.shape),
            "coefficient_shape": list(coefficient.shape),
            "z_scale": z_scale,
            "action_scale": action_scale,
            **degree1_info,
        },
        "exactness_and_instrument": {
            "factorization_relative_error": factorization_error,
            **instrument,
            "non_inert": non_inert,
            "component_action_rms": action_rms,
            "component_response_rms": response_rms,
            "max_mobius_closure_relative_error": max_closure_error,
        },
        "write_geometry": {
            "degree_one_centered_write_r2": _r2(native - mean, linear),
            "M_norm_ratio": float(mean.norm() / native.norm()),
            "L_norm_ratio": float(linear.norm() / native.norm()),
            "Q_norm_ratio": float(quadratic.norm() / native.norm()),
        },
        "arm_response_metrics": arm_metrics,
        "conditional_error_recovery": conditional,
        "mobius_term_metrics": mobius_metrics,
        "joined_LQ_plus_MLQ_interaction_norm_ratio": joined_lq_interaction_norm,
        'pred_a_exact_decomposition_and_instruments_hold': bool(pred_a),
        'pred_b_degree_one_has_aligned_increment_after_mean': bool(pred_b),
        'pred_c_quadratic_or_interaction_has_distinct_role': bool(pred_c),
        'pred_d_attention_and_mlp_consumer_roles_split': bool(pred_d),
        "null_wrong_token_action_is_equivalent_or_no_specificity": bool(strong_null),
        "next_route": (
            "consumer_effect_token_interchange" if pred_b and not strong_null
            else "consumer_aware_quadratic_spectrum"),
        "live_context_transfer_licensed": False,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 M/L/Q FACTORIAL DONE", flush=True)


if __name__ == "__main__":
    main()
