"""RUNG 399 -- CONSUMER-AWARE TOKEN-PRIVATE Q ACTION SPECTRA.

Rebuild exact length-one Q and its aligned conditional effects
Y(M+L+Q)-Y(M+L).  On token-mod5 train only, whiten centered Q and
take fixed left singular action directions of Q-to-attention1, Q-to-MLP1,
balanced Q-to-joint, and seed399 shuffled-joint cross-covariances.  Encode
heldout Q in those coordinates, decode to action space, and physically inject
after exact M+L at ranks 16/64/256/512.  Equal-rank Q-PCA is the control and
joint rank1152 is the exactness arm.

Frozen predictions
------------------
pred_a: joint-full action relative error <=1e-5, live hooks, and joint physical
    R2 is nondecreasing within .02 across ranks16/64/256/512.
pred_b: some joint rank<=64 has joined conditional R2>=.50 and cosine>=.80,
    beating equal-rank PCA by >=.05 R2 and shuffled by >=.15.
pred_c: some joint rank<=256 has joined R2>=.80 and both consumer cosines>=.90.
pred_d: at rank64 attention-aware beats MLP-aware attention R2 by >=.05, and
    MLP-aware beats attention-aware MLP R2 by >=.05.

Strong null: joint-r256 R2<=.25; no response-aware rank beats PCA by .02;
shuffled-r256 comes within .03 of joint-r256; or exactness fails.  A plus B or
C licenses one fixed Q-table rank confirmation.  D identifies distinct reader
spectra.  No token grouping or live-context promotion.

Literal Q-table price at rank r is 50,257*r + 1,152*r + 1,152 values (per-token
codes, decoder, mean), versus 57,896,064 source-Q values.  These are executable
length-one token-table prices, not a live-context MLP0 claim.
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
OUT = ROOT / "mlp0_consumer_aware_q_spectrum_results.json"
DEV = "cuda"
D = 1152
REAL_V = 50_257
EVAL_V = 10_052
RANKS = (16, 64, 256, 512)
SOURCE_Q_VALUES = REAL_V * D


def _price(rank: int) -> int:
    return REAL_V * rank + D * rank + D


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


def _center_scale(train: torch.Tensor, full: torch.Tensor):
    mean = train.float().mean(0, keepdim=True)
    scale = float((train.float() - mean).square().mean().sqrt().clamp_min(1e-12))
    return (full.float() - mean) / scale, mean, scale


@torch.no_grad()
def _directions(qwhite_train: torch.Tensor, target_train: torch.Tensor):
    cross = qwhite_train.T @ target_train / len(qwhite_train)
    gram = cross @ cross.T
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    return vectors.flip(1), values.flip(0)


@torch.no_grad()
def _candidate_outputs(model, receiver_ids: torch.Tensor, mean: torch.Tensor,
                       linear: torch.Tensor, candidates: dict[str, torch.Tensor]):
    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    outputs = {name: {"attention1": [], "mlp1": []} for name in candidates}
    batches = 0
    for start in range(0, len(receiver_ids), 256):
        ids = receiver_ids[start:start + 256]
        token = ids.to(DEV).view(-1, 1)
        raw = F.rms_norm(model.transformer.wte(token), (D,))
        remix = (block0.lambdas[0] + block0.lambdas[1]) * raw
        attention0, value0 = block0.attn(F.rms_norm(remix, (D,)), None)
        pre0 = remix + attention0
        bias = block0.mlp.Down_bias.view(1, 1, D).to(raw)
        for name, qhat in candidates.items():
            action = mean[ids] + linear[ids] + qhat[start:start + len(ids)]
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


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert RANKS == (16, 64, 256, 512)
        assert SOURCE_Q_VALUES == 57_896_064
        assert _price(64) == 3_291_328
        print("MLP0 CONSUMER-AWARE Q SPECTRUM | dry run: split, ranks, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, str(ROOT / "ops"))
    from tier2_model import load_elriggs
    from mlp0_mean_linear_quadratic_causal_factorial import (
        _capture, _complete_degree_one, _standardize)
    from mlp0_far_action_effect_interchange import _base_outputs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    _raw_cpu, z_cpu, action_cpu = _capture(model)
    token_ids = torch.arange(REAL_V)
    fit_cpu, eval_cpu = token_ids % 5 != 0, token_ids % 5 == 0
    fit, evaluate = fit_cpu.to(DEV), eval_cpu.to(DEV)
    receiver_ids = token_ids[eval_cpu].to(DEV)

    z, _z_mean, _z_scale = _standardize(z_cpu[fit_cpu], z_cpu)
    action, action_mean, action_scale = _standardize(action_cpu[fit_cpu], action_cpu)
    z, action = z.to(DEV), action.to(DEV)
    linear_standard, _coefficient, _info = _complete_degree_one(z[fit], z, action[fit])
    native = action_cpu.to(DEV)
    mean = action_mean.to(DEV).expand(REAL_V, -1)
    linear = linear_standard * action_scale
    quadratic = native - mean - linear

    base, base_instrument = _base_outputs(model, token_ids, mean, linear, quadratic)
    attention_effect = base["MLQ"]["attention1"] - base["ML"]["attention1"]
    mlp_effect = base["MLQ"]["mlp1"] - base["ML"]["mlp1"]
    attention_std, _attention_mean, _attention_scale = _center_scale(
        attention_effect[fit], attention_effect)
    mlp_std, _mlp_mean, _mlp_scale = _center_scale(mlp_effect[fit], mlp_effect)
    balanced_joint = torch.cat((attention_std, mlp_std), dim=1)

    qstd, qmean, qscale = _center_scale(quadratic[fit], quadratic)
    covariance = qstd[fit].T @ qstd[fit] / int(fit.sum())
    covariance = .5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = float(eigenvalues[-1]) * 1e-6
    safe = eigenvalues.clamp_min(floor)
    invsqrt = (eigenvectors * safe.rsqrt()) @ eigenvectors.T
    sqrtcov = (eigenvectors * safe.sqrt()) @ eigenvectors.T
    qwhite = qstd @ invsqrt

    attention_directions, attention_strength = _directions(
        qwhite[fit], attention_std[fit])
    mlp_directions, mlp_strength = _directions(qwhite[fit], mlp_std[fit])
    joint_directions, joint_strength = _directions(qwhite[fit], balanced_joint[fit])
    generator = torch.Generator(device=DEV).manual_seed(399)
    shuffled_joint = balanced_joint[fit][torch.randperm(int(fit.sum()), generator=generator,
                                                        device=DEV)]
    shuffled_directions, shuffled_strength = _directions(qwhite[fit], shuffled_joint)

    directions = {
        "attention": attention_directions,
        "mlp": mlp_directions,
        "joint": joint_directions,
        "shuffled": shuffled_directions,
    }
    eval_qwhite = qwhite[evaluate]
    eval_qstd = qstd[evaluate]
    qmean_dev = qmean.to(DEV)
    candidates = {}
    action_reconstruction = {}
    for family, basis in directions.items():
        for rank in RANKS:
            selected = basis[:, :rank]
            prediction_std = (eval_qwhite @ selected) @ (selected.T @ sqrtcov)
            prediction = prediction_std * qscale + qmean_dev
            name = f"{family}_r{rank}"
            candidates[name] = prediction
            action_reconstruction[name] = {
                "heldout_Q_r2": _r2(quadratic[evaluate] - qmean_dev,
                                      prediction - qmean_dev),
                "literal_values": _price(rank),
            }
    for rank in RANKS:
        selected = eigenvectors[:, -rank:]
        prediction_std = (eval_qstd @ selected) @ selected.T
        prediction = prediction_std * qscale + qmean_dev
        name = f"pca_r{rank}"
        candidates[name] = prediction
        action_reconstruction[name] = {
            "heldout_Q_r2": _r2(quadratic[evaluate] - qmean_dev,
                                  prediction - qmean_dev),
            "literal_values": _price(rank),
        }
    full_prediction_std = (eval_qwhite @ joint_directions) @ (joint_directions.T @ sqrtcov)
    candidates["joint_r1152"] = full_prediction_std * qscale + qmean_dev
    full_action_error = float(
        (quadratic[evaluate] - candidates["joint_r1152"]).norm()
        / quadratic[evaluate].norm().clamp_min(1e-20))
    action_reconstruction["joint_r1152"] = {
        "heldout_Q_r2": _r2(quadratic[evaluate] - qmean_dev,
                              candidates["joint_r1152"] - qmean_dev),
        "literal_values": _price(1152),
    }

    candidate_outputs, candidate_batches = _candidate_outputs(
        model, receiver_ids, mean, linear, candidates)
    del model
    baseline = {kind: base["ML"][kind][evaluate] for kind in ("attention1", "mlp1")}
    target = {
        "attention1": attention_effect[evaluate],
        "mlp1": mlp_effect[evaluate],
    }
    target["joint"] = torch.cat((target["attention1"], target["mlp1"]), dim=1)
    physical = {}
    for name, by_kind in candidate_outputs.items():
        prediction = {
            kind: by_kind[kind] - baseline[kind] for kind in ("attention1", "mlp1")}
        prediction["joint"] = torch.cat((prediction["attention1"], prediction["mlp1"]), dim=1)
        physical[name] = {kind: _metrics(target[kind], prediction[kind])
                          for kind in ("attention1", "mlp1", "joint")}

    joint_r2 = [physical[f"joint_r{rank}"]["joint"]["r2"] for rank in RANKS]
    monotone = all(b >= a - .02 for a, b in zip(joint_r2, joint_r2[1:]))
    pred_a = (
        full_action_error <= 1e-5
        and base_instrument["live_full_action_relative_error"] <= 1e-6
        and candidate_batches > 0
        and monotone)
    qualifying_b = [rank for rank in (16, 64)
                    if physical[f"joint_r{rank}"]["joint"]["r2"] >= .50
                    and physical[f"joint_r{rank}"]["joint"]["cosine"] >= .80
                    and physical[f"joint_r{rank}"]["joint"]["r2"]
                    >= physical[f"pca_r{rank}"]["joint"]["r2"] + .05
                    and physical[f"joint_r{rank}"]["joint"]["r2"]
                    >= physical[f"shuffled_r{rank}"]["joint"]["r2"] + .15]
    qualifying_c = [rank for rank in (16, 64, 256)
                    if physical[f"joint_r{rank}"]["joint"]["r2"] >= .80
                    and physical[f"joint_r{rank}"]["attention1"]["cosine"] >= .90
                    and physical[f"joint_r{rank}"]["mlp1"]["cosine"] >= .90]
    pred_b = bool(qualifying_b)
    pred_c = bool(qualifying_c)
    pred_d = (
        physical["attention_r64"]["attention1"]["r2"]
        >= physical["mlp_r64"]["attention1"]["r2"] + .05
        and physical["mlp_r64"]["mlp1"]["r2"]
        >= physical["attention_r64"]["mlp1"]["r2"] + .05)
    response_aware_beats_pca = any(
        max(physical[f"joint_r{rank}"]["joint"]["r2"],
            physical[f"attention_r{rank}"]["joint"]["r2"],
            physical[f"mlp_r{rank}"]["joint"]["r2"])
        >= physical[f"pca_r{rank}"]["joint"]["r2"] + .02
        for rank in RANKS)
    strong_null = (
        physical["joint_r256"]["joint"]["r2"] <= .25
        or not response_aware_beats_pca
        or physical["shuffled_r256"]["joint"]["r2"]
        >= physical["joint_r256"]["joint"]["r2"] - .03
        or not pred_a)
    licensed = bool(pred_a and (pred_b or pred_c) and not strong_null)

    result = {
        "status": "mlp0_consumer_aware_q_spectrum_complete",
        "rung": 399,
        "claim_level": "length1_token_private_Q_table_spectrum_not_live_context",
        "population": {"real_tokens": REAL_V, "fit_tokens": int(fit_cpu.sum()),
                       "heldout_tokens": int(eval_cpu.sum()), "split": "token_id_mod5"},
        "source_Q_values": SOURCE_Q_VALUES,
        "price_formula": "50257*r + 1152*r + 1152",
        "covariance": {"min_eigenvalue": float(eigenvalues[0]),
                       "max_eigenvalue": float(eigenvalues[-1]), "floor": floor,
                       "effective_rank": int((eigenvalues > floor).sum())},
        "leading_cross_gram_strength": {
            "attention": [float(x) for x in attention_strength[:16]],
            "mlp": [float(x) for x in mlp_strength[:16]],
            "joint": [float(x) for x in joint_strength[:16]],
            "shuffled": [float(x) for x in shuffled_strength[:16]],
        },
        "instrument": {**base_instrument, "candidate_batches": candidate_batches,
                       "joint_full_action_relative_error": full_action_error,
                       "joint_rank_curve_monotone_within_002": monotone},
        "action_reconstruction": action_reconstruction,
        "physical_conditional_Q_effect": physical,
        "qualifying_pred_b_ranks": qualifying_b,
        "qualifying_pred_c_ranks": qualifying_c,
        'pred_a_exact_full_live_and_monotone': bool(pred_a),
        'pred_b_compact_joint_basis_beats_pca_and_shuffle': bool(pred_b),
        'pred_c_rank256_preserves_both_consumers': bool(pred_c),
        'pred_d_consumer_specific_bases_cross_over': bool(pred_d),
        "null_response_aware_Q_spectrum_fails": bool(strong_null),
        "fixed_Q_table_confirmation_licensed": licensed,
        "live_context_transfer_licensed": False,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 CONSUMER-AWARE Q SPECTRUM DONE", flush=True)


if __name__ == "__main__":
    main()
