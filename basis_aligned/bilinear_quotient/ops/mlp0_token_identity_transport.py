"""RUNG 395 -- EXACT-TOKEN IDENTITY TRANSPORT IN THE MLP0 WRITE.

Rung394 rejected one small sparse code for the joined attention1+MLP1 response.
This changes the object instead of tuning that code.  Enumerate every length-one
token path and ask whether MLP0's dominant bias-free token write is a transformed
representation of the raw normalized token vector.

Fit full linear maps with intercept in both directions on token_id mod5 != 0 and
score the untouched fifth.  Decode predicted raw vectors against all 50,257 token
embeddings.  Then inject the heldout raw->write prediction (identity component)
and its exact residual separately through native block1, preserving attention0,
the raw x0 reinjection, MLP0 bias, and all native downstream weights.  Shuffled
token pairing is a live negative.  Full-rank linear maps are diagnostic objects,
not an executable compression or a claim that MLP0 is algebraically linear.

Frozen predictions
------------------
pred_a: write->raw heldout R2>=.50, exact-token retrieval top1>=.20 and top5>=.40;
    shuffled-pair top1<=.01.
pred_b: raw->write heldout R2>=.50 and orthogonal-Procrustes mean heldout cosine
    >=.40.  This is the transformed-token-identity prediction.
pred_c: injecting only the fitted identity component reproduces native removal
    responses at R2>=.60 for attention1, >=.40 for MLP1, and >=.45 jointly.
pred_d: the write residual has participation rank <=.75 of the native write and
    residual-only joint response norm <=.75 of native.  This predicts a smaller
    shared-feature remainder after token identity is removed.

Strong null: write->raw R2<=.15 and retrieval top1<=.01, or identity-only joint
response R2<=.15, or shuffled retrieval top1>.05.  A/B/C identify a private token
transport even if D fails; A/B/C/D license a residual-feature program.  Failure
pivots to the exact quadratic token kernel rather than sparse-rank tuning.
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
OUT = ROOT / "mlp0_token_identity_transport_results.json"
DEV = "cuda"
D = 1152
REAL_V = 50_257
RIDGE = 1.0


def _standardize(train: torch.Tensor, full: torch.Tensor):
    mean = train.float().mean(0, keepdim=True)
    scale = float((train.float() - mean).square().mean().sqrt().clamp_min(1e-12))
    return (full.float() - mean) / scale, mean, scale


@torch.no_grad()
def _ridge(train_source: torch.Tensor, train_target: torch.Tensor) -> torch.Tensor:
    design = torch.cat((train_source, torch.ones(len(train_source), 1, device=DEV)), dim=1)
    penalty = RIDGE * torch.eye(design.shape[1], device=DEV)
    penalty[-1, -1] = 0
    return torch.linalg.solve(design.T @ design + penalty, design.T @ train_target)


@torch.no_grad()
def _predict(source: torch.Tensor, map_: torch.Tensor) -> torch.Tensor:
    design = torch.cat((source, torch.ones(len(source), 1, device=DEV)), dim=1)
    return design @ map_


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float(1 - (target - prediction).square().sum()
                 / target.square().sum().clamp_min(1e-12))


def _cosine_and_norm(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    flat_target, flat_prediction = target.flatten(), prediction.flatten()
    cosine = float(torch.dot(flat_target, flat_prediction)
                   / (flat_target.norm() * flat_prediction.norm()).clamp_min(1e-12))
    return {
        "r2": _r2(target, prediction),
        "cosine": cosine,
        "norm_ratio": float(flat_prediction.norm() / flat_target.norm().clamp_min(1e-12)),
    }


@torch.no_grad()
def _retrieval(predicted_raw: torch.Tensor, candidate_raw: torch.Tensor,
               token_ids: torch.Tensor) -> dict[str, float]:
    predicted = F.normalize(predicted_raw.float(), dim=1)
    candidates = F.normalize(candidate_raw.float(), dim=1)
    top1 = top5 = reciprocal = 0.0
    for start in range(0, len(predicted), 256):
        query = predicted[start:start + 256]
        truth = token_ids[start:start + len(query)]
        similarity = query @ candidates.T
        top = similarity.topk(5, dim=1).indices
        top1 += float((top[:, 0] == truth).sum())
        top5 += float((top == truth[:, None]).any(1).sum())
        correct = similarity[torch.arange(len(query), device=DEV), truth]
        rank = (similarity > correct[:, None]).sum(1) + 1
        reciprocal += float(rank.float().reciprocal().sum())
    return {
        "top1_accuracy": top1 / len(predicted),
        "top5_accuracy": top5 / len(predicted),
        "mean_reciprocal_rank": reciprocal / len(predicted),
    }


@torch.no_grad()
def _capture_population(model):
    block0 = model.transformer.h[0]
    xs, actions = [], []
    for start in range(0, REAL_V, 512):
        token = torch.arange(start, min(start + 512, REAL_V), device=DEV).view(-1, 1)
        x0 = F.rms_norm(model.transformer.wte(token), (D,))
        remix0 = (block0.lambdas[0] + block0.lambdas[1]) * x0
        attention0, _value0 = block0.attn(F.rms_norm(remix0, (D,)), None)
        pre_mlp0 = remix0 + attention0
        write0 = block0.mlp(F.rms_norm(pre_mlp0, (D,)))
        action0 = write0 - block0.mlp.Down_bias.view(1, 1, D).to(write0)
        xs.append(x0[:, 0].float().cpu().half())
        actions.append(action0[:, 0].float().cpu().half())
    return torch.cat(xs), torch.cat(actions)


@torch.no_grad()
def _heldout_responses(model, token_ids: torch.Tensor,
                       predicted_action: torch.Tensor) -> dict[str, torch.Tensor]:
    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    output = {name: [] for name in (
        "native_attention", "native_mlp", "identity_attention", "identity_mlp",
        "residual_attention", "residual_mlp", "removed_attention", "removed_mlp")}
    for start in range(0, len(token_ids), 512):
        token = token_ids[start:start + 512].to(DEV).view(-1, 1)
        x0 = F.rms_norm(model.transformer.wte(token), (D,))
        remix0 = (block0.lambdas[0] + block0.lambdas[1]) * x0
        attention0, value0 = block0.attn(F.rms_norm(remix0, (D,)), None)
        pre_mlp0 = remix0 + attention0
        write0 = block0.mlp(F.rms_norm(pre_mlp0, (D,)))
        bias = block0.mlp.Down_bias.view(1, 1, D).to(write0)
        native_action = write0 - bias
        identity_action = predicted_action[start:start + len(token)].to(DEV)[:, None]
        residual_action = native_action - identity_action

        def response(action: torch.Tensor):
            post0 = pre_mlp0 + bias + action
            remixed1 = block1.lambdas[0] * post0 + block1.lambdas[1] * x0
            attention1, _ = block1.attn(F.rms_norm(remixed1, (D,)), value0)
            write1 = block1.mlp(F.rms_norm(remixed1 + attention1, (D,)))
            return attention1[:, 0].float().cpu(), write1[:, 0].float().cpu()

        for name, action in (
            ("native", native_action), ("identity", identity_action),
            ("residual", residual_action), ("removed", torch.zeros_like(native_action))):
            attention, mlp = response(action)
            output[f"{name}_attention"].append(attention)
            output[f"{name}_mlp"].append(mlp)
    return {key: torch.cat(value).to(DEV) for key, value in output.items()}


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert REAL_V == 50_257 and D == 1152 and RIDGE == 1.0
        print("MLP0 TOKEN IDENTITY TRANSPORT | dry run: split, maps, injection, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mlp0_exact_token_downstream_sparse_code import _spectrum
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    raw_cpu, action_cpu = _capture_population(model)
    token_ids_cpu = torch.arange(REAL_V)
    fit_cpu = token_ids_cpu % 5 != 0
    eval_cpu = ~fit_cpu
    raw, raw_mean, raw_scale = _standardize(raw_cpu[fit_cpu], raw_cpu)
    action, action_mean, action_scale = _standardize(action_cpu[fit_cpu], action_cpu)
    raw, action = raw.to(DEV), action.to(DEV)
    fit, evaluate = fit_cpu.to(DEV), eval_cpu.to(DEV)
    train_raw, eval_raw = raw[fit], raw[evaluate]
    train_action, eval_action = action[fit], action[evaluate]

    raw_to_action = _ridge(train_raw, train_action)
    action_to_raw = _ridge(train_action, train_raw)
    predicted_action_norm = _predict(eval_raw, raw_to_action)
    predicted_raw_norm = _predict(eval_action, action_to_raw)
    forward_r2 = _r2(eval_action, predicted_action_norm)
    reverse_r2 = _r2(eval_raw, predicted_raw_norm)

    cross = train_raw.T @ train_action
    u, _singular, vh = torch.linalg.svd(cross, full_matrices=False)
    orthogonal = u @ vh
    procrustes_prediction = eval_raw @ orthogonal
    procrustes_cosines = F.cosine_similarity(procrustes_prediction, eval_action, dim=1)
    procrustes = {
        "heldout_r2": _r2(eval_action, procrustes_prediction),
        "heldout_mean_cosine": float(procrustes_cosines.mean()),
        "heldout_median_cosine": float(procrustes_cosines.median()),
    }

    predicted_raw_actual = predicted_raw_norm * raw_scale + raw_mean.to(DEV)
    all_raw_actual = raw * raw_scale + raw_mean.to(DEV)
    eval_token_ids = token_ids_cpu[eval_cpu].to(DEV)
    retrieval = _retrieval(predicted_raw_actual, all_raw_actual, eval_token_ids)

    generator = torch.Generator(device=DEV).manual_seed(395)
    permutation = torch.randperm(len(train_action), generator=generator, device=DEV)
    shuffled_action_to_raw = _ridge(train_action[permutation], train_raw)
    shuffled_prediction = _predict(eval_action, shuffled_action_to_raw)
    shuffled_actual = shuffled_prediction * raw_scale + raw_mean.to(DEV)
    shuffled_retrieval = _retrieval(shuffled_actual, all_raw_actual, eval_token_ids)
    shuffled_reverse_r2 = _r2(eval_raw, shuffled_prediction)

    predicted_action_actual = predicted_action_norm * action_scale + action_mean.to(DEV)
    responses = _heldout_responses(model, token_ids_cpu[eval_cpu], predicted_action_actual.cpu())
    del model
    torch.cuda.empty_cache()
    native_attention = responses["native_attention"] - responses["removed_attention"]
    native_mlp = responses["native_mlp"] - responses["removed_mlp"]
    identity_attention = responses["identity_attention"] - responses["removed_attention"]
    identity_mlp = responses["identity_mlp"] - responses["removed_mlp"]
    residual_attention = responses["residual_attention"] - responses["removed_attention"]
    residual_mlp = responses["residual_mlp"] - responses["removed_mlp"]
    native_joint = torch.cat((native_attention, native_mlp), dim=1)
    identity_joint = torch.cat((identity_attention, identity_mlp), dim=1)
    residual_joint = torch.cat((residual_attention, residual_mlp), dim=1)
    causal = {
        "identity_attention1": _cosine_and_norm(native_attention, identity_attention),
        "identity_mlp1": _cosine_and_norm(native_mlp, identity_mlp),
        "identity_joint": _cosine_and_norm(native_joint, identity_joint),
        "residual_attention1": _cosine_and_norm(native_attention, residual_attention),
        "residual_mlp1": _cosine_and_norm(native_mlp, residual_mlp),
        "residual_joint": _cosine_and_norm(native_joint, residual_joint),
    }

    residual_action = eval_action - predicted_action_norm
    native_spectrum = _spectrum(eval_action)
    residual_spectrum = _spectrum(residual_action)
    residual_norm_ratio = float(residual_joint.norm() / native_joint.norm().clamp_min(1e-12))
    decomposition = {
        "native_write_spectrum": native_spectrum,
        "residual_write_spectrum": residual_spectrum,
        "residual_to_native_participation_rank_ratio": (
            residual_spectrum["participation_rank"] / native_spectrum["participation_rank"]),
        "residual_only_joint_response_norm_ratio": residual_norm_ratio,
        "identity_plus_residual_action_reconstruction_max_error": float(
            (predicted_action_norm + residual_action - eval_action).abs().max()),
    }

    pred_a = (
        reverse_r2 >= .50 and retrieval["top1_accuracy"] >= .20
        and retrieval["top5_accuracy"] >= .40 and shuffled_retrieval["top1_accuracy"] <= .01)
    pred_b = forward_r2 >= .50 and procrustes["heldout_mean_cosine"] >= .40
    pred_c = (
        causal["identity_attention1"]["r2"] >= .60
        and causal["identity_mlp1"]["r2"] >= .40
        and causal["identity_joint"]["r2"] >= .45)
    pred_d = (
        decomposition["residual_to_native_participation_rank_ratio"] <= .75
        and residual_norm_ratio <= .75)
    strong_null = (
        (reverse_r2 <= .15 and retrieval["top1_accuracy"] <= .01)
        or causal["identity_joint"]["r2"] <= .15
        or shuffled_retrieval["top1_accuracy"] > .05)
    private_identity_identified = bool(pred_a and pred_b and pred_c and not strong_null)
    residual_feature_licensed = bool(private_identity_identified and pred_d)
    result = {
        "status": "mlp0_token_identity_transport_complete",
        "rung": 395,
        "claim_level": "exhaustive_token_private_identity_transport_and_causal_decomposition_screen",
        "population": {
            "real_tokens": REAL_V, "fit_tokens": int(fit_cpu.sum()),
            "heldout_tokens": int(eval_cpu.sum()), "split": "token_id_mod5",
        },
        "maps": {
            "type": "full linear with intercept and ridge=1; diagnostic, not compressed",
            "raw_to_write_heldout_r2": forward_r2,
            "write_to_raw_heldout_r2": reverse_r2,
            "shuffled_write_to_raw_heldout_r2": shuffled_reverse_r2,
            "orthogonal_procrustes": procrustes,
            "stored_scalars_per_direction": (D + 1) * D,
        },
        "token_retrieval_from_mlp0_write": retrieval,
        "shuffled_pair_token_retrieval": shuffled_retrieval,
        "causal_component_injection": causal,
        "identity_residual_decomposition": decomposition,
        'pred_a_mlp0_write_linearly_recovers_exact_token_identity': bool(pred_a),
        'pred_b_raw_token_linearly_predicts_mlp0_write': bool(pred_b),
        'pred_c_identity_component_carries_downstream_response': bool(pred_c),
        'pred_d_residual_is_smaller_shared_feature_target': bool(pred_d),
        "null_no_token_identity_transport": bool(strong_null),
        "private_token_identity_transport_identified": private_identity_identified,
        "residual_feature_program_licensed": residual_feature_licensed,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 TOKEN IDENTITY TRANSPORT DONE", flush=True)


if __name__ == "__main__":
    main()
