"""RUNG 396 -- EXACT QUADRATIC TOKEN KERNEL: DEGREE-ONE CAUSAL RANK CURVE.

For every length-one token, capture raw x0, the exact normalized input z to
MLP0, and MLP0's bias-free quadratic write F(z).  Under the empirical token
measure, project F(z) onto constants plus degree-one functions of z.  This is
the canonical empirical orthogonal-polynomial split of the exact bilinear
layer, unlike rung395's map from the earlier raw-x0 coordinate.

Fit only token_id mod5 != 0.  In the z covariance metric, take the complete
reduced-rank curve r=16..1152 of the degree-one projection and inject every
heldout prediction through native block1.  Compare the same raw-x0 curve and
a shuffled z->write r256 negative.  Prices include both rank factors and both
means.  This is token-only causal identification, not live-context adoption.

Frozen predictions
------------------
pred_a: the full z degree-one projection has heldout write R2>=.65 and beats
    the full raw-x0 degree-one projection by >=.15.
pred_b: some z rank <=128 reproduces the joined attention1+MLP1 causal response
    at R2>=.90.
pred_c: some z rank <=256 permits a heldout linear decoder to retrieve exact
    token identity at top1>=.80 against all 50,257 raw token vectors.
pred_d: joined response R2 is nondecreasing within .02 from r128->256->512,
    and shuffled-z r256 joined response R2<=.25.

Strong null: full z degree-one joined response R2<=.50, z-r512 response R2<=.50,
or shuffled r256 comes within .05 of real r256.  Full pass identifies a compact
degree-one token kernel and licenses one live TT transfer.  A miss means the
causal token code needs the orthogonal quadratic remainder; do not tune ranks.
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
OUT = ROOT / "mlp0_quadratic_token_degree1_rank_curve_results.json"
DEV = "cuda"
D = 1152
REAL_V = 50_257
RANKS = (16, 32, 64, 128, 256, 512, 768, 1152)
RIDGE = 1.0


def _standardize(train: torch.Tensor, full: torch.Tensor):
    mean = train.float().mean(0, keepdim=True)
    scale = float((train.float() - mean).square().mean().sqrt().clamp_min(1e-12))
    return (full.float() - mean) / scale, mean, scale


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float(1 - (target - prediction).square().sum()
                 / target.square().sum().clamp_min(1e-12))


def _response_metrics(target: torch.Tensor, prediction: torch.Tensor):
    a, b = target.flatten(), prediction.flatten()
    return {
        "r2": _r2(target, prediction),
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)),
        "norm_ratio": float(b.norm() / a.norm().clamp_min(1e-12)),
    }


@torch.no_grad()
def _capture(model):
    block0 = model.transformer.h[0]
    raw_rows, z_rows, action_rows = [], [], []
    for start in range(0, REAL_V, 512):
        token = torch.arange(start, min(start + 512, REAL_V), device=DEV).view(-1, 1)
        raw = F.rms_norm(model.transformer.wte(token), (D,))
        remix = (block0.lambdas[0] + block0.lambdas[1]) * raw
        attention, _value = block0.attn(F.rms_norm(remix, (D,)), None)
        z = F.rms_norm(remix + attention, (D,))
        write = block0.mlp(z)
        action = write - block0.mlp.Down_bias.view(1, 1, D).to(write)
        raw_rows.append(raw[:, 0].float().cpu().half())
        z_rows.append(z[:, 0].float().cpu().half())
        action_rows.append(action[:, 0].float().cpu().half())
    return torch.cat(raw_rows), torch.cat(z_rows), torch.cat(action_rows)


@torch.no_grad()
def _rank_curve(train_source: torch.Tensor, eval_source: torch.Tensor,
                train_target: torch.Tensor, eval_target: torch.Tensor):
    covariance = train_source.T @ train_source / len(train_source)
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    floor = float(values[-1]) * 1e-6
    safe = values.clamp_min(floor)
    inv_sqrt = (vectors * safe.rsqrt()) @ vectors.T
    whitened_train = train_source @ inv_sqrt
    whitened_eval = eval_source @ inv_sqrt
    cross = whitened_train.T @ train_target / len(train_target)
    u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
    predictions, scores = {}, {}
    for rank in RANKS:
        train_code = whitened_train @ u[:, :rank]
        eval_code = whitened_eval @ u[:, :rank]
        decoder = singular[:rank, None] * vh[:rank]
        train_prediction = train_code @ decoder
        eval_prediction = eval_code @ decoder
        predictions[rank] = (train_prediction, eval_prediction)
        scores[rank] = {
            "heldout_write_r2": _r2(eval_target, eval_prediction),
            "literal_scalars": 2 * D * rank + 2 * D,
        }
    return predictions, scores


@torch.no_grad()
def _ridge(train_source: torch.Tensor, train_target: torch.Tensor):
    design = torch.cat((train_source, torch.ones(len(train_source), 1, device=DEV)), dim=1)
    penalty = RIDGE * torch.eye(design.shape[1], device=DEV)
    penalty[-1, -1] = 0
    return torch.linalg.solve(design.T @ design + penalty, design.T @ train_target)


@torch.no_grad()
def _retrieval(train_component: torch.Tensor, eval_component: torch.Tensor,
               train_raw: torch.Tensor, all_raw_actual: torch.Tensor,
               eval_token_ids: torch.Tensor):
    decoder = _ridge(train_component, train_raw)
    design = torch.cat((eval_component, torch.ones(len(eval_component), 1, device=DEV)), 1)
    predicted = F.normalize(design @ decoder, dim=1)
    candidates = F.normalize(all_raw_actual, dim=1)
    top1 = top5 = 0
    for start in range(0, len(predicted), 256):
        query = predicted[start:start + 256]
        truth = eval_token_ids[start:start + len(query)]
        top = (query @ candidates.T).topk(5, dim=1).indices
        top1 += int((top[:, 0] == truth).sum())
        top5 += int((top == truth[:, None]).any(1).sum())
    return {"top1_accuracy": top1 / len(predicted), "top5_accuracy": top5 / len(predicted)}


@torch.no_grad()
def _causal_responses(model, token_ids: torch.Tensor,
                      predictions_actual: dict[str, torch.Tensor]):
    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    names = ("native", "removed", *predictions_actual)
    values = {name: {"attention": [], "mlp": []} for name in names}
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

        def run(action):
            post0 = pre0 + bias + action
            remixed1 = block1.lambdas[0] * post0 + block1.lambdas[1] * raw
            attention1, _ = block1.attn(F.rms_norm(remixed1, (D,)), value0)
            mlp1 = block1.mlp(F.rms_norm(remixed1 + attention1, (D,)))
            return attention1[:, 0].float().cpu(), mlp1[:, 0].float().cpu()

        arms = {"native": native_action, "removed": torch.zeros_like(native_action)}
        for name, prediction in predictions_actual.items():
            arms[name] = prediction[start:start + len(token)].to(DEV)[:, None]
        for name, action in arms.items():
            attention, mlp = run(action)
            values[name]["attention"].append(attention)
            values[name]["mlp"].append(mlp)
    return {name: {kind: torch.cat(parts).to(DEV) for kind, parts in by_kind.items()}
            for name, by_kind in values.items()}


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert RANKS == (16, 32, 64, 128, 256, 512, 768, 1152)
        assert all(2 * D * rank + 2 * D > 0 for rank in RANKS)
        print("MLP0 QUADRATIC TOKEN DEGREE1 | dry run: split, ranks, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    raw_cpu, z_cpu, action_cpu = _capture(model)
    token_ids = torch.arange(REAL_V)
    fit_cpu, eval_cpu = token_ids % 5 != 0, token_ids % 5 == 0
    raw, raw_mean, raw_scale = _standardize(raw_cpu[fit_cpu], raw_cpu)
    z, z_mean, z_scale = _standardize(z_cpu[fit_cpu], z_cpu)
    action, action_mean, action_scale = _standardize(action_cpu[fit_cpu], action_cpu)
    raw, z, action = raw.to(DEV), z.to(DEV), action.to(DEV)
    fit, evaluate = fit_cpu.to(DEV), eval_cpu.to(DEV)

    z_predictions, z_scores = _rank_curve(z[fit], z[evaluate], action[fit], action[evaluate])
    raw_predictions, raw_scores = _rank_curve(
        raw[fit], raw[evaluate], action[fit], action[evaluate])

    generator = torch.Generator(device=DEV).manual_seed(396)
    shuffled = action[fit][torch.randperm(int(fit.sum()), generator=generator, device=DEV)]
    shuffled_predictions, shuffled_scores = _rank_curve(
        z[fit], z[evaluate], shuffled, action[evaluate])

    selected_ranks = (16, 32, 64, 128, 256, 512, 1152)
    eval_actual = {
        f"z_r{rank}": z_predictions[rank][1] * action_scale + action_mean.to(DEV)
        for rank in selected_ranks}
    eval_actual["z_shuffle_r256"] = (
        shuffled_predictions[256][1] * action_scale + action_mean.to(DEV))
    responses = _causal_responses(model, token_ids[eval_cpu], eval_actual)
    del model
    native_attention = responses["native"]["attention"] - responses["removed"]["attention"]
    native_mlp = responses["native"]["mlp"] - responses["removed"]["mlp"]
    native_joint = torch.cat((native_attention, native_mlp), 1)
    causal = {}
    for name in eval_actual:
        attention = responses[name]["attention"] - responses["removed"]["attention"]
        mlp = responses[name]["mlp"] - responses["removed"]["mlp"]
        causal[name] = {
            "attention1": _response_metrics(native_attention, attention),
            "mlp1": _response_metrics(native_mlp, mlp),
            "joint": _response_metrics(native_joint, torch.cat((attention, mlp), 1)),
        }

    retrieval = {}
    all_raw_actual = raw * raw_scale + raw_mean.to(DEV)
    eval_token_ids = token_ids[eval_cpu].to(DEV)
    for rank in (64, 128, 256, 512, 1152):
        retrieval[str(rank)] = _retrieval(
            z_predictions[rank][0], z_predictions[rank][1],
            raw[fit], all_raw_actual, eval_token_ids)

    full_z = z_scores[1152]["heldout_write_r2"]
    full_raw = raw_scores[1152]["heldout_write_r2"]
    qualifying_causal = [rank for rank in (16, 32, 64, 128)
                         if causal[f"z_r{rank}"]["joint"]["r2"] >= .90]
    qualifying_retrieval = [rank for rank in (64, 128, 256)
                            if retrieval[str(rank)]["top1_accuracy"] >= .80]
    pred_a = full_z >= .65 and full_z >= full_raw + .15
    pred_b = bool(qualifying_causal)
    pred_c = bool(qualifying_retrieval)
    pred_d = (
        causal["z_r256"]["joint"]["r2"] >= causal["z_r128"]["joint"]["r2"] - .02
        and causal["z_r512"]["joint"]["r2"] >= causal["z_r256"]["joint"]["r2"] - .02
        and causal["z_shuffle_r256"]["joint"]["r2"] <= .25)
    strong_null = (
        causal["z_r1152"]["joint"]["r2"] <= .50
        or causal["z_r512"]["joint"]["r2"] <= .50
        or causal["z_shuffle_r256"]["joint"]["r2"]
        >= causal["z_r256"]["joint"]["r2"] - .05)
    licensed = bool(pred_a and pred_b and pred_c and pred_d and not strong_null)
    result = {
        "status": "mlp0_quadratic_token_degree1_rank_curve_complete",
        "rung": 396,
        "claim_level": "exact_mlp0_input_empirical_degree1_token_causal_rank_screen",
        "population": {"real_tokens": REAL_V, "fit_tokens": int(fit_cpu.sum()),
                       "heldout_tokens": int(eval_cpu.sum()), "split": "token_id_mod5"},
        "definition": "F(z)=constant + best degree-one map in empirical z metric + orthogonal quadratic residual",
        "scales": {"raw": raw_scale, "z": z_scale, "write": action_scale},
        "z_degree1_rank_curve": {str(k): v for k, v in z_scores.items()},
        "raw_x0_rank_curve_control": {str(k): v for k, v in raw_scores.items()},
        "shuffled_z_rank256_write_r2": shuffled_scores[256]["heldout_write_r2"],
        "causal_injection": causal,
        "token_retrieval_from_z_degree1_component": retrieval,
        "qualifying_causal_rank_le128": qualifying_causal,
        "qualifying_retrieval_rank_le256": qualifying_retrieval,
        'pred_a_exact_z_degree1_explains_write_better_than_raw': bool(pred_a),
        'pred_b_compact_degree1_rank_preserves_downstream_response': bool(pred_b),
        'pred_c_compact_degree1_rank_preserves_token_identity': bool(pred_c),
        'pred_d_rank_curve_and_shuffle_control_hold': bool(pred_d),
        "null_degree1_token_kernel_not_causally_sufficient": bool(strong_null),
        "live_tt_transfer_licensed": licensed,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 QUADRATIC TOKEN DEGREE1 DONE", flush=True)


if __name__ == "__main__":
    main()
