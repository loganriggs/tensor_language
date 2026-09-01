"""RUNG 394 -- EXACT-TOKEN MLP0 DOWNSTREAM-EQUIVALENCE SPARSE-CODE SCREEN.

Enumerate all 50,257 real tokens at sequence length one.  Attention0 is then
self-only, so the complete block-0/MLP0 path is a deterministic token function.
For every token compare the native path with a counterfactual that removes only
MLP0's bias-free write while retaining the same raw x0, attention0 value, MLP0
bias, and block-1 remix.  The causal response signature is the resulting change
in attention1 and MLP1 writes.  This directly asks what MLP0 adds beyond the raw
token state that block 1 explicitly reinjects.

Fit on token_id mod 5 != 0 and score the untouched fifth.  At P=256,k=16 compare
an action-reconstruction signed TopK code, a joint action+response code (two
seeds), and an identical shuffled-response negative.  Dense action-PCA and raw
x0 linear maps are controls.  This is a deterministic exhaustive-token screen,
not yet a live-context TT/X/CC replacement or an adoption claim.

Frozen predictions
------------------
pred_a: block1's exact raw-x0 remix coefficient is at least 100 times its
    incoming-state coefficient; the median norm of the scaled bias-free MLP0
    contribution is <=.25 of the raw-x0 term and their median absolute cosine
    is <=.30.  This predicts nonlinear feature correction, not token copying.
pred_b: the concatenated attention1+MLP1 causal response has no larger 90%-energy
    rank or participation rank than the token-only MLP0 write.
pred_c: the heldout joint sparse code predicts response with R2>=.50, exceeds
    activation-only by >=.02 and shuffled-response by >=.10.
pred_d: the two joint seeds differ by <=.03 response R2, have top-64 decoder
    rowspace overlap >=.70, and joint-code nearest neighbors improve heldout
    response cosine by >=.05 over activation-only and >=.10 over shuffled.

Strong null: causal-response RMS <1e-5, raw-x0 reinjection is absent, joint
response R2 <=.25, or shuffled response comes within .02 of joint.  A full pass
licenses one live-context test of whether the same code predicts TT and TT-X
causal responses.  Individual atoms receive semantic names only if their token
memberships are stable across seeds; atom stability is reported, not assumed.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_exact_token_downstream_sparse_code_results.json"
DEV = "cuda"
D = 1152
REAL_V = 50_257
P = 256
TOPK = 16
STEPS = 1_000
BATCH = 1_024
RIDGE = 1e-3
SEEDS = (0, 1)


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    numerator = (target - prediction).square().sum()
    denominator = target.square().sum().clamp_min(1e-12)
    return float(1.0 - numerator / denominator)


def _standardize(train: torch.Tensor, full: torch.Tensor) -> tuple[torch.Tensor, float]:
    mean = train.float().mean(0, keepdim=True)
    scale = float((train.float() - mean).square().mean().sqrt().clamp_min(1e-12))
    return (full.float() - mean) / scale, scale


def _spectrum(matrix: torch.Tensor) -> dict[str, float | int]:
    gram = matrix.T @ matrix / len(matrix)
    values = torch.linalg.eigvalsh(0.5 * (gram + gram.T)).clamp_min(0).flip(0)
    total = values.sum().clamp_min(1e-30)
    cumulative = values.cumsum(0) / total

    def energy_rank(level: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(level, device=values.device))) + 1

    participation = float(total.square() / values.square().sum().clamp_min(1e-30))
    return {
        "rank80": energy_rank(.80),
        "rank90": energy_rank(.90),
        "rank95": energy_rank(.95),
        "participation_rank": participation,
    }


def _pca_response(train_x: torch.Tensor, eval_x: torch.Tensor,
                  train_s: torch.Tensor, eval_s: torch.Tensor,
                  rank: int) -> tuple[float, torch.Tensor]:
    gram = train_x.T @ train_x / len(train_x)
    _values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    basis = vectors[:, -rank:]
    z_train, z_eval = train_x @ basis, eval_x @ basis
    decoder = torch.linalg.solve(
        z_train.T @ z_train + RIDGE * torch.eye(rank, device=DEV),
        z_train.T @ train_s,
    )
    prediction = z_eval @ decoder
    return _r2(eval_s, prediction), z_eval


def _fit_code_response(train_code: torch.Tensor, eval_code: torch.Tensor,
                       train_s: torch.Tensor, eval_s: torch.Tensor) -> tuple[float, torch.Tensor]:
    ones_train = torch.ones(len(train_code), 1, device=DEV)
    ones_eval = torch.ones(len(eval_code), 1, device=DEV)
    z_train = torch.cat((train_code, ones_train), dim=1)
    z_eval = torch.cat((eval_code, ones_eval), dim=1)
    penalty = RIDGE * torch.eye(z_train.shape[1], device=DEV)
    penalty[-1, -1] = 0
    decoder = torch.linalg.solve(z_train.T @ z_train + penalty, z_train.T @ train_s)
    prediction = z_eval @ decoder
    return _r2(eval_s, prediction), prediction


class SparseCode(nn.Module):
    def __init__(self, response_dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        encoder = torch.randn(P, D, generator=generator) / math.sqrt(D)
        self.encoder = nn.Parameter(encoder)
        self.encoder_bias = nn.Parameter(torch.zeros(P))
        self.action_decoder = nn.Parameter(torch.randn(P, D, generator=generator) / math.sqrt(P))
        self.action_bias = nn.Parameter(torch.zeros(D))
        self.response_decoder = nn.Parameter(
            torch.randn(P, response_dim, generator=generator) / math.sqrt(P))
        self.response_bias = nn.Parameter(torch.zeros(response_dim))

    def code(self, x: torch.Tensor) -> torch.Tensor:
        dense = F.linear(x, self.encoder, self.encoder_bias)
        indices = dense.abs().topk(TOPK, dim=-1).indices
        sparse = torch.zeros_like(dense)
        return sparse.scatter(1, indices, dense.gather(1, indices))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.code(x)
        action = code @ self.action_decoder + self.action_bias
        response = code @ self.response_decoder + self.response_bias
        return code, action, response


def _train_sparse(train_y: torch.Tensor, train_s: torch.Tensor, *, seed: int,
                  mode: str, permutation: torch.Tensor | None = None) -> SparseCode:
    torch.manual_seed(seed)
    model = SparseCode(train_s.shape[1], seed).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    generator = torch.Generator(device=DEV).manual_seed(10_000 + seed)
    for step in range(STEPS):
        index = torch.randint(len(train_y), (BATCH,), generator=generator, device=DEV)
        y = train_y[index]
        target_s = train_s[index if permutation is None else permutation[index]]
        _code, pred_y, pred_s = model(y)
        loss_y = (pred_y - y).square().mean()
        loss_s = (pred_s - target_s).square().mean()
        loss = loss_y if mode == "activation" else loss_y + loss_s
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.encoder.div_(model.encoder.norm(dim=1, keepdim=True).clamp_min(1e-12))
        if step in (0, 249, 499, 749, 999):
            print(f"{mode} seed{seed} step{step + 1}: y={float(loss_y):.5f} "
                  f"s={float(loss_s):.5f}", flush=True)
    return model.eval()


@torch.no_grad()
def _capture_exact_token_population(model) -> dict[str, torch.Tensor | float]:
    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    xs, actions, attention_responses, mlp_responses = [], [], [], []
    contribution_ratios, contribution_cosines = [], []
    l_state = float(block1.lambdas[0])
    l_raw = float(block1.lambdas[1])
    for start in range(0, REAL_V, 512):
        token = torch.arange(start, min(start + 512, REAL_V), device=DEV).view(-1, 1)
        x0 = F.rms_norm(model.transformer.wte(token), (D,))
        remix0 = block0.lambdas[0] * x0 + block0.lambdas[1] * x0
        attention0, value0 = block0.attn(F.rms_norm(remix0, (D,)), None)
        pre_mlp0 = remix0 + attention0
        write0 = block0.mlp(F.rms_norm(pre_mlp0, (D,)))
        bias0 = block0.mlp.Down_bias.view(1, 1, D).to(write0)
        action0 = write0 - bias0

        native0 = pre_mlp0 + write0
        removed0 = pre_mlp0 + bias0

        def block1_outputs(previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            remixed = block1.lambdas[0] * previous + block1.lambdas[1] * x0
            attention1, _ = block1.attn(F.rms_norm(remixed, (D,)), value0)
            pre_mlp1 = remixed + attention1
            write1 = block1.mlp(F.rms_norm(pre_mlp1, (D,)))
            return attention1, write1

        attention_native, mlp_native = block1_outputs(native0)
        attention_removed, mlp_removed = block1_outputs(removed0)
        raw_term = block1.lambdas[1] * x0
        mlp_term = block1.lambdas[0] * action0
        ratio = mlp_term.float().norm(dim=-1) / raw_term.float().norm(dim=-1).clamp_min(1e-12)
        cosine = F.cosine_similarity(mlp_term.float(), raw_term.float(), dim=-1).abs()

        xs.append(x0[:, 0].float().cpu().half())
        actions.append(action0[:, 0].float().cpu().half())
        attention_responses.append(
            (attention_native - attention_removed)[:, 0].float().cpu().half())
        mlp_responses.append((mlp_native - mlp_removed)[:, 0].float().cpu().half())
        contribution_ratios.append(ratio[:, 0].cpu())
        contribution_cosines.append(cosine[:, 0].cpu())
    return {
        "x0": torch.cat(xs),
        "action": torch.cat(actions),
        "attention_response": torch.cat(attention_responses),
        "mlp_response": torch.cat(mlp_responses),
        "contribution_ratio": torch.cat(contribution_ratios),
        "contribution_abs_cosine": torch.cat(contribution_cosines),
        "block1_incoming_state_coefficient": l_state,
        "block1_raw_x0_coefficient": l_raw,
    }


def _nearest_neighbor_metrics(code: torch.Tensor, response: torch.Tensor,
                              action: torch.Tensor, raw: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    code = F.normalize(code.float(), dim=1)
    response = F.normalize(response.float(), dim=1)
    action = F.normalize(action.float(), dim=1)
    raw = F.normalize(raw.float(), dim=1)
    neighbors = []
    for start in range(0, len(code), 512):
        similarity = code[start:start + 512] @ code.T
        row = torch.arange(len(similarity), device=DEV)
        similarity[row, row + start] = -torch.inf
        neighbors.append(similarity.argmax(1))
    neighbor = torch.cat(neighbors)
    response_cosine = (response * response[neighbor]).sum(1)
    action_cosine = (action * action[neighbor]).sum(1)
    raw_cosine = (raw * raw[neighbor]).sum(1)
    discordant = (response_cosine >= .80) & (action_cosine <= .50)
    return {
        "mean_response_cosine": float(response_cosine.mean()),
        "median_response_cosine": float(response_cosine.median()),
        "mean_action_cosine": float(action_cosine.mean()),
        "median_action_cosine": float(action_cosine.median()),
        "mean_raw_x0_cosine": float(raw_cosine.mean()),
        "discordant_response_ge_080_action_le_050_fraction": float(discordant.float().mean()),
        "discordant_count": int(discordant.sum()),
    }, neighbor


def _decoder_stability(left: SparseCode, right: SparseCode,
                       left_code: torch.Tensor, right_code: torch.Tensor) -> dict[str, float | int]:
    from scipy.optimize import linear_sum_assignment

    left_decoder = F.normalize(left.action_decoder.detach().float(), dim=1)
    right_decoder = F.normalize(right.action_decoder.detach().float(), dim=1)
    similarities = (left_decoder @ right_decoder.T).abs().cpu().numpy()
    rows, cols = linear_sum_assignment(-similarities)
    atom_cosines = torch.tensor(similarities[rows, cols])
    left_v = torch.linalg.svd(left_decoder, full_matrices=False).Vh[:64]
    right_v = torch.linalg.svd(right_decoder, full_matrices=False).Vh[:64]
    subspace = float((left_v @ right_v.T).square().sum() / 64)

    jaccards = []
    for left_atom, right_atom in zip(rows, cols):
        left_top = set(left_code[:, left_atom].abs().topk(64).indices.cpu().tolist())
        right_top = set(right_code[:, right_atom].abs().topk(64).indices.cpu().tolist())
        jaccards.append(len(left_top & right_top) / max(1, len(left_top | right_top)))
    return {
        "mean_matched_atom_decoder_cosine": float(atom_cosines.mean()),
        "median_matched_atom_decoder_cosine": float(atom_cosines.median()),
        "top64_decoder_rowspace_overlap": subspace,
        "median_top64_token_membership_jaccard": float(torch.tensor(jaccards).median()),
        "atoms_membership_jaccard_ge_025": sum(value >= .25 for value in jaccards),
    }


def _decode_token(token: int) -> str:
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2").decode_single_token_bytes(token).decode(
            "utf-8", "backslashreplace")
    except Exception:
        return f"token_{token}"


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert REAL_V == 50_257 and P == 256 and TOPK == 16
        assert len(SEEDS) == 2 and STEPS == 1_000
        print("MLP0 EXACT TOKEN DOWNSTREAM SPARSE CODE | dry run: split, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    captured = _capture_exact_token_population(model)
    del model
    torch.cuda.empty_cache()

    token_ids = torch.arange(REAL_V)
    fit_mask_cpu = token_ids % 5 != 0
    eval_mask_cpu = ~fit_mask_cpu
    x0 = captured["x0"]
    action = captured["action"]
    attention_response = captured["attention_response"]
    mlp_response = captured["mlp_response"]

    x_all, x_scale = _standardize(x0[fit_mask_cpu], x0)
    y_all, y_scale = _standardize(action[fit_mask_cpu], action)
    a_all, attention_scale = _standardize(
        attention_response[fit_mask_cpu], attention_response)
    m_all, mlp_scale = _standardize(mlp_response[fit_mask_cpu], mlp_response)
    s_all = torch.cat((a_all, m_all), dim=1)
    del x0, action, attention_response, mlp_response

    fit_mask = fit_mask_cpu.to(DEV)
    eval_mask = eval_mask_cpu.to(DEV)
    x_all, y_all, s_all = x_all.to(DEV), y_all.to(DEV), s_all.to(DEV)
    train_x, eval_x = x_all[fit_mask], x_all[eval_mask]
    train_y, eval_y = y_all[fit_mask], y_all[eval_mask]
    train_s, eval_s = s_all[fit_mask], s_all[eval_mask]

    spectra = {
        "token_only_mlp0_write": _spectrum(train_y),
        "attention1_response": _spectrum(train_s[:, :D]),
        "mlp1_response": _spectrum(train_s[:, D:]),
        "joint_downstream_response": _spectrum(train_s),
    }
    dense_controls = {}
    for name, source_train, source_eval in (
        ("raw_x0", train_x, eval_x), ("mlp0_write", train_y, eval_y)):
        for rank in (16, 64, 256):
            score, _code = _pca_response(
                source_train, source_eval, train_s, eval_s, rank)
            dense_controls[f"{name}_pca{rank}_response_r2"] = score

    permutation = torch.randperm(len(train_s), generator=torch.Generator(device=DEV).manual_seed(394),
                                 device=DEV)
    activation_model = _train_sparse(train_y, train_s, seed=0, mode="activation")
    shuffled_model = _train_sparse(
        train_y, train_s, seed=0, mode="shuffled", permutation=permutation)
    joint_models = [_train_sparse(train_y, train_s, seed=seed, mode="joint") for seed in SEEDS]

    models = {
        "activation_only_seed0": activation_model,
        "shuffled_response_seed0": shuffled_model,
        **{f"joint_seed{seed}": model for seed, model in zip(SEEDS, joint_models)},
    }
    sparse = {}
    codes = {}
    for name, sparse_model in models.items():
        train_code, train_yhat, _ = sparse_model(train_y)
        eval_code, eval_yhat, _ = sparse_model(eval_y)
        response_r2, _prediction = _fit_code_response(
            train_code, eval_code, train_s, eval_s)
        sparse[name] = {
            "heldout_action_r2": _r2(eval_y, eval_yhat),
            "heldout_response_r2_refit_decoder": response_r2,
            "mean_active_coordinates": float((eval_code != 0).float().sum(1).mean()),
        }
        codes[name] = eval_code.detach()

    neighbor_metrics = {}
    neighbors = {}
    for name in models:
        metrics, neighbor = _nearest_neighbor_metrics(
            codes[name], eval_s, eval_y, eval_x)
        neighbor_metrics[name] = metrics
        neighbors[name] = neighbor

    stability = _decoder_stability(
        joint_models[0], joint_models[1], codes["joint_seed0"], codes["joint_seed1"])
    joint_r2 = [sparse[f"joint_seed{seed}"]["heldout_response_r2_refit_decoder"]
                for seed in SEEDS]
    stability["response_r2_seed_range"] = max(joint_r2) - min(joint_r2)

    eval_token_ids = token_ids[eval_mask_cpu].to(DEV)
    joint_neighbor = neighbors["joint_seed0"]
    response_norm = F.normalize(eval_s, dim=1)
    action_norm = F.normalize(eval_y, dim=1)
    response_cos = (response_norm * response_norm[joint_neighbor]).sum(1)
    action_cos = (action_norm * action_norm[joint_neighbor]).sum(1)
    discordance = response_cos - action_cos
    candidate = torch.where((response_cos >= .80) & (action_cos <= .50))[0]
    if len(candidate):
        candidate = candidate[discordance[candidate].argsort(descending=True)[:20]]
    examples = []
    for index in candidate.cpu().tolist():
        neighbor = int(joint_neighbor[index])
        left_token = int(eval_token_ids[index])
        right_token = int(eval_token_ids[neighbor])
        examples.append({
            "left_id": left_token, "left_text": _decode_token(left_token),
            "right_id": right_token, "right_text": _decode_token(right_token),
            "response_cosine": float(response_cos[index]),
            "mlp0_write_cosine": float(action_cos[index]),
        })

    l_state = float(captured["block1_incoming_state_coefficient"])
    l_raw = float(captured["block1_raw_x0_coefficient"])
    ratios = captured["contribution_ratio"].float()
    cosines = captured["contribution_abs_cosine"].float()
    response_rms = float(torch.cat((
        captured["attention_response"].float(), captured["mlp_response"].float()), dim=1
    ).square().mean().sqrt())
    raw_dominance = {
        "block1_incoming_state_coefficient": l_state,
        "block1_raw_x0_coefficient": l_raw,
        "absolute_coefficient_ratio_raw_over_state": abs(l_raw) / max(abs(l_state), 1e-12),
        "scaled_mlp0_to_raw_term_norm_median": float(ratios.median()),
        "scaled_mlp0_to_raw_term_norm_p95": float(torch.quantile(ratios, .95)),
        "scaled_mlp0_to_raw_term_norm_max": float(ratios.max()),
        "scaled_mlp0_vs_raw_term_abs_cosine_median": float(cosines.median()),
    }

    activation_r2 = sparse["activation_only_seed0"]["heldout_response_r2_refit_decoder"]
    shuffled_r2 = sparse["shuffled_response_seed0"]["heldout_response_r2_refit_decoder"]
    joint_best = max(joint_r2)
    joint_neighbor_score = neighbor_metrics["joint_seed0"]["mean_response_cosine"]
    activation_neighbor_score = neighbor_metrics["activation_only_seed0"]["mean_response_cosine"]
    shuffled_neighbor_score = neighbor_metrics["shuffled_response_seed0"]["mean_response_cosine"]
    pred_a = (
        raw_dominance["absolute_coefficient_ratio_raw_over_state"] >= 100
        and raw_dominance["scaled_mlp0_to_raw_term_norm_median"] <= .25
        and raw_dominance["scaled_mlp0_vs_raw_term_abs_cosine_median"] <= .30)
    pred_b = (
        spectra["joint_downstream_response"]["rank90"]
        <= spectra["token_only_mlp0_write"]["rank90"]
        and spectra["joint_downstream_response"]["participation_rank"]
        <= spectra["token_only_mlp0_write"]["participation_rank"])
    pred_c = joint_best >= .50 and joint_best >= activation_r2 + .02 \
        and joint_best >= shuffled_r2 + .10
    pred_d = (
        stability["response_r2_seed_range"] <= .03
        and stability["top64_decoder_rowspace_overlap"] >= .70
        and joint_neighbor_score >= activation_neighbor_score + .05
        and joint_neighbor_score >= shuffled_neighbor_score + .10)
    strong_null = (
        response_rms < 1e-5 or abs(l_raw) < 1e-5 or joint_best <= .25
        or shuffled_r2 >= joint_best - .02)
    live_tt_x_licensed = bool(pred_a and pred_b and pred_c and pred_d and not strong_null)

    result = {
        "status": "mlp0_exact_token_downstream_sparse_code_complete",
        "rung": 394,
        "claim_level": "exhaustive_exact_token_causal_response_sparse_code_screen_only",
        "population": {
            "real_tokens": REAL_V,
            "fit_tokens": int(fit_mask_cpu.sum()),
            "heldout_tokens": int(eval_mask_cpu.sum()),
            "split": "token_id_mod5",
            "sequence_length": 1,
        },
        "counterfactual": "remove only bias-free MLP0 write; retain raw x0, attention0 value, MLP0 bias, and block1 remix",
        "scales": {
            "raw_x0": x_scale, "mlp0_write": y_scale,
            "attention1_response": attention_scale, "mlp1_response": mlp_scale,
        },
        "downstream_response_rms_unscaled": response_rms,
        "raw_token_reinjection": raw_dominance,
        "spectra": spectra,
        "dense_controls": dense_controls,
        "sparse_program": {"dictionary_width": P, "active_coordinates": TOPK, "steps": STEPS},
        "sparse_arms": sparse,
        "seed_stability": stability,
        "nearest_neighbor_response": neighbor_metrics,
        "shared_effect_different_write_examples": examples,
        'pred_a_mlp0_is_small_nonlinear_correction_not_token_copy': bool(pred_a),
        'pred_b_downstream_response_is_more_compressed_than_write': bool(pred_b),
        'pred_c_joint_sparse_code_predicts_heldout_response': bool(pred_c),
        'pred_d_joint_sparse_code_is_stable_and_groups_response': bool(pred_d),
        "null_no_sparse_downstream_equivalence_signal": bool(strong_null),
        "live_tt_x_transfer_licensed": live_tt_x_licensed,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "shared_effect_different_write_examples"}, indent=2), flush=True)
    print("MLP0 EXACT TOKEN DOWNSTREAM SPARSE CODE DONE", flush=True)


if __name__ == "__main__":
    main()
