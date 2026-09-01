"""RUNG 314 -- EXACT-TOKEN-METRIC SHARED INPUT ENCODER FOR MLP0.

At position zero with sequence length one, every real GPT-2 token determines
MLP0's input exactly through embedding, block-0 remix, and self-only attention.
Use that complete finite population to look for structure in the bilinear maps
themselves, rather than another output SAE or hidden-unit support heuristic.

Concatenate A=[Left; Right] and replace it by one shared encoder B plus separate
latent maps A_L/A_R:

    z = B x;  l = A_L z;  r = A_R z;  y = Down[(l) odot (r)] + bias.

At latent rank p the literal MLP0 price is

    1152*p + 2*4608*p + 4608*1152 + 1152.

Thus p512 saves 5,308,416 scalars (33.33% of MLP0) and p768 saves 2,654,208
(16.67%).  Fit on token ids t mod 5 != 0 and evaluate the untouched fifth.

Three matched basis families use identical executable graphs and prices:
  token_rrr: reduced-rank regression under the exact folded-token covariance;
  weight_svd: ordinary Frobenius SVD of [Left;Right];
  input_pca: top folded-input covariance directions with projected weights.

Then install each literal factorization at MLP0 on disjoint FineWeb and
WikiText rows.  This is input-side structure and is distinct from the rejected
output-PCA campaign and the old coefficient-metric HOSVD at MLP16.

Frozen predictions
------------------
pred_a_exact_token_function_has_low_shared_input_rank:
    token_rrr output R2 is >=.95 at p512 OR >=.98 at p768 on heldout token ids.
pred_b_structure_transfers_to_contextual_inputs:
    The same qualifying arm has damage <=.06 (p512) or <=.04 (p768) on BOTH
    FineWeb and WikiText.
pred_c_token_metric_beats_matched_controls:
    The same arm has >=20% lower mean nonnegative contextual damage than BOTH
    weight_svd and input_pca at the same rank.

Null: both token_rrr arms have heldout-token output R2 <=.80, or both have
contextual damage >=.15 on at least one corpus.  A pass is a single-MLP screen;
mixed104 composition, census, certificates, exact whole-model bill, and signed
interventions remain mandatory.
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
OUT = ROOT / "mlp0_exact_token_shared_input_encoder_results.json"
DEV = "cuda"
D = 1152
H = 4608
REAL_V = 50257
RANKS = (512, 768)
EVAL_ROWS = 12
WIKI_SKIP = 60000
NATIVE_MLP = 3 * H * D + D


def _price(rank: int) -> int:
    return D * rank + 2 * H * rank + H * D + D


@torch.no_grad()
def _folded_inputs(model, cfg) -> torch.Tensor:
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import apply_rot, rope_tables

    heads = cfg["n_head"]
    head_dim = D // heads
    block = model.transformer.h[0]
    rows = []
    for start in range(0, REAL_V, 512):
        token = torch.arange(start, min(start + 512, REAL_V), device=DEV).view(-1, 1)
        batch, length = token.shape
        x0 = F.rms_norm(model.transformer.wte(token), (D,))
        x = block.lambdas[0] * x0 + block.lambdas[1] * x0
        attention = block.attn
        current = F.rms_norm(x, (D,))
        cosine, sine = rope_tables(length, head_dim, DEV, x.dtype, "bf16")
        cosine, sine = cosine[None, :, None, :], sine[None, :, None, :]

        def qk(linear):
            value = F.rms_norm(linear(current).view(batch, length, heads, head_dim), (head_dim,))
            return apply_rot(value, cosine, sine)

        value = attention.c_v(current).view(batch, length, heads, head_dim)
        query, key = qk(attention.c_q), qk(attention.c_k)
        query2, key2 = qk(attention.c_q2), qk(attention.c_k2)
        score = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim
        score2 = torch.einsum("bqhd,bkhd->bhqk", query2, key2) / head_dim
        output = torch.einsum(
            "bhqk,bkhd->bqhd", score * score2, value
        ).reshape(batch, length, D)
        x = x + attention.c_proj(output)
        rows.append(F.rms_norm(x, (D,)).reshape(batch, D).float().cpu())
    folded = torch.cat(rows)
    assert folded.shape == (REAL_V, D) and bool(torch.isfinite(folded).all())
    return folded


@torch.no_grad()
def _fit_programs(x_fit, left, right):
    x = x_fit.to(DEV)
    covariance = x.T @ x / len(x)
    covariance = 0.5 * (covariance + covariance.T)
    values_x, vectors_x = torch.linalg.eigh(covariance)
    order_x = torch.argsort(values_x, descending=True)
    values_x, vectors_x = values_x[order_x], vectors_x[:, order_x]
    floor = float(values_x[0]) * 1e-6
    safe = values_x.clamp_min(floor)
    covariance_sqrt = (vectors_x * safe.sqrt()) @ vectors_x.T
    covariance_inv_sqrt = (vectors_x * safe.rsqrt()) @ vectors_x.T

    stacked = torch.cat((left, right), dim=0).to(DEV)
    weight_gram = stacked.T @ stacked
    metric_gram = covariance_sqrt @ weight_gram @ covariance_sqrt
    metric_gram = 0.5 * (metric_gram + metric_gram.T)
    values_rrr, vectors_rrr = torch.linalg.eigh(metric_gram)
    vectors_rrr = vectors_rrr[:, torch.argsort(values_rrr, descending=True)]
    values_w, vectors_w = torch.linalg.eigh(0.5 * (weight_gram + weight_gram.T))
    vectors_w = vectors_w[:, torch.argsort(values_w, descending=True)]

    programs = {}
    for rank in RANKS:
        vr = vectors_rrr[:, :rank]
        encoder_rrr = vr.T @ covariance_inv_sqrt
        coefficient_rrr = stacked @ covariance_sqrt @ vr
        vw = vectors_w[:, :rank]
        vx = vectors_x[:, :rank]
        for method, encoder, coefficient in (
            ("token_rrr", encoder_rrr, coefficient_rrr),
            ("weight_svd", vw.T, stacked @ vw),
            ("input_pca", vx.T, stacked @ vx),
        ):
            programs[f"{method}_{rank}"] = {
                "rank": rank,
                "method": method,
                "encoder": encoder.float().cpu(),
                "left": coefficient[:H].float().cpu(),
                "right": coefficient[H:].float().cpu(),
            }
    diagnostics = {
        "folded_input_top512_energy": float(values_x[:512].sum() / values_x.clamp_min(0).sum()),
        "folded_input_top768_energy": float(values_x[:768].sum() / values_x.clamp_min(0).sum()),
        "folded_input_condition_after_floor": float(values_x[0] / safe[-1]),
        "covariance_eigenvalue_floor": floor,
    }
    return programs, diagnostics


@torch.no_grad()
def _token_function_scores(x_eval, programs, left, right, down):
    targets = []
    for start in range(0, len(x_eval), 256):
        x = x_eval[start:start + 256].to(DEV)
        hidden = (x @ left.T) * (x @ right.T)
        targets.append((hidden @ down.T).float().cpu())
    target = torch.cat(targets)
    target_mean = target.mean(0, keepdim=True)
    denominator = float((target - target_mean).square().sum())
    scores = {}
    for name, program in programs.items():
        encoder = program["encoder"].to(DEV)
        left_latent = program["left"].to(DEV)
        right_latent = program["right"].to(DEV)
        numerator = 0.0
        for start in range(0, len(x_eval), 256):
            x = x_eval[start:start + 256].to(DEV)
            z = x @ encoder.T
            hidden = (z @ left_latent.T) * (z @ right_latent.T)
            prediction = hidden @ down.T
            truth = target[start:start + len(x)].to(DEV)
            numerator += float((prediction - truth).square().sum())
        scores[name] = 1.0 - numerator / max(denominator, 1e-12)
        print(f"{name}: heldout-token output R2 {scores[name]:.6f}", flush=True)
    return scores


def _manual_logits(model, index):
    x = F.rms_norm(model.transformer.wte(index), (D,))
    x0 = x
    value0 = None
    for block in model.transformer.h:
        x, value0 = block(x, value0, x0)
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def _score_context(model, rows, program, down, bias):
    handle = None
    if program is not None:
        encoder = program["encoder"].to(DEV)
        left_latent = program["left"].to(DEV)
        right_latent = program["right"].to(DEV)

        def hook(_module, args, output):
            x = args[0].float()
            z = x @ encoder.T
            hidden = (z @ left_latent.T) * (z @ right_latent.T)
            return (hidden @ down.T + bias).to(output.dtype)

        handle = model.transformer.h[0].mlp.register_forward_hook(hook)
    total, count = 0.0, 0
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index, target = batch[:, :-1].to(DEV), batch[:, 1:].to(DEV)
            logits = _manual_logits(model, index)
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), reduction="sum"
            ))
            count += target.numel()
    finally:
        if handle is not None:
            handle.remove()
    return total / count


def _mean_nonnegative(values):
    return sum(max(0.0, value) for value in values) / len(values)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for filename in ("fineweb_n192_skip11000.pt",):
            assert (ROOT / ".rowcache" / filename).exists()
        assert _price(512) == 10_617_984 and NATIVE_MLP - _price(512) == 5_308_416
        assert _price(768) == 13_272_192 and NATIVE_MLP - _price(768) == 2_654_208
        print("MLP0 EXACT-TOKEN SHARED INPUT ENCODER | dry run: split, prices, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from mlp0_signed_response_rank_screen import _load_rows, _wikitext_rows
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == 18
    mlp = model.transformer.h[0].mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    bias = mlp.Down_bias.detach().float()
    folded = _folded_inputs(model, cfg)
    token_ids = torch.arange(REAL_V)
    fit_mask = token_ids % 5 != 0
    eval_mask = ~fit_mask
    programs, diagnostics = _fit_programs(folded[fit_mask], left, right)
    token_scores = _token_function_scores(folded[eval_mask], programs, left, right, down)

    fineweb = _load_rows(ROOT / ".rowcache/fineweb_n192_skip11000.pt", EVAL_ROWS)
    wikitext, fingerprint = _wikitext_rows(EVAL_ROWS, skip=WIKI_SKIP)
    native = {
        "fineweb": _score_context(model, fineweb, None, down, bias),
        "wikitext": _score_context(model, wikitext, None, down, bias),
    }
    arms = {}
    for name, program in programs.items():
        ce_fineweb = _score_context(model, fineweb, program, down, bias)
        ce_wikitext = _score_context(model, wikitext, program, down, bias)
        rank = program["rank"]
        arms[name] = {
            "method": program["method"],
            "rank": rank,
            "heldout_token_output_r2": token_scores[name],
            "fineweb_damage": ce_fineweb - native["fineweb"],
            "wikitext_damage": ce_wikitext - native["wikitext"],
            "literal_mlp0_scalars": _price(rank),
            "saving_scalars": NATIVE_MLP - _price(rank),
        }
        print(f"{name}: FW/WT {arms[name]['fineweb_damage']:+.5f}/"
              f"{arms[name]['wikitext_damage']:+.5f}", flush=True)

    qualifying = []
    for rank in RANKS:
        candidate = arms[f"token_rrr_{rank}"]
        r2_bar, damage_bar = ((.95, .06) if rank == 512 else (.98, .04))
        candidate_damage = _mean_nonnegative([
            candidate["fineweb_damage"], candidate["wikitext_damage"]
        ])
        control_damages = []
        for method in ("weight_svd", "input_pca"):
            control = arms[f"{method}_{rank}"]
            control_damages.append(_mean_nonnegative([
                control["fineweb_damage"], control["wikitext_damage"]
            ]))
        if (candidate["heldout_token_output_r2"] >= r2_bar
                and max(candidate["fineweb_damage"], candidate["wikitext_damage"]) <= damage_bar
                and all(candidate_damage <= .8 * value for value in control_damages)):
            qualifying.append(rank)
    pred_a = (arms["token_rrr_512"]["heldout_token_output_r2"] >= .95
              or arms["token_rrr_768"]["heldout_token_output_r2"] >= .98)
    pred_b = ((arms["token_rrr_512"]["heldout_token_output_r2"] >= .95
               and max(arms["token_rrr_512"]["fineweb_damage"],
                       arms["token_rrr_512"]["wikitext_damage"]) <= .06)
              or (arms["token_rrr_768"]["heldout_token_output_r2"] >= .98
                  and max(arms["token_rrr_768"]["fineweb_damage"],
                          arms["token_rrr_768"]["wikitext_damage"]) <= .04))
    pred_c = bool(qualifying)
    null = (all(arms[f"token_rrr_{rank}"]["heldout_token_output_r2"] <= .80 for rank in RANKS)
            or all(max(arms[f"token_rrr_{rank}"]["fineweb_damage"],
                       arms[f"token_rrr_{rank}"]["wikitext_damage"]) >= .15 for rank in RANKS))
    result = {
        "status": "mlp0_exact_token_shared_input_encoder_complete",
        "rung": 314,
        "claim_level": "single_mlp_exact_token_metric_and_two_corpus_contextual_screen_only",
        "convention": "CE added above native; lower is better",
        "population": {"real_tokens": REAL_V, "fit_tokens": int(fit_mask.sum()),
                       "heldout_tokens": int(eval_mask.sum()), "split": "token_id_mod5"},
        "diagnostics": diagnostics,
        "native_context_ce": native,
        "wikitext_fingerprint": str(fingerprint),
        "arms": arms,
        "jointly_qualifying_ranks": qualifying,
        'pred_a_exact_token_function_has_low_shared_input_rank': bool(pred_a),
        'pred_b_structure_transfers_to_contextual_inputs': bool(pred_b),
        'pred_c_token_metric_beats_matched_controls': bool(pred_c),
        "null_no_useful_shared_input_structure": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"qualifying": qualifying, "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP0 EXACT-TOKEN SHARED INPUT ENCODER DONE", flush=True)


if __name__ == "__main__":
    main()
