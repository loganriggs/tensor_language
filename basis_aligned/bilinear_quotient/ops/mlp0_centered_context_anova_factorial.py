"""RUNG 400 -- CENTERED TOKEN/CONTEXT/NORMALIZATION CAUSAL ANOVA.

Reuse MLP0's exact bilinear TT/X/CC algebra, but change the statistical object.
Estimate token-input and attention-write moments on frozen FIT rows.  For the
unnormalized quadratic G(e,a)=T(e+a,e+a), construct the exact product-reference
split

    G = mu + FT(e) + FC(a) + FTC(e,a).

Writing s^2 for the observed RMSNorm gain and sbar^2 for its FIT mean, the
deployed float numerator is exactly

    sbar^2*(mu+FT+FC+FTC) + (s^2-sbar^2)*G.

Retain sbar^2*mu and factorially intervene on T=sbar^2*FT,
C=sbar^2*FC, I=sbar^2*FTC, and S=(s^2-sbar^2)*G across all 16
subsets on FIT and disjoint SELECT CE. Native bias and bf16 residual remain.

Frozen predictions
------------------
pred_a: analytical identity relative MSE <=1e-8 in both roles, full arm is the
    exact native path, and every arm has the frozen live-call census.
pred_b: SELECT context-main Shapley benefit C>=.50 nat and centered cross I>=.10.
pred_c: |SELECT scale-modulation Shapley S|>=.05 and C+I+S>=T.
pred_d: T/C/I/S Shapley signs and complete rank order match FIT and SELECT.

Strong null: combined SELECT context Shapley C+I+S<=.50; both |I| and |S|
<=.02; identity/instrument failure; or a SELECT context component reverses a
FIT sign after |FIT|>=.05.  The result selects centered X, CC, or RMS-gain
modulation for the next grammar; it cannot adopt a compressor.

Literal diagnostic storage reuses the native 15,926,400 MLP0 values and adds
five 1,152-vectors plus one scalar of reference moments.  The branch evaluator
uses six input projections and three Down applications per position.  This is
an attribution assay, not compression.
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
OUT = BQ / "mlp0_centered_context_anova_factorial_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
BRANCHES = ("T", "C", "I", "S")
ARMS = tuple(frozenset(branch for index, branch in enumerate(BRANCHES)
                       if mask & (1 << index)) for mask in range(16))
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
D = 1152
NATIVE_VALUES = 15_926_400
REFERENCE_VALUES = 5 * D + 1


def _arm_name(subset: frozenset[str]) -> str:
    return "+".join(branch for branch in BRANCHES if branch in subset) or "EMPTY"


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
    n = len(BRANCHES)
    result = {}
    for branch in BRANCHES:
        total = 0.0
        others = tuple(item for item in BRANCHES if item != branch)
        for size in range(len(others) + 1):
            coefficient = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for chosen in itertools.combinations(others, size):
                subset = frozenset(chosen)
                total += coefficient * (performance[subset | {branch}] - performance[subset])
        result[branch] = total
    return result


def _T(u: torch.Tensor, v: torch.Tensor, left: torch.Tensor,
       right: torch.Tensor, down: torch.Tensor):
    return F.linear(F.linear(u.float(), left) * F.linear(v.float(), right), down)


def _gain(normalized: torch.Tensor, token_base: torch.Tensor,
          attention_write: torch.Tensor):
    z = normalized.float()
    raw = token_base.float() + attention_write.float()
    denominator = raw.square().sum(-1, keepdim=True).clamp_min(1e-30)
    scale = (z * raw).sum(-1, keepdim=True) / denominator
    collinearity = (z - scale * raw).square().sum(-1) / z.square().sum(-1).clamp_min(1e-30)
    return scale.square(), collinearity


@torch.no_grad()
def _capture_inputs(model, rows: torch.Tensor, device: torch.device):
    block0 = model.transformer.h[0]
    token_parts, context_parts, gains = [], [], []
    for start in range(0, len(rows), DOCUMENT_BATCH):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        attention, _ = block0.attn(F.rms_norm(token_base, (D,)), None)
        normalized = F.rms_norm(token_base + attention, (D,))
        gain, _ = _gain(normalized, token_base, attention)
        token_parts.append(token_base.float().cpu())
        context_parts.append(attention.float().cpu())
        gains.append(gain.float().cpu())
    return torch.cat(token_parts).reshape(-1, D), torch.cat(context_parts).reshape(-1, D), \
        torch.cat(gains).reshape(-1, 1)


@torch.no_grad()
def _reference_moments(model, fit_rows: torch.Tensor, device: torch.device):
    token_cpu, context_cpu, gain_cpu = _capture_inputs(model, fit_rows, device)
    token_mean = token_cpu.mean(0).to(device)
    context_mean = context_cpu.mean(0).to(device)
    gain_mean = float(gain_cpu.mean())
    left = model.transformer.h[0].mlp.Left.weight.detach().float()
    right = model.transformer.h[0].mlp.Right.weight.detach().float()
    down = model.transformer.h[0].mlp.Down.weight.detach().float()
    token_self = torch.zeros(D, device=device)
    context_self = torch.zeros(D, device=device)
    count = 0
    for start in range(0, len(token_cpu), 256):
        token = token_cpu[start:start + 256].to(device) - token_mean
        context = context_cpu[start:start + 256].to(device) - context_mean
        token_self += _T(token, token, left, right, down).sum(0)
        context_self += _T(context, context, left, right, down).sum(0)
        count += len(token)
    token_self /= count
    context_self /= count
    total_mean = token_mean + context_mean
    mu_g = _T(total_mean[None], total_mean[None], left, right, down)[0] \
        + token_self + context_self
    return {
        "token_mean": token_mean,
        "context_mean": context_mean,
        "token_self_mean": token_self,
        "context_self_mean": context_self,
        "mu_g": mu_g,
        "gain_mean": gain_mean,
        "fit_positions": count,
        "fit_gain_mean": gain_mean,
        "fit_gain_p05": float(gain_cpu.quantile(.05)),
        "fit_gain_p95": float(gain_cpu.quantile(.95)),
    }


@torch.no_grad()
def _components(token_base: torch.Tensor, attention_write: torch.Tensor,
                normalized: torch.Tensor, reference: dict, left: torch.Tensor,
                right: torch.Tensor, down: torch.Tensor):
    token_delta = token_base.float() - reference["token_mean"]
    context_delta = attention_write.float() - reference["context_mean"]
    total_mean = reference["token_mean"] + reference["context_mean"]
    lt, rt = F.linear(token_delta, left), F.linear(token_delta, right)
    lc, rc = F.linear(context_delta, left), F.linear(context_delta, right)
    lm, rm = F.linear(total_mean, left), F.linear(total_mean, right)
    token_main = F.linear(lt * rm + lm * rt + lt * rt, down) \
        - reference["token_self_mean"]
    context_main = F.linear(lc * rm + lm * rc + lc * rc, down) \
        - reference["context_self_mean"]
    interaction = F.linear(lt * rc + lc * rt, down)
    g = reference["mu_g"] + token_main + context_main + interaction
    gain, collinearity = _gain(normalized, token_base, attention_write)
    gain_mean = reference["gain_mean"]
    branches = {
        "T": gain_mean * token_main,
        "C": gain_mean * context_main,
        "I": gain_mean * interaction,
        "S": (gain - gain_mean) * g,
    }
    retained_constant = gain_mean * reference["mu_g"]
    return retained_constant, branches, g, gain, collinearity


@torch.no_grad()
def _score_role(model, rows: torch.Tensor, device: torch.device, reference: dict):
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade

    left = model.transformer.h[0].mlp.Left.weight.detach().float()
    right = model.transformer.h[0].mlp.Right.weight.detach().float()
    down = model.transformer.h[0].mlp.Down.weight.detach().float()
    labels = {subset: _arm_name(subset) for subset in ARMS}
    document_ce = {label: [] for label in labels.values()}
    calls = {label: {"forwards": 0, "attention": 0, "site0": 0, "other_mlp": 0}
             for label in labels.values()}
    diagnostics = {"identity_num": 0.0, "identity_den": 0.0,
                   "deployed_num": 0.0, "deployed_den": 0.0,
                   "collinearity_sum": 0.0, "collinearity_count": 0,
                   "collinearity_max": 0.0}
    gram = torch.zeros(4, 4, dtype=torch.float64)
    gram_count = 0
    current = {"label": None}
    for start in range(0, len(rows), DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        block0 = model.transformer.h[0]
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        for subset in ARMS:
            label = labels[subset]
            current["label"] = label

            def attention(event, label=label):
                calls[label]["attention"] += 1
                return event.block.attn(event.state, event.first_value)

            def mlp(event, subset=subset, label=label):
                nonlocal gram_count
                if event.site != 0:
                    calls[label]["other_mlp"] += 1
                    return event.block.mlp(event.state)
                calls[label]["site0"] += 1
                native = event.block.mlp(event.state)
                retained, branches, g, gain, collinearity = _components(
                    token_base, event.attention_write, event.state,
                    reference, left, right, down)
                omitted = sum((branches[name] for name in BRANCHES if name not in subset),
                              start=torch.zeros_like(branches["T"]))
                if len(subset) == len(BRANCHES):
                    analytical = retained + sum(branches.values())
                    direct = _T(event.state, event.state, left, right, down)
                    deployed = native.float() - event.block.mlp.Down_bias.detach().float()
                    diagnostics["identity_num"] += float(
                        (analytical.double() - direct.double()).square().sum())
                    diagnostics["identity_den"] += float(direct.double().square().sum())
                    diagnostics["deployed_num"] += float(
                        (analytical.double() - deployed.double()).square().sum())
                    diagnostics["deployed_den"] += float(deployed.double().square().sum())
                    diagnostics["collinearity_sum"] += float(collinearity.sum())
                    diagnostics["collinearity_count"] += collinearity.numel()
                    diagnostics["collinearity_max"] = max(
                        diagnostics["collinearity_max"], float(collinearity.max()))
                    flat = torch.stack([
                        branches[name][:, SCORING].double().reshape(-1, D)
                        for name in BRANCHES])
                    gram += torch.einsum("and,bnd->ab", flat, flat).cpu()
                    gram_count += flat.shape[1]
                return native - omitted.to(native.dtype)

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            losses = F.cross_entropy(
                logits[:, SCORING].transpose(1, 2), targets[:, SCORING], reduction="none").mean(1)
            document_ce[label].extend(float(loss) for loss in losses)
            calls[label]["forwards"] += 1
            current["label"] = None

    expected = len(rows) // DOCUMENT_BATCH
    wanted = {"forwards": expected, "attention": 18 * expected,
              "site0": expected, "other_mlp": 17 * expected}
    live_census = all(calls[label] == wanted and len(document_ce[label]) == len(rows)
                      for label in document_ce)
    pooled_ce = {label: float(torch.tensor(values, dtype=torch.float64).mean())
                 for label, values in document_ce.items()}
    performance = {subset: -pooled_ce[labels[subset]] for subset in ARMS}
    shapley = _shapley(performance)
    diagonal = gram.diag().clamp_min(1e-30).sqrt()
    return {
        "pooled_ce": pooled_ce,
        "full_minus_empty_ce_benefit": pooled_ce["EMPTY"] - pooled_ce["T+C+I+S"],
        "shapley_ce_benefit": shapley,
        "mobius_negative_ce": _mobius(performance),
        "component_gram_per_position": (gram / max(gram_count, 1)).tolist(),
        "component_correlations": (gram / (diagonal[:, None] * diagonal[None, :])).tolist(),
        "diagnostics": {
            "analytical_identity_relative_mse": diagnostics["identity_num"]
            / max(diagnostics["identity_den"], 1e-30),
            "deployed_bf16_residual_relative_mse": diagnostics["deployed_num"]
            / max(diagnostics["deployed_den"], 1e-30),
            "mean_state_collinearity_relative_mse": diagnostics["collinearity_sum"]
            / max(diagnostics["collinearity_count"], 1),
            "max_state_collinearity_relative_mse": diagnostics["collinearity_max"],
            "live_census": live_census,
        },
        "calls": calls,
    }


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(ARMS) == 16 and set(BRANCHES) == {"T", "C", "I", "S"}
        assert REFERENCE_VALUES == 5761 and NATIVE_VALUES == 15_926_400
        print("MLP0 CENTERED CONTEXT ANOVA | dry run: moments, 16 arms, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    reference = _reference_moments(model, fit_rows, device)
    roles = {
        "FIT": _score_role(model, fit_rows, device, reference),
        "SELECT": _score_role(model, select_rows, device, reference),
    }
    fit_shapley = roles["FIT"]["shapley_ce_benefit"]
    select_shapley = roles["SELECT"]["shapley_ce_benefit"]
    fit_order = sorted(BRANCHES, key=lambda name: fit_shapley[name], reverse=True)
    select_order = sorted(BRANCHES, key=lambda name: select_shapley[name], reverse=True)
    identities_hold = all(
        roles[role]["diagnostics"]["analytical_identity_relative_mse"] <= 1e-8
        and roles[role]["diagnostics"]["live_census"] for role in roles)
    pred_a = identities_hold
    pred_b = select_shapley["C"] >= .50 and select_shapley["I"] >= .10
    pred_c = abs(select_shapley["S"]) >= .05 and (
        select_shapley["C"] + select_shapley["I"] + select_shapley["S"]
        >= select_shapley["T"])
    signs_match = all((fit_shapley[name] >= 0) == (select_shapley[name] >= 0)
                      for name in BRANCHES)
    pred_d = signs_match and fit_order == select_order
    combined_context = select_shapley["C"] + select_shapley["I"] + select_shapley["S"]
    sign_reversal = any(
        abs(fit_shapley[name]) >= .05
        and (fit_shapley[name] >= 0) != (select_shapley[name] >= 0)
        for name in ("C", "I", "S"))
    strong_null = (
        combined_context <= .50
        or (abs(select_shapley["I"]) <= .02 and abs(select_shapley["S"]) <= .02)
        or not identities_hold
        or sign_reversal)
    context_values = {name: abs(select_shapley[name]) for name in ("C", "I", "S")}
    next_branch = max(context_values, key=context_values.get)
    result = {
        "status": "mlp0_centered_context_anova_factorial_complete",
        "rung": 400,
        "claim_level": "causal_context_attribution_not_compression",
        "definition": {
            "T": "fixed-gain centered token main effect",
            "C": "fixed-gain centered continuous-context main effect",
            "I": "fixed-gain centered token-context bilinear interaction",
            "S": "observed RMS-gain modulation of the complete numerator",
        },
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows), "FINAL_opened": 0},
        "positions_per_document": SCORING.stop - SCORING.start,
        "literal_diagnostic_storage": {
            "native_values_reused": NATIVE_VALUES,
            "reference_values_added": REFERENCE_VALUES,
            "compression_claim": False,
        },
        "reference": {key: value for key, value in reference.items()
                      if not isinstance(value, torch.Tensor)},
        "roles": roles,
        "transport": {"signs_match": signs_match, "fit_order": fit_order,
                      "select_order": select_order, "rank_order_matches": fit_order == select_order},
        "select_combined_context_shapley": combined_context,
        'pred_a_exact_identity_and_live_census': bool(pred_a),
        'pred_b_context_main_and_centered_cross_are_large': bool(pred_b),
        'pred_c_normalization_matters_and_context_dominates': bool(pred_c),
        'pred_d_fit_select_sign_and_order_transport': bool(pred_d),
        "null_context_anova_not_stable_or_not_substantial": bool(strong_null),
        "next_context_branch_by_absolute_shapley": next_branch,
        "compression_or_adoption_licensed": False,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 CENTERED CONTEXT ANOVA DONE", flush=True)


if __name__ == "__main__":
    main()
