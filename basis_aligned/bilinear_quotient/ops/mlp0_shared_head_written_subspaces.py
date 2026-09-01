"""RUNG 416 -- GAUGE-STABLE SHARED HEAD-WRITTEN SUBSPACES FOR MLP0 I/C.

Rung401 assigned SELECT CE benefit (nat) I=.125540, T=.081710,
C=.054333, S=.000774.  Rung402 then split I by architectural head, but a
head label is not itself a semantic direction.  Here every head is first mapped
through its native output projection into the common 1152-dimensional residual
stream.  Fit four frozen rank-64 bases there:

  SHARED: top covariance modes after pooling the nine centered head writes;
  TOTAL:  top covariance modes of their exact summed attention write;
  HEAD3:  top covariance modes of the dominant rung402 head only;
  HAAR:   seeded random orthonormal control.

For each basis P=UU^T, replace the full centered context c by Pc inside the
exact token-context branch I and the centered context-only branch C.  Score
both singleton and removal endpoints physically, with all other MLP0 branches
fixed.  This tests a common output subspace which several heads can write to;
it does not assume independent heads and it does not allocate head-pair effects.

Frozen predictions
------------------
pred_a: all U are orthonormal, live hook census holds, and FULL/ZERO_I/ZERO_C
    reproduce rung401 SELECT within 1e-6.
pred_b: SHARED transports across the two FIT halves with projector overlap
    >=.50 and has SELECT median effective writer count >=3.
pred_c: SHARED I endpoint-average benefit >=.025 nat, beats HAAR by >=.015,
    and heldout I relative-MSE improvement >=.20.
pred_d: SHARED is the best I basis and TOTAL the best C basis, each beating the
    other by >=.003 nat in endpoint-average benefit.

Strong null: instrument/boundary failure; half-overlap<.10; median writers<2;
SHARED-I endpoint average<.01; or SHARED fails to beat HAAR in both CE and
squared error.  This is a diagnostic/identification screen, not compression or
adoption.  One stored rank-64 basis costs 73,728 scalar values.  FINAL is sealed.
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
OUT = BQ / "mlp0_shared_head_written_subspaces_results.json"
PARENT = BQ / "mlp0_centered_context_anova_exact_residual_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
D = 1152
N_HEAD = 9
HEAD_DIM = 128
RANK = 64
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
BASES = ("SHARED", "TOTAL", "HEAD3", "HAAR")
ARMS = (
    "FULL", "ZERO_I", "ZERO_C",
    *(f"I_{name}" for name in BASES), *(f"DROP_I_{name}" for name in BASES),
    *(f"C_{name}" for name in BASES), *(f"DROP_C_{name}" for name in BASES),
)


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
    heads = torch.stack([
        F.linear(joined[..., h * HEAD_DIM:(h + 1) * HEAD_DIM].float(),
                 weight[:, h * HEAD_DIM:(h + 1) * HEAD_DIM])
        for h in range(N_HEAD)
    ])
    return write, next_value, heads


@torch.no_grad()
def _head_means(model, rows, device):
    block0 = model.transformer.h[0]
    total = torch.zeros(N_HEAD, D, dtype=torch.float64, device=device)
    count = 0
    for start in range(0, len(rows), DOCUMENT_BATCH):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)
        token = (block0.lambdas[0] + block0.lambdas[1]) * F.rms_norm(
            model.transformer.wte(tokens), (D,))
        _, _, heads = _attention0_with_heads(block0, F.rms_norm(token, (D,)), None)
        total += heads.double().sum((1, 2))
        count += heads.shape[1] * heads.shape[2]
    return (total / count).float(), count


@torch.no_grad()
def _fit_bases(model, rows, device, head_means, context_mean):
    block0 = model.transformer.h[0]
    cov = {name: torch.zeros(D, D, device=device) for name in ("SHARED", "TOTAL", "HEAD3")}
    half_cov = [torch.zeros(D, D, device=device), torch.zeros(D, D, device=device)]
    counts = [0, 0]
    for start in range(0, len(rows), DOCUMENT_BATCH):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)
        token = (block0.lambdas[0] + block0.lambdas[1]) * F.rms_norm(
            model.transformer.wte(tokens), (D,))
        write, _, heads = _attention0_with_heads(block0, F.rms_norm(token, (D,)), None)
        centered = heads.float() - head_means[:, None, None, :]
        flat_heads = centered.reshape(N_HEAD, -1, D)
        for h in range(N_HEAD):
            x = flat_heads[h]
            gram = x.T @ x
            cov["SHARED"].add_(gram)
            half_cov[0 if start < len(rows) // 2 else 1].add_(gram)
        total = (write.float() - context_mean).reshape(-1, D)
        head3 = flat_heads[3]
        cov["TOTAL"].add_(total.T @ total)
        cov["HEAD3"].add_(head3.T @ head3)
        half = 0 if start < len(rows) // 2 else 1
        counts[half] += total.shape[0]

    def top_basis(matrix):
        values, vectors = torch.linalg.eigh(matrix)
        return vectors[:, -RANK:].contiguous(), values[-RANK:].flip(0)

    bases, spectra = {}, {}
    for name in ("SHARED", "TOTAL", "HEAD3"):
        bases[name], spectra[name] = top_basis(cov[name])
    generator = torch.Generator(device=device).manual_seed(416)
    bases["HAAR"] = torch.linalg.qr(
        torch.randn(D, RANK, generator=generator, device=device), mode="reduced").Q
    spectra["HAAR"] = torch.zeros(RANK, device=device)
    half_bases = [top_basis(matrix)[0] for matrix in half_cov]
    half_overlap = float((half_bases[0].T @ half_bases[1]).square().sum() / RANK)
    orthogonality = {
        name: float((basis.T @ basis - torch.eye(RANK, device=device)).abs().max())
        for name, basis in bases.items()
    }
    return bases, spectra, half_overlap, orthogonality, counts


@torch.no_grad()
def _fit_context_self_means(model, rows, device, reference, bases, base):
    block0 = model.transformer.h[0]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    sums = {name: torch.zeros(D, device=device) for name in BASES}
    count = 0
    for start in range(0, len(rows), DOCUMENT_BATCH):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)
        token = (block0.lambdas[0] + block0.lambdas[1]) * F.rms_norm(
            model.transformer.wte(tokens), (D,))
        write, _ = block0.attn(F.rms_norm(token, (D,)), None)
        context = write.float() - reference["context_mean"]
        for name, basis in bases.items():
            projected = (context @ basis) @ basis.T
            sums[name] += base._T(projected, projected, left, right, down).sum((0, 1))
        count += context.shape[0] * context.shape[1]
    return {name: value / count for name, value in sums.items()}, count


def _partial_branches(token_delta, context_delta, reference, basis, self_mean,
                      left, right, down):
    projected = (context_delta @ basis) @ basis.T
    lt, rt = F.linear(token_delta, left), F.linear(token_delta, right)
    lc, rc = F.linear(projected, left), F.linear(projected, right)
    total_mean = reference["token_mean"] + reference["context_mean"]
    lm, rm = F.linear(total_mean, left), F.linear(total_mean, right)
    gain = reference["gain_mean"]
    interaction = gain * F.linear(lt * rc + lc * rt, down)
    context = gain * (F.linear(lc * rm + lm * rc + lc * rc, down) - self_mean)
    return interaction, context


@torch.no_grad()
def _score_select(model, rows, device, reference, head_means, bases, self_means, base):
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade

    block0 = model.transformer.h[0]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    document_ce = {arm: [] for arm in ARMS}
    calls = {arm: {"forwards": 0, "attention": 0, "site0": 0, "other_mlp": 0}
             for arm in ARMS}
    errors = {branch: {name: [0.0, 0.0] for name in BASES} for branch in ("I", "C")}
    writer_energy = torch.zeros(N_HEAD, RANK, dtype=torch.float64, device=device)
    state_max = 0.0

    for start in range(0, len(rows), DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * F.rms_norm(
            model.transformer.wte(tokens), (D,))
        attn_state = F.rms_norm(token_base, (D,))
        write, first_value, heads = _attention0_with_heads(block0, attn_state, None)
        normalized = F.rms_norm(token_base + write, (D,))
        native = block0.mlp(normalized)
        _, branches, _, _, _ = base._components(
            token_base, write, normalized, reference, left, right, down)
        token_delta = token_base.float() - reference["token_mean"]
        context_delta = write.float() - reference["context_mean"]
        partial = {"I": {}, "C": {}}
        for name, basis in bases.items():
            pi, pc = _partial_branches(
                token_delta, context_delta, reference, basis, self_means[name],
                left, right, down)
            partial["I"][name], partial["C"][name] = pi, pc
            for branch, value in (("I", pi), ("C", pc)):
                errors[branch][name][0] += float(
                    (branches[branch].double() - value.double()).square().sum())
                errors[branch][name][1] += float(branches[branch].double().square().sum())

        centered_heads = heads.float() - head_means[:, None, None, :]
        coefficients = centered_heads @ bases["SHARED"]
        writer_energy += coefficients.double().square().sum((1, 2))

        cached = {"FULL": native,
                  "ZERO_I": native - branches["I"].to(native.dtype),
                  "ZERO_C": native - branches["C"].to(native.dtype)}
        for branch in ("I", "C"):
            for name in BASES:
                value = partial[branch][name]
                cached[f"{branch}_{name}"] = native - (
                    branches[branch] - value).to(native.dtype)
                cached[f"DROP_{branch}_{name}"] = native - value.to(native.dtype)

        for arm in ARMS:
            def attention(event, arm=arm):
                calls[arm]["attention"] += 1
                if event.site == 0:
                    return write, first_value
                return event.block.attn(event.state, event.first_value)

            def mlp(event, arm=arm):
                nonlocal state_max
                if event.site == 0:
                    calls[arm]["site0"] += 1
                    state_max = max(state_max, float(
                        (event.state.float() - normalized.float()).abs().max()))
                    return cached[arm]
                calls[arm]["other_mlp"] += 1
                return event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            losses = F.cross_entropy(
                logits[:, SCORING].float().transpose(1, 2), targets[:, SCORING],
                reduction="none").mean(1)
            document_ce[arm].extend(float(loss) for loss in losses)
            calls[arm]["forwards"] += 1

    expected = len(rows) // DOCUMENT_BATCH
    wanted = {"forwards": expected, "attention": 18 * expected,
              "site0": expected, "other_mlp": 17 * expected}
    live = all(calls[arm] == wanted and len(document_ce[arm]) == len(rows) for arm in ARMS)
    pooled = {arm: float(torch.tensor(values, dtype=torch.float64).mean())
              for arm, values in document_ce.items()}
    endpoints = {branch: {} for branch in ("I", "C")}
    improvement = {branch: {} for branch in ("I", "C")}
    for branch in ("I", "C"):
        for name in BASES:
            singleton = pooled[f"ZERO_{branch}"] - pooled[f"{branch}_{name}"]
            removal = pooled[f"DROP_{branch}_{name}"] - pooled["FULL"]
            endpoints[branch][name] = {
                "singleton_benefit": singleton,
                "removal_benefit": removal,
                "endpoint_average_benefit": (singleton + removal) / 2,
            }
            numerator, denominator = errors[branch][name]
            improvement[branch][name] = 1.0 - numerator / max(denominator, 1e-30)

    shares = writer_energy / writer_energy.sum(0, keepdim=True).clamp_min(1e-30)
    effective = 1.0 / shares.square().sum(0).clamp_min(1e-30)
    return {
        "pooled_ce": pooled,
        "endpoint_benefits": endpoints,
        "heldout_relative_mse_improvement": improvement,
        "shared_mode_effective_writer_count": effective.cpu().tolist(),
        "median_effective_writer_count": float(effective.median()),
        "fraction_modes_effective_writers_ge_3": float((effective >= 3).float().mean()),
        "shared_mode_head_energy_share": shares.cpu().tolist(),
        "diagnostics": {"live_census": live, "state_max_abs_error": state_max},
        "calls": calls,
    }


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(ARMS) == 19 and N_HEAD * HEAD_DIM == D
        assert RANK * D == 73_728 and len(set(ARMS)) == len(ARMS)
        print("MLP0 SHARED HEAD SUBSPACES | dry run: 19 arms, rank64, controls valid")
        return

    started = time.time()
    parent = json.loads(PARENT.read_text())
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
    head_means, head_mean_positions = _head_means(model, fit_rows, device)
    bases, spectra, half_overlap, orthogonality, half_counts = _fit_bases(
        model, fit_rows, device, head_means, reference["context_mean"])
    self_means, self_mean_positions = _fit_context_self_means(
        model, fit_rows, device, reference, bases, base)
    select = _score_select(
        model, select_rows, device, reference, head_means, bases, self_means, base)

    parent_ce = parent["roles"]["SELECT"]["pooled_ce"]
    boundaries = {
        "FULL": abs(select["pooled_ce"]["FULL"] - parent_ce["T+C+I+S"]),
        "ZERO_I": abs(select["pooled_ce"]["ZERO_I"] - parent_ce["T+C+S"]),
        "ZERO_C": abs(select["pooled_ce"]["ZERO_C"] - parent_ce["T+I+S"]),
    }
    exact = (max(orthogonality.values()) <= 2e-5
             and select["diagnostics"]["live_census"]
             and select["diagnostics"]["state_max_abs_error"] == 0.0
             and max(boundaries.values()) <= 1e-6)
    median_writers = select["median_effective_writer_count"]
    i_endpoint = {name: select["endpoint_benefits"]["I"][name]["endpoint_average_benefit"]
                  for name in BASES}
    c_endpoint = {name: select["endpoint_benefits"]["C"][name]["endpoint_average_benefit"]
                  for name in BASES}
    i_mse = select["heldout_relative_mse_improvement"]["I"]
    pred_a = exact
    pred_b = half_overlap >= .50 and median_writers >= 3
    pred_c = (i_endpoint["SHARED"] >= .025
              and i_endpoint["SHARED"] - i_endpoint["HAAR"] >= .015
              and i_mse["SHARED"] >= .20)
    pred_d = (max(i_endpoint, key=i_endpoint.get) == "SHARED"
              and max(c_endpoint, key=c_endpoint.get) == "TOTAL"
              and i_endpoint["SHARED"] - i_endpoint["TOTAL"] >= .003
              and c_endpoint["TOTAL"] - c_endpoint["SHARED"] >= .003)
    strong_null = (
        not exact or half_overlap < .10 or median_writers < 2
        or i_endpoint["SHARED"] < .01
        or (i_endpoint["SHARED"] <= i_endpoint["HAAR"]
            and i_mse["SHARED"] <= i_mse["HAAR"]))

    result = {
        "status": "mlp0_shared_head_written_subspaces_complete",
        "rung": 416,
        "claim_level": "gauge_stable_shared_subspace_screen_not_compression",
        "definition": {
            "I": "fixed-gain centered token-by-context bilinear branch",
            "C": "fixed-gain centered context-only linear-plus-quadratic branch",
            "endpoint_average": "mean of singleton recovery and removal damage; not Shapley",
            "effective_writer_count": "inverse squared head-energy shares per residual-stream mode",
            "gauge_scope": "SHARED and TOTAL use output-projected writes and are invariant to private within-head OV coordinates and head permutation; HEAD3 is a label-dependent comparator",
        },
        "documents": {"FIT_basis": len(fit_rows), "SELECT_physical": len(select_rows),
                      "FINAL_opened": 0},
        "positions_per_document": SCORING.stop - SCORING.start,
        "rank": RANK,
        "literal_diagnostic_price_per_basis_values": D * RANK,
        "fit": {
            "head_mean_positions": head_mean_positions,
            "context_self_mean_positions": self_mean_positions,
            "half_position_counts": half_counts,
            "shared_half_projector_overlap": half_overlap,
            "orthogonality_max_abs": orthogonality,
            "top64_covariance_eigenvalues": {
                name: spectra[name].cpu().tolist() for name in ("SHARED", "TOTAL", "HEAD3")},
        },
        "select": select,
        "parent_boundary_absolute_differences": boundaries,
        "best_basis": {"I": max(i_endpoint, key=i_endpoint.get),
                       "C": max(c_endpoint, key=c_endpoint.get)},
        'pred_a_exact_live_boundaries': bool(pred_a),
        'pred_b_shared_multhead_and_split_stable': bool(pred_b),
        'pred_c_shared_modes_causally_material_for_I': bool(pred_c),
        'pred_d_I_and_C_prefer_different_bases': bool(pred_d),
        'null_shared_residual_subspace_not_identified': bool(strong_null),
        "next_step": (
            "source_position_and_token_resolution_plus_causal_response_weighted_basis"
            if pred_a and pred_b and pred_c and not strong_null
            else "retain_exact_head_attribution_and_test_nonlinear_context_modes"),
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 SHARED HEAD SUBSPACES DONE", flush=True)


if __name__ == "__main__":
    main()
