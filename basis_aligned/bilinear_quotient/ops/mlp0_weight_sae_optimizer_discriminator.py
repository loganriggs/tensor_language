"""MLP0 WEIGHT-ACTION SAE OPTIMIZER DISCRIMINATOR.

Why this run exists
-------------------
The historical weight-action top-k result (§750) was promising but not convergence
certified: one seed per k, a fixed 1,200-step budget, and no held-out reconstruction
curve.  Later work found atom instability across seeds (§763/§764).  Before building
a larger joint MLP0->consumer SAE/DAG, decide whether the remaining error is caused
by the amortized encoder and optimizer, or by the flat sparse-dictionary model class.

Bias correction: ``Down`` is bias-free.  The containing bilinear MLP adds the
separate parameter ``Down_bias`` *after* a Down forward hook, so the historical hook
did not omit that native bias.  Its learned ``b=mean(Wg)`` was a compressor intercept
on the linear weight action, in addition to the unchanged native ``Down_bias``.  This
run preserves and audits that distinction explicitly.

Physical object
---------------
For native MLP0 Down,

    a(g) = g W^T,
    MLP(x) = a(Left(x) * Right(x)) + Down_bias.

Fit executable programs ``a_hat(g)=D sparse_k(E g)+c`` on native MLP0 gate
inputs.  ``c`` is a learned, explicitly priced compressor intercept.  The separate
native ``Down_bias`` stays in the live MLP and is never approximated or omitted.

Registered arms
---------------
  old_relu:
      Historical positive ReLU top-k parameterization, with a learned intercept and
      the exact external native Down_bias retained.
  signed_unit:
      Select k coordinates by |E g|, retain their signs, and row-normalize E inside
      every forward.  This removes encoder scale as a way to win top-k routing and
      avoids needing paired positive/negative atoms.
  signed_unit_noise:
      The signed-unit arm trained with 3% diagonal-covariance Gaussian perturbations
      of g and exact targets W(g+eps).  This is a local robustness regularizer, not
      evidence of natural OOD generalization.

All arms use P=512, k=32, 128 fit documents, 64 untouched evaluation documents,
three seeds, minibatch Adam, 2,400 steps, and a held-out R2 curve every 200 steps.
The final replacement is evaluated by live-model CE.  No arm is selected on CE.

The cheapest classical-dictionary discriminator
------------------------------------------------
For each arm's best held-out seed, freeze D and refine evaluation codes directly by
20 iterations of iterative hard thresholding (IHT).  This is deliberately an oracle:
it is not executable from g and earns no simplicity credit.  Its only purpose is to
separate an amortized-encoder/optimization gap from a dictionary-capacity gap.

Registered decisions
--------------------
  (0) INTERFACE SANITY: float32 Wg agrees with captured native Down output to relative
      RMS <= 3e-3; Down.bias is absent; and zeroing the Down action while retaining
      Down_bias differs from cancelling the complete MLP output (proves the external
      bias is physically present in the assay).
  (a) PERFORMANCE CONVERGENCE: for the winning executable arm, held-out R2 range over
      the last three checkpoints <= .01 and final R2 is within .005 of its best.
  (b) SEED ROBUSTNESS: final held-out R2 standard deviation <= .02.  Atom cosine and
      decoder-subspace overlap are reported but are not required to pass: stable
      function with unstable atoms means the atom is not a canonical semantic unit.
  (c) OPTIMIZER/ENCODER BOTTLENECK: oracle IHT improves held-out R2 by >= .05.  If
      true, queue alternating MOD/K-SVD or learned iterative inference.  If false,
      classical sparse-code optimization cannot plausibly close much of the gap with
      this decoder and flat P/k model; prioritize joint causal readers or hierarchy.
  (d) NOISE EARNS ITS COST: signed_unit_noise improves noisy-input R2 by >= .03 over
      signed_unit while losing <= .02 clean held-out R2.  Otherwise prune noise.
  (e) EXECUTABLE VALUE: the best bias-correct arm has held-out MLP0 CE recovery >= .90
      and beats old_relu by >= .01 or matches it within .01 with better convergence.

This is Stage 0 of the joint synthesis, not the joint causal result.  It licenses a
joint producer-consumer run only after the optimizer/model-class ambiguity is closed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/workspace/rspd")

import census_lib as cl
from bilin18_joint_removal import DEV, m


D_MODEL = 1152
HIDDEN = 4608
LAYER = 0
P = 512
K = 32
NFIT = 128
NEVAL = 64
SEEDS = (0, 1, 2)
STEPS = 2400
EVAL_EVERY = 200
TOKEN_BATCH = 1024
NOISE_SIGMA = 0.03
ORACLE_SAMPLES = 4096
ORACLE_STEPS = 20
ARMS = ("old_relu", "signed_unit", "signed_unit_noise")
PT = "/workspace/tensor_language/basis_aligned/bilinear_quotient/"
OUT = PT + "ops/mlp0_weight_sae_optimizer_discriminator_results.json"
REPLACEMENT: dict[str, object | None] = {"fn": None}


def _topk(pre: torch.Tensor, *, signed: bool) -> torch.Tensor:
    order = pre.abs() if signed else pre
    _, index = order.topk(K, dim=-1)
    value = pre.gather(-1, index)
    if not signed:
        value = F.relu(value)
    result = torch.zeros_like(pre)
    return result.scatter(-1, index, value)


def _preactivation(inputs: torch.Tensor, encoder: torch.Tensor, arm: str) -> torch.Tensor:
    if arm == "old_relu":
        return inputs @ encoder.T
    return inputs @ F.normalize(encoder, dim=1).T


def _code(inputs: torch.Tensor, encoder: torch.Tensor, arm: str) -> torch.Tensor:
    return _topk(_preactivation(inputs, encoder, arm), signed=arm != "old_relu")


def _predict(
    inputs: torch.Tensor, decoder: torch.Tensor, encoder: torch.Tensor,
    bias: torch.Tensor, arm: str,
) -> torch.Tensor:
    return _code(inputs, encoder, arm) @ decoder.T + bias


def _r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    denominator = (target - target.mean(0)).square().sum().clamp_min(1e-12)
    return float(1.0 - (prediction - target).square().sum() / denominator)


def _down_hook(_module, inputs, output):
    fn = REPLACEMENT["fn"]
    if fn is None:
        return output
    gate = inputs[0].float().reshape(-1, HIDDEN)
    replacement = fn(gate).reshape(output.shape)
    return replacement.to(output.dtype)


@torch.no_grad()
def _forward_logits(tokens: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(m.transformer.wte(tokens), (D_MODEL,))
    x0, value = x, None
    for block in m.transformer.h:
        x, value = block(x, value, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D_MODEL,))) / 30.0)


@torch.no_grad()
def _capture_gate_and_native_output(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gates: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    module = m.transformer.h[LAYER].mlp.Down

    def capture(_module, inputs, output):
        gates.append(inputs[0].detach().float().reshape(-1, HIDDEN))
        outputs.append(output.detach().float().reshape(-1, D_MODEL))

    handle = module.register_forward_hook(capture)
    try:
        for start in range(0, len(rows), 4):
            tokens = rows[start:start + 4, :257].to(DEV)[:, :-1].contiguous()
            _forward_logits(tokens)
    finally:
        handle.remove()
    return torch.cat(gates), torch.cat(outputs)


@torch.no_grad()
def _ce(rows: torch.Tensor) -> float:
    total, count = 0.0, 0
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4, :257].to(DEV)
        tokens, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        logits = _forward_logits(tokens)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        )
        total += float(loss) * len(batch)
        count += len(batch)
    return total / count


@dataclass
class Fit:
    arm: str
    seed: int
    decoder: torch.Tensor
    encoder: torch.Tensor
    bias: torch.Tensor
    curve: list[dict[str, float | int]]


def _train(
    *, arm: str, seed: int, fit_x: torch.Tensor, eval_x: torch.Tensor,
    weight: torch.Tensor,
) -> Fit:
    generator = torch.Generator(device=DEV).manual_seed(2026083300 + 10 * seed + ARMS.index(arm))
    torch.manual_seed(2026083300 + 10 * seed + ARMS.index(arm))
    decoder = (torch.randn(D_MODEL, P, device=DEV) / math.sqrt(D_MODEL)).requires_grad_(True)
    encoder = (torch.randn(P, HIDDEN, device=DEV) / math.sqrt(HIDDEN)).requires_grad_(True)
    # This is a compressor intercept for the bias-free Down action.  The bilinear
    # module's separate Down_bias remains exact outside the hook.
    intercept = (fit_x @ weight.T).mean(0).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam((decoder, encoder, intercept), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS)
    feature_scale = fit_x.std(0, unbiased=False).clamp_min(1e-6)
    eval_target = eval_x @ weight.T
    curve: list[dict[str, float | int]] = []
    count = len(fit_x)
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                prediction = _predict(eval_x, decoder, encoder, intercept, arm)
                curve.append({"step": step, "eval_r2": round(_r2(prediction, eval_target), 6)})
        if step == STEPS:
            break
        index = torch.randint(0, count, (TOKEN_BATCH,), generator=generator, device=DEV)
        batch_x = fit_x[index]
        if arm == "signed_unit_noise":
            noise = torch.randn(
                batch_x.shape, generator=generator, device=DEV, dtype=batch_x.dtype,
            ) * feature_scale * NOISE_SIGMA
            batch_x = batch_x + noise
        target = batch_x @ weight.T
        prediction = _predict(batch_x, decoder, encoder, intercept, arm)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((decoder, encoder), 10.0)
        optimizer.step()
        scheduler.step()
    return Fit(
        arm=arm, seed=seed, decoder=decoder.detach().cpu(),
        encoder=encoder.detach().cpu(), bias=intercept.detach().cpu(), curve=curve,
    )


@torch.no_grad()
def _evaluate_fit(
    fit: Fit, eval_x: torch.Tensor, eval_rows: torch.Tensor,
    weight: torch.Tensor, feature_scale: torch.Tensor, ce_full: float, ce_zero: float,
) -> dict[str, object]:
    decoder = fit.decoder.to(DEV)
    encoder = fit.encoder.to(DEV)
    bias = fit.bias.to(DEV)
    target = eval_x @ weight.T
    clean = _predict(eval_x, decoder, encoder, bias, fit.arm)
    clean_r2 = _r2(clean, target)
    noise_generator = torch.Generator(device=DEV).manual_seed(2026083390)
    noise = torch.randn(
        eval_x.shape, generator=noise_generator, device=DEV, dtype=eval_x.dtype,
    ) * feature_scale * NOISE_SIGMA
    noisy_x = eval_x + noise
    noisy_target = noisy_x @ weight.T
    noisy = _predict(noisy_x, decoder, encoder, bias, fit.arm)
    noisy_r2 = _r2(noisy, noisy_target)
    clean_support = _code(eval_x, encoder, fit.arm).ne(0)
    noisy_support = _code(noisy_x, encoder, fit.arm).ne(0)
    intersection = (clean_support & noisy_support).sum(-1).double()
    union = (clean_support | noisy_support).sum(-1).double().clamp_min(1)
    support_jaccard = float((intersection / union).mean())

    def replacement(gate: torch.Tensor) -> torch.Tensor:
        return _predict(gate, decoder, encoder, bias, fit.arm)

    REPLACEMENT["fn"] = replacement
    ce_value = _ce(eval_rows)
    REPLACEMENT["fn"] = None
    ce_recovery = (ce_zero - ce_value) / max(ce_zero - ce_full, 1e-12)
    return {
        "clean_r2": round(clean_r2, 6),
        "noisy_r2": round(noisy_r2, 6),
        "support_jaccard": round(support_jaccard, 6),
        "ce": round(ce_value, 6),
        "ce_recovery": round(ce_recovery, 6),
        "curve": fit.curve,
    }


@torch.no_grad()
def _oracle_iht_r2(
    fit: Fit, eval_x: torch.Tensor, weight: torch.Tensor,
) -> float:
    decoder = fit.decoder.to(DEV)
    encoder = fit.encoder.to(DEV)
    bias = fit.bias.to(DEV)
    sample_x = eval_x[:ORACLE_SAMPLES]
    target = sample_x @ weight.T
    code = _code(sample_x, encoder, fit.arm)
    spectral = torch.linalg.matrix_norm(decoder, ord=2)
    step_size = 0.95 / spectral.square().clamp_min(1e-12)
    for _ in range(ORACLE_STEPS):
        residual = code @ decoder.T + bias - target
        code = code - step_size * (residual @ decoder)
        code = _topk(code, signed=True)
    return _r2(code @ decoder.T + bias, target)


@torch.no_grad()
def _stability(fits: list[Fit]) -> dict[str, float]:
    reference = F.normalize(fits[0].decoder.float(), dim=0)
    atom_scores, subspace_scores = [], []
    reference_u = torch.linalg.svd(fits[0].decoder.float(), full_matrices=False).U[:, :64]
    for fit in fits[1:]:
        decoder = F.normalize(fit.decoder.float(), dim=0)
        atom_scores.append(float((reference.T @ decoder).abs().max(1).values.mean()))
        u = torch.linalg.svd(fit.decoder.float(), full_matrices=False).U[:, :64]
        subspace_scores.append(float(torch.linalg.svdvals(reference_u.T @ u).mean()))
    return {
        "mean_atom_best_cosine": round(float(np.mean(atom_scores)), 6),
        "mean_rank64_subspace_overlap": round(float(np.mean(subspace_scores)), 6),
    }


def main() -> None:
    started = time.time()
    cl.use_state(PT + "census_state_diverse.pt")
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit_rows, eval_rows = rows[:NFIT], rows[NFIT:NFIT + NEVAL]
    fit_x, captured_fit_y = _capture_gate_and_native_output(fit_rows)
    eval_x, captured_eval_y = _capture_gate_and_native_output(eval_rows)
    module = m.transformer.h[LAYER].mlp.Down
    weight = module.weight.detach().float().to(DEV)
    mlp = m.transformer.h[LAYER].mlp
    if module.bias is not None:
        raise RuntimeError("expected bias-free Down linear module")
    if not hasattr(mlp, "Down_bias"):
        raise RuntimeError("bilinear MLP is missing its external Down_bias")
    external_bias = mlp.Down_bias.detach().float().to(DEV)
    exact_fit_y = fit_x @ weight.T
    exact_eval_y = eval_x @ weight.T
    affine_drift = float(
        torch.sqrt(torch.mean((captured_eval_y - exact_eval_y).square()))
        / torch.sqrt(torch.mean(captured_eval_y.square())).clamp_min(1e-12)
    )
    bias_rms = float(torch.sqrt(torch.mean(external_bias.square())))

    handle = module.register_forward_hook(_down_hook)
    REPLACEMENT["fn"] = None
    ce_full = _ce(eval_rows)
    # Down's hook fires before the containing MLP adds external Down_bias.
    REPLACEMENT["fn"] = lambda gate: torch.zeros(
        gate.shape[0], D_MODEL, device=gate.device,
    )
    ce_zero_action_bias_retained = _ce(eval_rows)
    REPLACEMENT["fn"] = lambda gate: -external_bias.expand(gate.shape[0], -1)
    ce_zero_complete_mlp = _ce(eval_rows)
    REPLACEMENT["fn"] = None
    feature_scale = fit_x.std(0, unbiased=False).clamp_min(1e-6)

    fits_by_arm: dict[str, list[Fit]] = {arm: [] for arm in ARMS}
    results: dict[str, dict[str, object]] = {arm: {} for arm in ARMS}
    try:
        for arm in ARMS:
            for seed in SEEDS:
                print(f"training {arm} seed={seed}", flush=True)
                with torch.enable_grad():
                    fit = _train(
                        arm=arm, seed=seed, fit_x=fit_x, eval_x=eval_x,
                        weight=weight,
                    )
                fits_by_arm[arm].append(fit)
                results[arm][str(seed)] = _evaluate_fit(
                    fit, eval_x, eval_rows, weight, feature_scale, ce_full,
                    ce_zero_action_bias_retained,
                )
                print(
                    f"  clean R2={results[arm][str(seed)]['clean_r2']:.4f} "
                    f"CE-rec={results[arm][str(seed)]['ce_recovery']:.4f}",
                    flush=True,
                )
    finally:
        REPLACEMENT["fn"] = None
        handle.remove()

    arm_summary: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        records = [results[arm][str(seed)] for seed in SEEDS]
        clean = np.array([record["clean_r2"] for record in records], dtype=float)
        ce_recovery = np.array([record["ce_recovery"] for record in records], dtype=float)
        best_index = int(np.argmax(clean))
        oracle = _oracle_iht_r2(fits_by_arm[arm][best_index], eval_x, weight)
        curve = records[best_index]["curve"]
        last = np.array([point["eval_r2"] for point in curve[-3:]], dtype=float)
        all_curve = np.array([point["eval_r2"] for point in curve], dtype=float)
        arm_summary[arm] = {
            "mean_clean_r2": round(float(clean.mean()), 6),
            "std_clean_r2": round(float(clean.std()), 6),
            "mean_ce_recovery": round(float(ce_recovery.mean()), 6),
            "best_seed": SEEDS[best_index],
            "best_clean_r2": round(float(clean[best_index]), 6),
            "oracle_iht_r2": round(float(oracle), 6),
            "oracle_gap": round(float(oracle - clean[best_index]), 6),
            "last_three_r2_range": round(float(last.max() - last.min()), 6),
            "final_to_best_r2_gap": round(float(all_curve.max() - all_curve[-1]), 6),
            **_stability(fits_by_arm[arm]),
        }

    winner = max(ARMS, key=lambda arm: arm_summary[arm]["mean_clean_r2"])
    winning = arm_summary[winner]
    signed = arm_summary["signed_unit"]
    noisy = arm_summary["signed_unit_noise"]
    noisy_eval = np.mean([
        results["signed_unit_noise"][str(seed)]["noisy_r2"] for seed in SEEDS
    ])
    signed_noisy_eval = np.mean([
        results["signed_unit"][str(seed)]["noisy_r2"] for seed in SEEDS
    ])
    affine_sanity = (
        affine_drift <= 3e-3
        and module.bias is None
        and abs(ce_zero_complete_mlp - ce_zero_action_bias_retained) > 1e-5
    )
    pred_a = winning["last_three_r2_range"] <= 0.01 and winning[
        "final_to_best_r2_gap"
    ] <= 0.005
    pred_b = winning["std_clean_r2"] <= 0.02
    pred_c = winning["oracle_gap"] >= 0.05
    pred_d = noisy_eval - signed_noisy_eval >= 0.03 and noisy[
        "mean_clean_r2"
    ] >= signed["mean_clean_r2"] - 0.02
    old_ce = arm_summary["old_relu"]["mean_ce_recovery"]
    winning_ce = winning["mean_ce_recovery"]
    pred_e = winning_ce >= 0.90 and (
        winning_ce >= old_ce + 0.01 or (
            abs(winning_ce - old_ce) <= 0.01 and pred_a
        )
    )
    output = {
        "status": "mlp0_weight_sae_optimizer_discriminator_complete",
        "object": (
            "native MLP0 bias-free Down action with learned compressor intercept; "
            "external native Down_bias retained exactly"
        ),
        "data": {"fit_documents": NFIT, "eval_documents": NEVAL},
        "model": {"P": P, "k": K, "seeds": list(SEEDS)},
        "training": {
            "steps": STEPS, "eval_every": EVAL_EVERY,
            "token_batch": TOKEN_BATCH, "noise_sigma": NOISE_SIGMA,
        },
        "affine_sanity": {
            "captured_vs_float32_relative_rms": affine_drift,
            "down_linear_has_bias": module.bias is not None,
            "external_down_bias_rms": bias_rms,
            "ce_full": ce_full,
            "ce_zero_down_action_bias_retained": ce_zero_action_bias_retained,
            "ce_zero_complete_mlp": ce_zero_complete_mlp,
            "passed": bool(affine_sanity),
        },
        "per_seed": results,
        "summary": arm_summary,
        "winner": winner,
        "pred_a_performance_converged": bool(pred_a),
        "pred_b_seed_robust_performance": bool(pred_b),
        "pred_c_oracle_gap_ge_005": bool(pred_c),
        "pred_d_noise_earned": bool(pred_d),
        "pred_e_executable_value": bool(pred_e),
        "decision": (
            "queue_alternating_dictionary_learning" if pred_c else
            "amortized_encoder_not_the_main_flat_dictionary_bottleneck"
        ),
        "runtime_s": time.time() - started,
    }
    with open(OUT, "w", encoding="utf-8") as handle_out:
        json.dump(output, handle_out, indent=1)
    print(json.dumps({
        "winner": winner,
        "summary": arm_summary,
        "pred_a": bool(pred_a), "pred_b": bool(pred_b),
        "pred_c": bool(pred_c), "pred_d": bool(pred_d), "pred_e": bool(pred_e),
        "decision": output["decision"],
    }, indent=2), flush=True)
    print(f"wrote {OUT} ({output['runtime_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
