#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_hessian_symmetric pred_b_diagonal_matches_prior_expansion pred_c_offdiag_significant pred_d_noun_verb_pair_dominates pred_e_eigenfilter_concentrated
"""Mixed partials / eigenfilters on the number circuit (v606; approved 19 Sep 22:12 UTC; Logan asked what this means -- explained in the docstring
since he said he forgot).

FIX (v606, after v603's instrument bug): v603 perturbed a position's embedding by a positive RESCALING, e*(1+alpha) -- exactly in the kernel of
RMSNorm's derivative (d/dc RMSNorm(c*v) = 0 for c > 0, since RMSNorm is scale-invariant), so every Hessian entry was forced to zero regardless of the
model, and the "finding" was a bug, not a null result. Fixed here by perturbing ADDITIVELY in a direction NOT parallel to the token's own embedding: a
fixed (seeded, reproducible) isotropic random unit vector r_t per position, x_t(alpha) = e_t + alpha_t * ||e_t|| * r_t (scaled by the token's own norm
so the perturbation size is comparable across positions/tokens). This survives RMSNorm generically (only the component of r_t parallel to e_t is
"used up" by the norm; the orthogonal part, almost all of a random vector's mass in D=1152 dimensions, is not).

WHAT THIS IS: the exact eight-term degree expansion (v401-v449, run all day) is a FIRST-order-style account -- it asks "how much does the noun's own
state change the margin," one source at a time. Mixed partials ask the next question: does perturbing position i and perturbing position j INTERACT --
does the margin's response to i change depending on what j is doing? Formally, for a scalar output f (the they-he margin) and per-position embedding-
scale perturbations alpha_1..alpha_T (position t's embedding is scaled by (1+alpha_t)), the Hessian H_ij = d^2f / d(alpha_i) d(alpha_j) at alpha=0 is
EXACTLY zero for two positions whose effects are purely additive, and nonzero exactly where they are not -- this is "test nonadditivity at its source"
(better_circuits 3.6) done directly by calculus instead of by trying edit combinations one pair at a time. Because bilin18 is an exactly polynomial
function of the token embeddings, autograd's second derivative here is EXACT (no finite-difference approximation), not merely a local linear estimate
the way a single gradient would be. Eigendecomposing H (a T x T symmetric matrix per sentence) gives "eigenfilters": the dominant PATTERNS of joint
position-interaction (an eigenvector concentrated on {noun, verb} says those two positions interact as a pair; a spread-out eigenvector says the
interaction is diffuse across many positions).

METHOD: for each of a few natural sentences (v406's natural rows, they - he margin at the final position), scale-perturb every position's embedding by
alpha_t, run the REAL model forward (native module calls, not a manual reimplementation -- differentiability is the only requirement here, so this
reuses attn(xin, v1_) and mlp(xin) directly rather than re-deriving squared_attention), and take the exact Hessian of the margin w.r.t. alpha via
torch.autograd.functional.hessian. T ~ 24 positions, so this is ~T extra backward passes per sentence (cheap): NOT the eight-term expansion's per-writer
symbolic algebra, a direct numerical Hessian instead.
PREDICTIONS (scored as written; failures preserved; priors from the exact degree expansion and the noun-verb dependency structure)
    pred_a_hessian_symmetric        H is symmetric to float32 precision (|H - H^T| <= 1e-2 max) -- instrument, must hold for any true Hessian
    pred_b_diagonal_matches_prior_expansion  the diagonal entry H_ii at the noun position has the same sign as the noun's own linear effect (the
                                     margin's gradient w.r.t. alpha_noun) for every sentence -- the curvature does not flip the sign of the linear story
    pred_c_offdiag_significant      at least one off-diagonal |H_ij| (i != j) exceeds 0.10 x the diagonal's RMS, for every sentence -- genuine
                                     nonadditivity exists somewhere, not just noise. Prior: likely (the whole circuit is built from products, not sums)
    pred_d_noun_verb_pair_dominates the largest-magnitude off-diagonal entry involves the noun position, for at least 3 of 4 sentences. Prior: unsure
    pred_e_eigenfilter_concentrated the top eigenvector of H (by |eigenvalue|) has its mass (sum of squared entries) at least 0.30 on its single
                                     largest-magnitude position -- the dominant interaction mode is not perfectly diffuse. Prior: unsure
PRICE (registered maximum): 4 sentences x (1 forward + ~24 Hessian-backward passes) ~= 100 forward-equivalents; 0 fits. Bar <= 110.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_value_copy_writers_v406 as v406
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mixed_partials_number_fixed_v606_result.json"
CANDIDATE_ID = "eigenfilters.mixed_partials_number_fixed_v606"
N_SENT = 4
REPLAY_TOL, SYM_TOL, OFFDIAG_MIN, CONC_MIN = 1e-3, 1e-2, 0.10, 0.30
FORWARDS_MAX = 110
PREDICTIONS = {"pred_a_hessian_symmetric": "<= 1e-2", "pred_b_diagonal_matches_prior_expansion": "same sign x N", "pred_c_offdiag_significant": ">= 0.10x RMS x N",
               "pred_d_noun_verb_pair_dominates": ">= 3/4", "pred_e_eigenfilter_concentrated": ">= 0.30"}


def main() -> None:
    recs = [r_ for p_ in v406.NATURAL for r_ in json.loads(p_.read_text())["rows"]][:N_SENT]
    plan = {"candidate_id": CANDIDATE_ID, "sentences": len(recs), "forwards_max": FORWARDS_MAX, "model_backwards": "hessian_via_autograd", "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "sym_tol": SYM_TOL, "offdiag_min": OFFDIAG_MIN, "conc_min": CONC_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; THEY, HE = L._single(" they"), L._single(" he")
    forwards = 0
    per_sentence = []
    for r_ in recs:
        tokens = torch.tensor([r_["ids"]], device="cuda"); T = tokens.shape[1]; noun_pos = r_["cue_offset"]
        e0 = model.transformer.wte(tokens).detach().float()   # [1, T, D], fixed (not itself differentiated -- alpha scales it)

        g = torch.Generator(device="cuda").manual_seed(1000003 + hash(tuple(r_["ids"])) % 100000)
        R = torch.randn(T, D, generator=g, device="cuda"); R = R / R.norm(dim=-1, keepdim=True)   # fixed random direction per position, orthonormal-ish (not exactly, but generic)
        e_norm = e0.norm(dim=-1, keepdim=True)   # [1, T, 1]

        def margin_of_alpha(alpha):   # alpha: [T], additive perturbation in the fixed random direction R, scaled by each token's own embedding norm
            x = F.rms_norm(e0 + (alpha[None, :, None] * e_norm) * R[None], (D,))
            x0, v1_ = x, None
            for block in blocks:
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (D,)))
            final = F.rms_norm(x, (D,))
            z = 30.0 * torch.tanh(model.lm_head(final) / 30.0)
            return (z[0, -1, THEY] - z[0, -1, HE])

        alpha0 = torch.zeros(T, device="cuda")
        with torch.no_grad():
            ref = float(margin_of_alpha(alpha0))
        H = torch.autograd.functional.hessian(margin_of_alpha, alpha0)   # [T, T], exact via double backward
        grad = torch.autograd.functional.jacobian(margin_of_alpha, alpha0)
        forwards += T + 2
        per_sentence.append({"noun_pos": noun_pos, "T": T, "ref_margin": ref, "H": H.cpu(), "grad": grad.cpu()})

    reports = []
    for i, item in enumerate(per_sentence):
        H, grad, noun_pos, T = item["H"], item["grad"], item["noun_pos"], item["T"]
        sym_err = float((H - H.T).abs().max())
        diag = H.diagonal()
        diag_sign_match = bool((diag[noun_pos] > 0) == (grad[noun_pos] > 0)) if abs(float(grad[noun_pos])) > 1e-6 else True
        offdiag = H.clone(); offdiag.fill_diagonal_(0)
        diag_rms = float(diag.square().mean().sqrt().clamp_min(1e-9))
        max_offdiag_val = float(offdiag.abs().max()); max_offdiag_idx = [int(x) for x in (offdiag.abs() == offdiag.abs().max()).nonzero()[0]]
        offdiag_sig = max_offdiag_val >= OFFDIAG_MIN * diag_rms
        noun_involved = noun_pos in max_offdiag_idx
        evals, evecs = torch.linalg.eigh(H)
        top = evecs[:, evals.abs().argmax()]
        conc = float(top.square().max())
        reports.append({"sentence_idx": i, "noun_pos": noun_pos, "T": T, "sym_err": sym_err, "diag_sign_match": diag_sign_match,
                         "max_offdiag": max_offdiag_val, "max_offdiag_positions": max_offdiag_idx, "diag_rms": diag_rms, "offdiag_significant": offdiag_sig,
                         "noun_involved_in_max_offdiag": noun_involved, "top_eigenvalue": float(evals[evals.abs().argmax()]), "top_eigenvector_concentration": conc,
                         "H": H.tolist(), "grad": grad.tolist()})
        print(f"sentence {i}: sym_err={sym_err:.2e} max_offdiag={max_offdiag_val:.3f} (pos {max_offdiag_idx}, noun={noun_pos}) diag_rms={diag_rms:.3f} conc={conc:.3f}")

    predictions = {"pred_a_hessian_symmetric": all(r["sym_err"] <= SYM_TOL for r in reports),
                   "pred_b_diagonal_matches_prior_expansion": all(r["diag_sign_match"] for r in reports),
                   "pred_c_offdiag_significant": all(r["offdiag_significant"] for r in reports),
                   "pred_d_noun_verb_pair_dominates": sum(r["noun_involved_in_max_offdiag"] for r in reports) >= 3,
                   "pred_e_eigenfilter_concentrated": all(r["top_eigenvector_concentration"] >= CONC_MIN for r in reports)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mixed_partials_number_fixed_result_v606", "candidate_id": CANDIDATE_ID, "plan": plan, "reports": reports,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
