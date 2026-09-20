#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_tables_replay pred_b_coefficient_ranks_reproduce pred_c_vocab_fold_shrinks_input_rank pred_d_frequency_fold_shrinks_further pred_e_embedding_family_dominates
"""Embedding-forward tensor folding, rung 1 (v609; Logan 20 Sep ~03:20 UTC: "try from embedding forward instead; Codex is doing unembedding back").

THE OBJECT. Codex's lane contracts the OUTPUT mode of a bilinear layer with the readout (C = U D) and the INPUT modes with the preceding output
projections (A = L E, B = R E). This lane does the mirror image at the front of the network: contract the INPUT modes of an early bilinear MLP with
what the token embedding actually supplies. For a single token t at position 0 every quantity up to the MLP-1 input is an EXACT function of t
(rotary at position 0 is the identity, the only key is the query's own, so every attention pattern is the scalar (q.k/128)(q2.k2/128) of the token
itself), so the MLP-l input is an exact table X_l[t] in R^D over the vocabulary, x^_t = rmsnorm(X_l[t]). The token-folded tensor of MLP l is
    T'[o, t, t'] = sum_k Down[o,k] (L x^_t)_k (R x^_t')_k        (its diagonal t = t' is the single-token behaviour; off-diagonal = the L/R cross-token
                                                                  term that appears whenever attention mixes two tokens into the same MLP input)
and its energy / mode ranks are those of T folded with the token second moment Sigma = sum_t p_t x^_t x^_t^T on both input modes:
    T_Sigma = T x_2 Sigma^(1/2) x_3 Sigma^(1/2),  built ONLY through 4608x4608 Grams (Down^T Down, L Sigma L^T, R Sigma R^T, L Sigma R^T) --
    no 1152^3 tensor is ever formed (same trick as the Aug-28 MLP1/MLP2 HOSVD findings, which did Sigma = I).
Two weightings: p_t uniform over the vocabulary, and p_t = unigram frequency from the 480x513 FineWeb row cache (246k tokens, add-0.1 smoothing).
The coefficient-space ranks (Sigma = I) are recomputed as the instrument check against the Aug-28 finding (MLP1: input 937 / output 835 at 90%).
SOURCE-FAMILY CENSUS. X_1 = X_r + X_a0 + X_m0 + X_a1 exactly (pure-embedding paths through the lambda chain; attention-0's output; MLP-0's write incl.
bias; attention-1's output), so the folded energy splits into family pairs E[F,F'] = sum Gd o (L Sigma_FF L^T) o (R Sigma_F'F' R^T) with Sigma_FF =
sum_t p_t x^F_t x^F_t^T, plus a cross-pair interference remainder (reported, not predicted). early_fold.py measured lambda ratios of ~630:1 in favour
of the raw token at block 1; this census is the tensor-energy version of that statement.
PREDICTIONS (scored as written; failures preserved)
    pred_a_tables_replay                  the manual single-token tables match the model's own forward (hook at each block's MLP input) on 256 random
                                          tokens, rel-L2 <= 1e-4 (fp32), for MLP-0 and MLP-1 (instrument)
    pred_b_coefficient_ranks_reproduce    at Sigma = I, MLP-1's symmetric-form 90%-energy input-mode rank is in [900, 980] and output-mode rank in
                                          [800, 870] (the Aug-28 HOSVD finding, 937 / 835, allowing for the symmetrisation convention)
    pred_c_vocab_fold_shrinks_input_rank  MLP-1's uniform-vocabulary-folded 90% input-mode rank <= 0.80 x its coefficient-space 90% input-mode rank
                                          (the vocabulary exercises materially fewer directions than the coefficient tensor has). Prior: unsure --
                                          this is the question
    pred_d_frequency_fold_shrinks_further MLP-1's frequency-weighted 90% input-mode rank <= 0.85 x the uniform-vocabulary one. Prior: likely
    pred_e_embedding_family_dominates     at MLP-1's input (uniform fold) the pure-embedding self-pair E[r,r] >= 0.90 of the total folded energy and
                                          each of E[a0,a0], E[m0,m0], E[a1,a1] <= 0.05. Prior: expected from the 630:1 lambda ratio
Also reported (no prediction): the same ranks at 95% / 99%; MLP-0 folds; Sigma's OWN energy ranks (the trivial upper bound on any folded rank --
the fold is interesting only where the tensor's rank drops by MORE than Sigma's own spectrum forces); cross-family interference.
PRICE (registered maximum): 13 batched T=1 manual table forwards (4096 tokens each, blocks 0-1 only) + 1 batched verification forward through the
real model (256 tokens as T=1 rows) = 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import disk_guard

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_fold_v609_result.json"
OUT_PT = ROOT / "circuits/followups/embedding_forward_fold_v609_tensors.pt"
FREQ_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.fold_v609"
FORWARDS_MAX = 16
BATCH = 4096
N_VERIFY = 256
REPLAY_TOL = 1e-4
COEF_IN_BAND, COEF_OUT_BAND = (900, 980), (800, 870)
VOCAB_SHRINK, FREQ_SHRINK = 0.80, 0.85
FAMILY_MIN, OTHER_MAX = 0.90, 0.05
SMOOTH = 0.1
QUANTILES = (0.90, 0.95, 0.99)
PREDICTIONS = {"pred_a_tables_replay": "rel-L2 <= 1e-4 x 2 layers", "pred_b_coefficient_ranks_reproduce": "in [900,980] / [800,870]",
               "pred_c_vocab_fold_shrinks_input_rank": "<= 0.80 x coefficient", "pred_d_frequency_fold_shrinks_further": "<= 0.85 x uniform",
               "pred_e_embedding_family_dominates": ">= 0.90 self; others <= 0.05"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "layers": [0, 1], "batch": BATCH,
            "bars": {"replay_tol": REPLAY_TOL, "coef_in_band": COEF_IN_BAND, "coef_out_band": COEF_OUT_BAND, "vocab_shrink": VOCAB_SHRINK,
                     "freq_shrink": FREQ_SHRINK, "family_min": FAMILY_MIN, "other_max": OTHER_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; V = model.config.vocab_size
    dev = "cuda"; forwards = 0

    def rms(x):
        return F.rms_norm(x, (x.size(-1),))

    def attn_single(block, n, v1):
        """Attention at a lone position 0: rotary is the identity, the only key is the query's own token."""
        a = block.attn
        q = a.c_q(n).view(-1, H, hd); k = a.c_k(n).view(-1, H, hd); q2 = a.c_q2(n).view(-1, H, hd); k2 = a.c_k2(n).view(-1, H, hd)
        v = a.c_v(n).view(-1, H, hd)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1
        qn, kn, q2n, k2n = rms(q), rms(k), rms(q2), rms(k2)
        score = ((qn * kn).sum(-1) / hd) * ((q2n * k2n).sum(-1) / hd)                      # [N, H]
        y = (score[..., None] * v).reshape(-1, D)                                            # concatenated head outputs (pre-c_proj)
        return a.c_proj(y), v1, score

    # ---- exact single-token tables, by source family --------------------------------------------------------------------------
    b0, b1 = blocks[0], blocks[1]
    with torch.no_grad():
        lam0, lam1 = b0.lambdas.detach().float(), b1.lambdas.detach().float()
        E = model.transformer.wte.weight.detach().float()
        X0_in = torch.empty(V, D, device=dev); fam0 = {"r": torch.empty(V, D, device=dev), "a0": torch.empty(V, D, device=dev)}
        fam1 = {k: torch.empty(V, D, device=dev) for k in ("r", "a0", "m0", "a1")}
        for s in range(0, V, BATCH):
            ids = torch.arange(s, min(s + BATCH, V), device=dev)
            x0 = rms(E[ids])
            live0 = lam0[0] * x0 + lam0[1] * x0
            y0, v1, _ = attn_single(b0, rms(live0), None)
            xa = live0 + y0                                                                  # MLP-0 input (pre-norm)
            fam0["r"][ids] = live0; fam0["a0"][ids] = y0; X0_in[ids] = xa
            n0 = rms(xa); m0 = b0.mlp.Left(n0) * b0.mlp.Right(n0); w0 = b0.mlp.Down(m0) + b0.mlp.Down_bias
            x1 = xa + w0
            live1 = lam1[0] * x1 + lam1[1] * x0
            y1, _, _ = attn_single(b1, rms(live1), v1)
            fam1["r"][ids] = lam1[0] * live0 + lam1[1] * x0; fam1["a0"][ids] = lam1[0] * y0; fam1["m0"][ids] = lam1[0] * w0; fam1["a1"][ids] = y1
            forwards += 1
        X1_in = sum(fam1.values())
        # verification against the real model's own forward (hooks on the MLP inputs), 256 random tokens as T=1 rows
        gen = torch.Generator(device="cpu").manual_seed(609)
        vid = torch.randint(0, V, (N_VERIFY,), generator=gen).to(dev)
        cache = {}
        hooks = [blocks[l].mlp.register_forward_pre_hook(lambda m, a, l=l: cache.__setitem__(l, a[0].detach().float().clone())) for l in (0, 1)]
        model(vid[:, None], vid[:, None].clone()); forwards += 1
        for h_ in hooks:
            h_.remove()
        # hooked value is rmsnorm(x) (the MLP sees the normalised input); compare normalised tables
        replay = {l: float((rms(Xin[vid]) - cache[l][:, 0]).norm() / cache[l][:, 0].norm()) for l, Xin in ((0, X0_in), (1, X1_in))}
        print("replay rel-L2:", replay)

        # ---- weightings ---------------------------------------------------------------------------------------------------------
        rows = torch.load(FREQ_ROWS, map_location="cpu")
        counts = torch.bincount(rows.reshape(-1).long(), minlength=V).double()
        p_freq = (counts + SMOOTH); p_freq = (p_freq / p_freq.sum()).to(dev)
        p_unif = torch.full((V,), 1.0 / V, device=dev, dtype=torch.float64)
        print(f"unigram rows: {tuple(rows.shape)}, tokens {int(rows.numel())}, vocab seen {int((counts > 0).sum())}")

        def second_moment(Xa, Xb, p):                                                        # sum_t p_t x^a_t x^b_t^T, fp64
            return (Xa.double() * p[:, None]).T @ Xb.double()

        def energy_ranks(evals):
            ev = evals.clamp_min(0).flip(0); c = ev.cumsum(0) / ev.sum()
            return {str(q): int((c < q).sum().item()) + 1 for q in QUANTILES}

        def fold_ranks(block, Sigma):
            """Symmetric-form mode Grams of T x_2 Sigma^(1/2) x_3 Sigma^(1/2), through 4608x4608 Grams only."""
            L = block.mlp.Left.weight.detach().double(); R = block.mlp.Right.weight.detach().double(); Dw = block.mlp.Down.weight.detach().double()
            Gd = Dw.T @ Dw
            GLL, GRR, GLR = L @ Sigma @ L.T, R @ Sigma @ R.T, L @ Sigma @ R.T
            e_unsym = float((Gd * GLL * GRR).sum()); e_cross = float((Gd * GLR * GLR.T).sum())
            e_sym = 0.5 * (e_unsym + e_cross)
            M2 = 0.25 * (L.T @ (Gd * GRR) @ L + L.T @ (Gd * GLR.T) @ R + R.T @ (Gd * GLR) @ L + R.T @ (Gd * GLL) @ R)
            w, U = torch.linalg.eigh(Sigma); S_half = (U * w.clamp_min(0).sqrt()) @ U.T
            in_ev = torch.linalg.eigvalsh(S_half @ M2 @ S_half)
            out_ev = torch.linalg.eigvalsh(0.5 * Dw @ (GLL * GRR + GLR * GLR.T) @ Dw.T)
            return {"energy_sym": e_sym, "energy_unsym": e_unsym, "input_ranks": energy_ranks(in_ev), "output_ranks": energy_ranks(out_ev),
                    "input_evals_top": in_ev.flip(0)[:16].tolist()}

        def family_census(block, fams, p):
            L = block.mlp.Left.weight.detach().double(); R = block.mlp.Right.weight.detach().double(); Dw = block.mlp.Down.weight.detach().double()
            Gd = Dw.T @ Dw
            total = sum(fams.values()); s = rms(total).norm(dim=-1, keepdim=True) / total.norm(dim=-1, keepdim=True)   # per-token 1/rms scale
            hat = {k: v * s for k, v in fams.items()}
            names = list(fams); Sig = {(a, b): second_moment(hat[a], hat[b], p) for a in names for b in names}
            GL = {a: L @ Sig[(a, a)] @ L.T for a in names}; GR = {b: R @ Sig[(b, b)] @ R.T for b in names}
            pair = {f"{a}|{b}": float((Gd * GL[a] * GR[b]).sum()) for a in names for b in names}
            Sig_tot = sum(Sig[(a, b)] for a in names for b in names)
            e_tot = float((Gd * (L @ Sig_tot @ L.T) * (R @ Sig_tot @ R.T)).sum())
            return {"pair_energy": pair, "total_energy": e_tot, "pair_fraction": {k: v / e_tot for k, v in pair.items()},
                     "interference_fraction": (e_tot - sum(pair.values())) / e_tot,
                     "family_rms": {k: float(hat[k].norm() / (V ** 0.5)) for k in names}}

        report = {"replay": replay, "lambdas": {"block0": lam0.tolist(), "block1": lam1.tolist()}, "layers": {}}
        eye = torch.eye(D, device=dev, dtype=torch.float64)
        for l, Xin, fams in ((0, X0_in, fam0), (1, X1_in, fam1)):
            xh = rms(Xin)
            Sig_u, Sig_f = second_moment(xh, xh, p_unif), second_moment(xh, xh, p_freq)
            rep = {"coefficient": fold_ranks(blocks[l], eye), "vocab_uniform": fold_ranks(blocks[l], Sig_u), "vocab_frequency": fold_ranks(blocks[l], Sig_f),
                   "sigma_own_ranks": {"uniform": energy_ranks(torch.linalg.eigvalsh(Sig_u)), "frequency": energy_ranks(torch.linalg.eigvalsh(Sig_f))},
                   "family_census_uniform": family_census(blocks[l], fams, p_unif), "family_census_frequency": family_census(blocks[l], fams, p_freq)}
            report["layers"][str(l)] = rep
            print(f"MLP-{l}: coef in/out 90% = {rep['coefficient']['input_ranks']['0.9']}/{rep['coefficient']['output_ranks']['0.9']}; "
                  f"vocab-uniform in/out = {rep['vocab_uniform']['input_ranks']['0.9']}/{rep['vocab_uniform']['output_ranks']['0.9']}; "
                  f"frequency in/out = {rep['vocab_frequency']['input_ranks']['0.9']}/{rep['vocab_frequency']['output_ranks']['0.9']}; "
                  f"Sigma own 90% = {rep['sigma_own_ranks']['uniform']['0.9']}/{rep['sigma_own_ranks']['frequency']['0.9']}")
            print(f"  family fractions (uniform): { {k: round(v, 4) for k, v in rep['family_census_uniform']['pair_fraction'].items()} } "
                  f"interference {rep['family_census_uniform']['interference_fraction']:.4f}")
        disk_guard.guard_torch_save({"Sigma_uniform_mlp1": second_moment(rms(X1_in), rms(X1_in), p_unif).float().cpu(),
                                     "Sigma_frequency_mlp1": second_moment(rms(X1_in), rms(X1_in), p_freq).float().cpu(),
                                     "p_freq": p_freq.float().cpu()}, str(OUT_PT), "v609 second moments")

    r1 = report["layers"]["1"]; fam = r1["family_census_uniform"]["pair_fraction"]
    ci, co = r1["coefficient"]["input_ranks"]["0.9"], r1["coefficient"]["output_ranks"]["0.9"]
    vi, fi = r1["vocab_uniform"]["input_ranks"]["0.9"], r1["vocab_frequency"]["input_ranks"]["0.9"]
    predictions = {"pred_a_tables_replay": all(v <= REPLAY_TOL for v in replay.values()),
                   "pred_b_coefficient_ranks_reproduce": COEF_IN_BAND[0] <= ci <= COEF_IN_BAND[1] and COEF_OUT_BAND[0] <= co <= COEF_OUT_BAND[1],
                   "pred_c_vocab_fold_shrinks_input_rank": vi <= VOCAB_SHRINK * ci,
                   "pred_d_frequency_fold_shrinks_further": fi <= FREQ_SHRINK * vi,
                   "pred_e_embedding_family_dominates": fam["r|r"] >= FAMILY_MIN and all(fam[f"{k}|{k}"] <= OTHER_MAX for k in ("a0", "m0", "a1"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_fold_result_v609", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
