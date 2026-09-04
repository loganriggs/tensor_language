# late_tail_read_operator_rank_probe — preregistration (Registered 2026-09-04 01:35Z (box clock); amended centered covariance before enqueue)

Lane 1 (Claude). Parent: ops/late_tail_writer_kind_probe.py (frames, heads, run/head machinery). Prior results frozen:
late_tail_writer_recency_probe_results.json (§2790). Script: bilinear_quotient/ops/late_tail_read_operator_rank_probe.py.
Results: bilinear_quotient/late_tail_read_operator_rank_probe_results.json. Strategic review 0131 item T2.

SIGN CONVENTION (§2135): the two CE instrument numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 (FRESH split) — LOWER IS
BETTER. Everything else is an exact weight / second-moment quantity (no CE).

## Question
§2780–§2790: each late MLP (blocks 8–17) = quadratic on the 768-dim bus core c + a core-gated LINEAR read of the tail t,
Down(Lc∘Rt + Lt∘Rc) = J(c)·t, J linear in c, supported on the last ~3–4 blocks' writes. Is this read LOW-RANK in any of its three
indices — the c-dependence (could W_l(c) be a low-rank function of c?), the tail input (which tail directions are read?), the output
(does it land in the core?) — or is it, like the token operators of §2673/§2675, high-rank everywhere? A low-rank c-dependence would
admit the first parameter-REDUCING surrogate of the lineage (P3); a high-rank one closes that route exactly.

## Method (exact)
Frames from the parent's fit pass (docs 96–191): U_8 = core_of(blocks 8–17), Uk = first 768 columns (core), Ut = remaining 384 (tail).
Per late block l: J_i = Down(diag(L u_i) R Ut + diag(R u_i) L Ut) for the 768 core generators u_i; G_ij = ⟨J_i, J_j⟩_F;
Ma = Ukᵀ Cov[xh] Uk (fit-set CENTERED covariance of the MLP input in core coordinates); gate-dependence spectrum = eig(Ma^½ G Ma^½)
(the nonzero spectrum of Cov_c[vec J(c)] — the rank of the gate's FLUCTUATION about its mean; the mean gate itself is pred_e). Amended
from the second moment before enqueue (smoke showed the mean would dominate that spectrum and double-count pred_e). Monte-Carlo over N_MC = 512 real fit-set inputs (first fit chunk, pinned
permutation): E[J(c)ᵀJ(c)] (tail-input side, 384×384), E[J(c)J(c)ᵀ] (output side, D×D; core energy fraction), constant-gate share
‖J(mean c)‖²_F / E‖J(c)‖²_F, and the exact split check ‖MLP(c+t) − MLP(c) − MLP(t) − J(c)t‖/‖·‖ on Gaussian tail t (bias-free).
Effective rank = exp(entropy of the normalised spectrum) (R.spectrum), as in §2673–§2679.

## Instruments
baseline 3.0322401 (tol 1e-4); SPLIT8_1024 .0374 and LATE_MLP_768 .1249 (tol 0.015); split check rel err ≤ 1e-3 in every block.

## Predictions (scored exactly as written)
- pred_a_instrument: the four instruments above.
- pred_b_gate_dependence_is_high_rank: min over blocks 8–17 of eff_rank(Ma^½ G Ma^½) ≥ 200 (of 768). Null: some block ≤ 64 (P3 admissible).
- pred_c_tail_input_read_is_dense: min over blocks of eff_rank(E[JᵀJ]) ≥ 200 (of 384). Null: some block ≤ 96.
- pred_d_cross_output_lands_in_core: mean over blocks of the core energy fraction of E[JJᵀ] ≥ 0.70. Null: ≤ 0.50.
- pred_e_constant_gate_is_a_minority: median over blocks of the constant-gate share ≤ 0.50. Null: ≥ 0.80.

BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "split_tol": 1e-3, "b_min": 200.0, "c_min": 200.0, "d_min": 0.70, "e_max": 0.50}
NULLS = {"b_max": 64.0, "c_max": 96.0, "d_max": 0.50, "e_min": 0.80}

## Price
GPU: parent fit pass (96 docs ×2) + 1 chunk grab + baseline + 2 arms × 64 docs ≈ 400 doc-forwards; weight algebra ≈ 10 × (768 J_i of
1152×384 + 512-token Monte-Carlo) ≈ 30–60 s on the 5090. Nothing installs into the §312 frontier.

## What would change my mind
b null met → the gate's c-dependence is low-rank: register P3 (rank-r W_l(c) surrogate, CE-scored) next. c null met → the read selects a
low-dimensional tail sub-channel, contradicting §2779/§2781 at the weight level. e null met → a constant gate would suffice, contradicting §2782.
