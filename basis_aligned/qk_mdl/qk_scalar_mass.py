"""SCALAR-MASS COLLAPSE RETEST at bilin18 (aggregate list #1; BILIN18_LAYERS_0_1.md section 7,
Correction 2; registered predictions: qk_scalar_mass_predictions.json — written before this file).

Refit the layer-0 QK dictionary under the SCALAR-ONLY objective — the context-expected OV
objective of qk_ctx_train.py with every token's OV Gram replaced by its isotropic scalar mass
(trace/head_dim x identity), i.e. all geometry discarded — and compare, at IDENTICAL bits and
identical init/steps/encoder, against (a) the plain reconstruction-MSE dictionary and (b) the
full context-expected-metric dictionary. This is the bilin18 retest of the small-scale FINDING 13
section 4b result (tiny_full_interp/RESULTS.md), where the scalar-only control BEAT the full
metric at both budgets.

SCALAR-ONLY OBJECTIVE — derivation from the full context-expected loss (ctx() in qk_ctx_train.py
and qk_pareto_sweep.py; eq. dagger of ov_metric_explainer.md):
    L_full = sum_i q_i [ T*(s_i - |mu_i|^2)_+ + T^2*|mu_i|^2 ],
      mu_i = sum_j q_j dP_ij U_j,    s_i = sum_j q_j dP_ij^2 |U_j|^2,
      U_j  = W_o^h W_v^h e_hat_j  (the token's OV output vector),  dP = S1h*S2h - S1*S2.
The OV geometry enters ONLY through the Gram inner products U_j^T U_j'. Replace every token's
OV Gram U_j U_j^T by (tr(U_j U_j^T)/hd) * I_hd, independent across tokens (the FINDING 13 4b
ablation). Then U_j^T U_j' -> 0 for j != j' and |U_j|^2 -> w2_j = tr(Gram) (unchanged: the /hd
of the substitution cancels against tr(I_hd) = hd), which gives EXACTLY
    L_scalar = sum_i q_i [ T*(s_i - m_i)_+ + T^2*m_i ],   m_i = sum_j q_j^2 dP_ij^2 w2_j
             = sum_{i,j} q_i * q_j * w2_j * (T + (T^2 - T)*q_j) * dP_ij^2 .
KEPT:    unigram exposure on both sides (q_i query, q_j key); each key token's scalar mass
         w2_j = trace of its OV Gram; the T/T^2 ledger structure (which now only distinguishes
         a token paired with itself, since all cross-token coherence is gone); the pattern-level
         squared error dP_ij^2; the same normalization by the identical functional of the true P.
DROPPED: every cross-token OV inner product U_j^T U_j' — the OV directions and the coherent
         T^2 cancellation geometry. Nothing else changes; Gate B verifies the reduction is exact
         (running the UNMODIFIED ctx() code with an isotropized U reproduces L_scalar to machine
         precision, so the ablation removed exactly the geometry and nothing else).
Per-token scalar mass in the predictions-file sense: w_t = q_t * (tr(OV Gram_t)/hd) * tr(I_hd)
         = q_t * w2_t — unigram exposure times the trace/head_dim of the OV Gram term.

BUDGETS (both at the qk_pareto_sweep.py bit convention, dl_sparse_dict per head-branch):
  n=1024, k=8  -> 455.4 Mbit (the headline dictionary point; init qk_dict_l0_seed0.pt)
  n=256,  k=4  -> 182.8 Mbit (the existing "183 Mbit" context arm: qk_pareto_sweep.json job
                  n256_k4_s0, dce_ctx +0.0073, dce_omp +0.0149 — MSE fit refit here at seed 0
                  with the identical train_dict recipe, since no n=256 checkpoint was saved)
  (No third budget: it is not free — each extra budget costs an 18-head-branch MSE fit, two
  1500-step finetunes and three-four full 307k audits.)

AUDIT: the standard 307k-prediction FineWeb audit at T=512 (600 seqs x 512 preds), delta-CE vs
baseline 3.0763, per-position losses kept so every contrast carries a PAIRED per-position SE
(mean and SE of the per-position loss difference; red-team item 8) plus a seq-clustered SE.

GATES (recorded in the output JSON; the main result is untrusted unless both pass):
  Gate A — audit-path reproduction: qk_dict_l0_ctx.pt audited through THIS script's audit code
           must reproduce the published dce_fw = +0.0054 (qk_ctx_train.json) within 0.0005
           (within 0.01 on the ~2k-prediction smoke subset).
  Gate B — objective reduction: on a small random case, L_scalar == ctx() with U isotropized
           (two exact constructions + a Monte-Carlo check of the (trace/hd)*I semantics).

Writes qk_scalar_mass.json (+ scalar/ctx/mse dict checkpoints and per-position loss .npz).
Smoke mode (--smoke): CPU-only, tiny finetune (head 0, 5 steps, M=128), ~2k-prediction audit,
writes qk_scalar_mass_smoke.json. Proves plumbing + Gate B; touches nothing the full run trusts.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from mdl_accounting import dl_sparse_dict
from qk_sae_lib import train_dict, encode_token, encode_omp, fvu

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
NAMES = ('q1', 'k1', 'q2', 'k2')
BRANCHES = (('q1', 'k1'), ('q2', 'k2'))
CTX_STEPS, CTX_M, CTX_LR, T_CTX = 1500, 1024, 3e-4, 512.0   # qk_ctx_train / qk_pareto_sweep verbatim
PUBLISHED = {
    'baseline_ce_fw': 3.0763,
    'n1024_k8': {'ctx': 0.0054, 'plain_lin': 0.0076, 'plain_omp': 0.0059, 'Mbits': 455.4},
    'n256_k4': {'ctx': 0.0073, 'plain_lin': 0.0171, 'plain_omp': 0.0149, 'Mbits': 182.8},
}

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true', help='CPU plumbing test: tiny finetune, ~2k-pred audit')
A = ap.parse_args()
SMOKE = A.smoke
DEV = 'cpu' if SMOKE else 'cuda'
OUT = f'{QK}/qk_scalar_mass_smoke.json' if SMOKE else f'{QK}/qk_scalar_mass.json'
N_SEQ = 4 if SMOKE else None            # 4 seqs x 512 = 2048 predictions in smoke
GATE_A_TOL = 0.01 if SMOKE else 0.0005
BUDGETS = [(1024, 8)] if SMOKE else [(1024, 8), (256, 4)]
FT_STEPS = 5 if SMOKE else CTX_STEPS
FT_M = 128 if SMOKE else CTX_M
FT_HEADS = [0] if SMOKE else None       # smoke: finetune head 0 only, rest stay at MSE init

torch.manual_seed(0)
T0 = time.time()
res = {'experiment': 'scalar-mass collapse retest (aggregate list #1)', 'smoke': SMOKE,
       'published': PUBLISHED, 'gates': {}, 'budgets': {},
       'scalar_weight_formula':
           'w_t = q_t * w2_t (unigram exposure x trace of OV Gram); per-entry weight on dP_ij^2 '
           'is q_i * q_j * w2_j * (T + (T^2-T)*q_j), T=512 — the exact Gram->(tr/hd)*I collapse '
           'of the qk_ctx_train.py objective (see module docstring)'}


def save():
    json.dump(res, open(OUT, 'w'), indent=2)


# ---------------------------------------------------------------- objectives
def ctx_loss(mat, qs, Us, w2):
    """Full context-expected OV loss — VERBATIM ctx() of qk_ctx_train.py / qk_pareto_sweep.py."""
    mu = (mat * qs[None, :]) @ Us
    mu2 = mu.pow(2).sum(1)
    s_ = (mat.pow(2) * (qs * w2)[None, :]).sum(1)
    return (qs * (T_CTX * (s_ - mu2).clamp_min(0) + T_CTX * T_CTX * mu2)).sum()


def scalar_loss(mat, qs, w2):
    """Scalar-only collapse of ctx_loss (derivation in module docstring). Same code shape:
    only mu2 changes, from |sum_j q_j dP_ij U_j|^2 to its geometry-free diagonal."""
    mu2 = (mat.pow(2) * (qs.pow(2) * w2)[None, :]).sum(1)
    s_ = (mat.pow(2) * (qs * w2)[None, :]).sum(1)
    return (qs * (T_CTX * (s_ - mu2).clamp_min(0) + T_CTX * T_CTX * mu2)).sum()


# ---------------------------------------------------------------- Gate B (pure CPU, no model)
def gate_b():
    """Verify: scalar_loss == ctx_loss with every token's OV Gram replaced by (tr/hd)*I.
    Two exact orthogonalized constructions realize the substitution inside the UNMODIFIED
    ctx_loss code; a Monte-Carlo draw of isotropic U_j ~ N(0,(w2_j/hd) I_hd) checks the
    (trace/hd)*I second-moment semantics directly (exact in expectation)."""
    g = torch.Generator().manual_seed(1)
    M, D, hd = 32, 48, 8
    mat = torch.randn(M, M, generator=g)
    qs = torch.softmax(torch.randn(M, generator=g), 0)
    Us = torch.randn(M, D, generator=g)
    w2 = Us.pow(2).sum(1)
    sc = scalar_loss(mat, qs, w2).item()
    full = ctx_loss(mat, qs, Us, w2).item()
    # exact 1: one private orthogonal axis per token, norm sqrt(w2_j)
    Ud = torch.diag(w2.sqrt())
    d1 = ctx_loss(mat, qs, Ud, Ud.pow(2).sum(1)).item()
    # exact 2: per-token private hd-dim block with entries sqrt(w2_j/hd) — trace/hd spread over hd dims
    Ub = torch.zeros(M, M * hd)
    for j in range(M):
        Ub[j, j * hd:(j + 1) * hd] = (w2[j] / hd).sqrt()
    d2 = ctx_loss(mat, qs, Ub, Ub.pow(2).sum(1)).item()
    mc = float(np.mean([ctx_loss(mat, qs, torch.randn(M, hd, generator=g) * (w2 / hd).sqrt()[:, None],
                                 w2).item() for _ in range(500)]))
    r1, r2 = abs(d1 - sc) / sc, abs(d2 - sc) / sc
    rmc = abs(mc - sc) / sc
    rgeo = abs(full - sc) / sc
    ok = r1 < 1e-5 and r2 < 1e-5 and rmc < 0.05 and rgeo > 1e-4
    res['gates']['B'] = {
        'scalar_loss': sc, 'ctx_iso_diag': d1, 'ctx_iso_block': d2, 'ctx_real_geometry': full,
        'mc_iso_mean_500': mc, 'rel_err_diag': r1, 'rel_err_block': r2, 'rel_err_mc': rmc,
        'rel_diff_vs_real_geometry': rgeo,
        'pass': bool(ok),
        'note': 'exact constructions must match to 1e-5; MC to 5% (clamp bias + sampling); '
                'real-geometry loss must DIFFER (else nothing was ablated)'}
    print(f'Gate B: scalar {sc:.4f} | iso-diag rel {r1:.2e} | iso-block rel {r2:.2e} | '
          f'MC rel {rmc:.3f} | vs real geometry {rgeo:.3f} -> {"PASS" if ok else "FAIL"}', flush=True)
    return ok


gate_b()
save()

# ---------------------------------------------------------------- model + factor tables
print(f'loading bilin18 on {DEV}', flush=True)
m, cfg = load_elriggs('bilin18', device=DEV)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
NHB, ROW = NH * 2, 2 * HD
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
if N_SEQ is not None:
    FINEWEB = FINEWEB[:N_SEQ]

TAB = {}
for br, (qn, kn) in enumerate(BRANCHES, start=1):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
HB = [(h, qn, kn) for h in range(NH) for (qn, kn) in BRANCHES]

with torch.no_grad():
    a = m.transformer.h[0].attn
    E = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
    Vv = a.c_v(E).view(V, NH, HD)
    Wo = a.c_proj.weight.detach().float().view(D, NH, HD)
QFULL = (torch.bincount(
    torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64)).flatten(),
    minlength=V).float() + 0.5).to(DEV)   # full-corpus unigram, as in qk_ctx_train.py


def rows(h, qn, kn):
    return torch.cat([TAB[qn][:, h], TAB[kn][:, h]], 1)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


def tables_from(recs):
    out = {n: TAB[n].clone() for n in NAMES}
    for (h, qn, kn), rec in zip(HB, recs):
        out[qn][:, h] = rec[:, :HD]
        out[kn][:, h] = rec[:, HD:]
    return {n: unit_rms(out[n]) for n in NAMES}


# ---------------------------------------------------------------- audit with per-position losses
@torch.no_grad()
def audit_losses(tabs, batch=4):
    """Standard FineWeb audit (T=512), but returns the (n_seq, 512) per-position loss matrix so
    every contrast can carry a PAIRED per-position SE. Mean equals the standard audit CE."""
    out = []
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16', score_patch=None if tabs is None else patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
                             reduction='none').view(b.shape[0], -1)
        out.append(ce.cpu())
    return torch.cat(out)


def paired(la, lb):
    """Per-position paired contrast: difference of losses at each audit position, then mean and
    SE of that difference (red-team item 8), plus a seq-clustered SE."""
    d = (la - lb).double()
    n = d.numel()
    seq_means = d.mean(1)
    return {'mean': round(d.mean().item(), 6),
            'se_pos': round((d.std(unbiased=True) / n ** 0.5).item(), 6),
            'se_seq': round((seq_means.std(unbiased=True) / len(seq_means) ** 0.5).item(), 6),
            'n_pos': int(n), 'n_seq': int(len(seq_means))}


# ---------------------------------------------------------------- finetunes (qk_pareto_sweep recipe)
def finetune(fits, n, k, seed, kind):
    """Per head, both branch dicts jointly — VERBATIM ctx_finetune of qk_pareto_sweep.py except
    that kind='scalar' swaps ctx_loss for scalar_loss (and needs only w2, never the U directions).
    Same init (the MSE fits), same steps/lr/M/sampling stream => same bits, matched comparison."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    out_fits = list(fits)
    losses = []
    heads = range(NH) if FT_HEADS is None else FT_HEADS
    for h in heads:
        parts, params = {}, []
        for br in (0, 1):
            Dn0, b0, We0 = fits[h * 2 + br]
            Dm = Dn0.clone().requires_grad_(True)
            We = We0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            parts[br] = (Dm, We, b)
            params += [Dm, We, b]
        opt = torch.optim.Adam(params, lr=CTX_LR)
        Uh = Vv[:, h] @ Wo[:, h].T
        first = last = None
        for step in range(FT_STEPS):
            sample = torch.randperm(V, generator=g)[:FT_M].to(DEV)
            qs = QFULL[sample]
            qs = qs / qs.sum()
            Us = Uh[sample]
            w2 = Us.pow(2).sum(1)
            Shat, S = [], []
            for br, (qn, kn) in enumerate(BRANCHES):
                X = torch.cat([TAB[qn][sample, h], TAB[kn][sample, h]], 1)
                Dm, We, b = parts[br]
                Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                z = (X - b) @ We.T
                vals, idx = z.abs().topk(k, dim=1)
                coeff = torch.gather(z, 1, idx)
                rec = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
                Shat.append(unit_rms(rec[:, :HD]) @ unit_rms(rec[:, HD:]).T / HD)
                S.append(X[:, :HD] @ X[:, HD:].T / HD)
            P = S[0] * S[1]
            dP = Shat[0] * Shat[1] - P
            fn = (lambda mat: ctx_loss(mat, qs, Us, w2)) if kind == 'ctx' \
                else (lambda mat: scalar_loss(mat, qs, w2))
            with torch.no_grad():
                den = fn(P).clamp_min(1e-12)
            loss = fn(dP) / den
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        losses.append((round(first, 4), round(last, 4)))
        print(f'  {kind} head {h}: {first:.4f} -> {last:.4f}', flush=True)
        for br in (0, 1):
            Dm, We, b = parts[br]
            out_fits[h * 2 + br] = ((Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(),
                                    b.detach(), We.detach())
    return out_fits, losses


def fits_to_blob(fits):
    blob = {}
    for i, (Dn, b, We) in enumerate(fits):
        blob[f'Dn{i}'] = Dn.cpu(); blob[f'b{i}'] = b.cpu(); blob[f'We{i}'] = We.cpu()
    return blob


def blob_to_fits(blob):
    return [(blob[f'Dn{i}'].to(DEV), blob[f'b{i}'].to(DEV), blob[f'We{i}'].to(DEV))
            for i in range(NHB)]


# ---------------------------------------------------------------- baseline + Gate A
LOSS_STORE = {}
print('auditing baseline', flush=True)
base = audit_losses(None)
LOSS_STORE['baseline'] = base
CE0 = base.double().mean().item()
res['baseline_ce_fw'] = round(CE0, 4)
res['n_predictions'] = int(base.numel())
print(f'baseline CE {CE0:.4f} on {base.numel()} predictions', flush=True)
save()

print('Gate A: auditing the published full-metric dictionary qk_dict_l0_ctx.pt', flush=True)
ctx_blob = torch.load(f'{QK}/qk_dict_l0_ctx.pt', map_location=DEV)
ctx_fits_1024 = blob_to_fits(ctx_blob)
recs = [encode_token(rows(*hb), f[0], f[1], f[2], 8) for f, hb in zip(ctx_fits_1024, HB)]
ctx1024_losses = audit_losses(tables_from(recs))
LOSS_STORE['n1024_k8_ctx'] = ctx1024_losses
dce_a = ctx1024_losses.double().mean().item() - CE0
gate_a_ok = abs(dce_a - PUBLISHED['n1024_k8']['ctx']) <= GATE_A_TOL
res['gates']['A'] = {'dce_reproduced': round(dce_a, 5), 'published': PUBLISHED['n1024_k8']['ctx'],
                     'tol': GATE_A_TOL, 'pass': bool(gate_a_ok),
                     'note': ('smoke: ~2k-pred subset, loose plumbing-level tolerance' if SMOKE else
                              'full 307k audit vs qk_ctx_train.json dce_fw')}
print(f'Gate A: dce {dce_a:+.5f} vs published +{PUBLISHED["n1024_k8"]["ctx"]:.4f} '
      f'(tol {GATE_A_TOL}) -> {"PASS" if gate_a_ok else "FAIL"}', flush=True)
save()

# ---------------------------------------------------------------- budgets
for (n, k) in BUDGETS:
    key = f'n{n}_k{k}'
    bits = NHB * dl_sparse_dict(n, ROW, V * k)
    row = {'n': n, 'k': k, 'Mbits': round(bits / 1e6, 1), 'arms': {}, 'contrasts': {}}
    res['budgets'][key] = row
    print(f'=== {key} ({row["Mbits"]} Mbit)', flush=True)

    # MSE fits (the shared init for both finetunes; also the plain arm)
    if (n, k) == (1024, 8):
        fits = blob_to_fits(torch.load(f'{QK}/qk_dict_l0_seed0.pt', map_location=DEV))
        print('  loaded cached seed-0 n=1024 k=8 MSE fits', flush=True)
    else:
        mse_path = f'{QK}/qk_dict_l0_mse_{key}.pt'
        if os.path.exists(mse_path):
            fits = blob_to_fits(torch.load(mse_path, map_location=DEV))
            print(f'  loaded cached {mse_path}', flush=True)
        else:
            fits = [train_dict(rows(*hb), n, k, seed=0) for hb in HB]
            torch.save(fits_to_blob(fits), mse_path)
            print(f'  fitted {NHB} head-branches (MSE), saved {mse_path}', flush=True)

    # arm: plain MSE, linear encoder (same encoder family as the finetuned arms)
    recs = [encode_token(rows(*hb), f[0], f[1], f[2], k) for f, hb in zip(fits, HB)]
    row['arms']['plain_lin'] = {'fvu': round(sum(fvu(r, rows(*hb)) for r, hb in zip(recs, HB)) / NHB, 5)}
    ll = audit_losses(tables_from(recs))
    LOSS_STORE[f'{key}_plain_lin'] = ll
    row['arms']['plain_lin']['dce'] = round(ll.double().mean().item() - CE0, 5)
    print(f'  plain lin: dce {row["arms"]["plain_lin"]["dce"]:+.5f}', flush=True)
    save()

    # arm: plain MSE, OMP encoder (the published strong plain arm) — full mode only
    if not SMOKE:
        recs = [encode_omp(rows(*hb), f[0], f[1], k) for f, hb in zip(fits, HB)]
        ll = audit_losses(tables_from(recs))
        LOSS_STORE[f'{key}_plain_omp'] = ll
        row['arms']['plain_omp'] = {'dce': round(ll.double().mean().item() - CE0, 5)}
        print(f'  plain omp: dce {row["arms"]["plain_omp"]["dce"]:+.5f}', flush=True)
        save()

    # arm: full context-expected metric
    if (n, k) == (1024, 8):
        row['arms']['ctx'] = {'dce': round(dce_a, 5), 'source': 'qk_dict_l0_ctx.pt (Gate A audit)'}
    else:
        ctx_path = f'{QK}/qk_dict_l0_ctx_{key}.pt'
        if os.path.exists(ctx_path):
            cf = blob_to_fits(torch.load(ctx_path, map_location=DEV))
            closs = None
        else:
            cf, closs = finetune(fits, n, k, 0, 'ctx')
            torch.save(fits_to_blob(cf), ctx_path)
        recs = [encode_token(rows(*hb), f[0], f[1], f[2], k) for f, hb in zip(cf, HB)]
        ll = audit_losses(tables_from(recs))
        LOSS_STORE[f'{key}_ctx'] = ll
        row['arms']['ctx'] = {'dce': round(ll.double().mean().item() - CE0, 5),
                              'train_losses': closs, 'source': f'refit here ({ctx_path})'}
        print(f'  ctx refit: dce {row["arms"]["ctx"]["dce"]:+.5f}', flush=True)
    save()

    # arm: SCALAR-ONLY (the experiment)
    sc_path = f'{QK}/qk_dict_l0_scalar_{key}.pt'
    if not SMOKE and os.path.exists(sc_path):
        sf = blob_to_fits(torch.load(sc_path, map_location=DEV))
        sloss = None
    else:
        sf, sloss = finetune(fits, n, k, 0, 'scalar')
        if not SMOKE:
            torch.save(fits_to_blob(sf), sc_path)
    recs = [encode_token(rows(*hb), f[0], f[1], f[2], k) for f, hb in zip(sf, HB)]
    ll = audit_losses(tables_from(recs))
    LOSS_STORE[f'{key}_scalar'] = ll
    row['arms']['scalar'] = {'dce': round(ll.double().mean().item() - CE0, 5), 'train_losses': sloss}
    print(f'  scalar   : dce {row["arms"]["scalar"]["dce"]:+.5f}', flush=True)
    save()

    # paired contrasts (per-position differences; red-team item 8)
    sc = LOSS_STORE[f'{key}_scalar']
    cx = LOSS_STORE['n1024_k8_ctx'] if (n, k) == (1024, 8) else LOSS_STORE[f'{key}_ctx']
    row['contrasts']['scalar_minus_ctx'] = paired(sc, cx)
    row['contrasts']['scalar_minus_plain_lin'] = paired(sc, LOSS_STORE[f'{key}_plain_lin'])
    row['contrasts']['ctx_minus_plain_lin'] = paired(cx, LOSS_STORE[f'{key}_plain_lin'])
    if f'{key}_plain_omp' in LOSS_STORE:
        row['contrasts']['scalar_minus_plain_omp'] = paired(sc, LOSS_STORE[f'{key}_plain_omp'])
    for cname, c in row['contrasts'].items():
        print(f'  {cname}: {c["mean"]:+.6f} +- {c["se_pos"]:.6f} (pos SE) '
              f'+- {c["se_seq"]:.6f} (seq SE)', flush=True)
    save()
    if DEV == 'cuda':
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- persist per-position losses
if not SMOKE:
    np.savez_compressed(f'{QK}/qk_scalar_mass_losses.npz',
                        **{k_: v.numpy() for k_, v in LOSS_STORE.items()})
res['runtime_s'] = round(time.time() - T0, 1)
save()
print(f'done in {res["runtime_s"]}s -> {OUT}', flush=True)
