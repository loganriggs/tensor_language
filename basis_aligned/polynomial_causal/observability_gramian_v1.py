# OBSERVABILITY QUOTIENT v1 -- the first brick of the entry point that replaced causal-response factorization.
#
# explanation_1405 §15 named the alternate entry point after v1's rejection: "choose early directions by which
# downstream readers and losses can distinguish them, merge states that have the same measured future
# consequences, and factor only the resulting quotient." Lane 1's §2086-§2088 saw the phenomenon this rests on:
# the assembly's stream error PEAKS at block 6 (rel-MSE 1.74) and is attenuated to 0.59 by block 17 --
# downstream computation ignores, compensates or repairs most of an early error. The quotient makes that
# quantitative: at a stream site k, the OBSERVABLE subspace is what the downstream loss is sensitive to.
#
# OBJECT. For site k in {2, 5, 9} (the residual stream entering block k), the first-order observability Gramian
#   G_k = E_positions[ g g^T ],   g = dCE_t / dx_k(t)   (per-position gradient of that position's own CE)
# on N fresh rows (bilin18_eval_tokens_large.pt, the zero-overlap window of §2036), positions >= 64 (house
# convention). Its eigen-spectrum gives the effective observable rank; the activation covariance of x_k gives
# the activation rank. The quotient claim is that the former is much smaller than the latter -- most of what
# the stream carries at block k is not distinguished by anything downstream -- and that this is CAUSAL: a
# perturbation of fixed norm inside the top-r observable subspace changes CE far more than the same norm in
# its complement. The first-order theory predicts the ratio; the test is whether it holds at the error
# magnitudes an actual compressed program produces (rel-norm 0.5 and 1.0, bracketing §2086's block-6 error).
#
# REGISTERED PREDICTIONS (r90 = number of eigenvalues to reach 90% of trace):
#   (a) THE QUOTIENT IS SMALL: at every site, r90(G_k) <= 0.5 * r90(Cov x_k), and the observable subspace
#       fitted on half A of the rows captures >= 0.80 of the gradient energy on half B (document-stable, the
#       §2098 standard). If FALSE, the loss is sensitive to most of the stream at first order and "factor only
#       the quotient" buys no reduction at these sites.
#   (b) AND IT IS CAUSAL AT REAL MAGNITUDES: at every site and both relative norms, the mean CE increase from
#       a random perturbation inside the top-r90 observable subspace is >= 3x the increase from a random
#       perturbation of the SAME norm in the orthogonal complement (8 draws each, 64 rows). The 3x is a design
#       bar: linear theory gives the eigenvalue ratio (typically >= 10x), and 3x is where the anisotropy is
#       large enough to matter for pricing a program's error budget. If FALSE at norm 1.0 but TRUE at 0.5, the
#       quotient is a small-error object and does not cover the assembly's actual error size -- reported as such.
#   (c) NULL: a RANDOM r90-dimensional subspace, same norm, costs less than the observable subspace at every
#       site (8 draws), and within a factor of 2 of the complement. Without this, (b) could be an artefact of
#       subspace dimension rather than direction.
#
# Descriptive: r50/r90/r99 of both spectra per site, the block-by-block attenuation of an observable vs a
# complement perturbation (norm of the induced stream deviation at every later block), and the top-8 observable
# directions' overlap with the lm_head row space (are they just "logit directions"?).
#
# Self-reviewed; no independent auditor on this instance. Writes observability_gramian_v1_results.json.
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BQ = os.path.join(ROOT, 'basis_aligned', 'bilinear_quotient')
for p in (ROOT, HERE, BQ):
    sys.path.insert(0, p)
os.chdir(HERE)

ROWS_PATH = os.path.join(BQ, 'bilin18_eval_tokens_large.pt')
SITES = (2, 5, 9)
NROWS, NPERT_ROWS, T, SKIP = 256, 64, 256, 64
REL_NORMS = (0.5, 1.0)
NDRAW = 8
OUT = os.path.join(HERE, 'observability_gramian_v1_results.json')
if os.environ.get('BQLIB_DRYRUN') == '1':
    if not os.path.exists(ROWS_PATH):
        print(f'DRYRUN FAIL: missing {ROWS_PATH}'); raise SystemExit(1)
    if os.path.exists(OUT):
        print(f'DRYRUN FAIL: {OUT} exists (create-only)'); raise SystemExit(1)
    print(f'DRYRUN OK: sites {SITES}, {NROWS} rows for the Gramian, {NPERT_ROWS} rows x {NDRAW} draws x '
          f'{REL_NORMS} for the causal test')
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import bilin18_observed_model_facade as FAC                               # noqa: E402

T0 = time.time()
M, RECEIPT = FAC.load_bilin18()
DEV = next(M.parameters()).device
D = M.config.n_embd
H = M.transformer.h
ROWS = torch.load(ROWS_PATH, map_location='cpu')[:, :T + 1].long()
print(f'model {RECEIPT.weights_sha256[:12]} | rows {tuple(ROWS.shape)} | {time.time() - T0:.0f}s', flush=True)


def run(idx, site=None, delta=None, want_grad=False, trace=False):
    """Forward; optionally add delta at the stream entering block `site`; return per-position CE (and x_site)."""
    x = F.rms_norm(M.transformer.wte(idx), (D,))
    x0, v1, xs, devs = x, None, None, []
    for li, blk in enumerate(H):
        if li == site:
            if delta is not None:
                x = x + delta
            if want_grad:
                x = x.detach().requires_grad_(True)
            xs = x
        if trace and site is not None and li > site:
            devs.append(x)
        x, v1 = blk(x, v1, x0)
    if trace and site is not None:
        devs.append(x)
    logits = M.lm_head(F.rms_norm(x, (D,)))
    logits = 30 * torch.tanh(logits / 30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                         idx[:, 1:].reshape(-1), reduction='none').view(idx.shape[0], -1)
    return ce, xs, devs


def r_at(eig, frac):
    c = torch.cumsum(eig, 0) / eig.sum()
    return int((c < frac).sum()) + 1


def spectrum(G):
    e = torch.linalg.eigvalsh(G).flip(0).clamp_min(0)
    return e, {f'r{int(f * 100)}': r_at(e, f) for f in (0.5, 0.9, 0.99)}


results = {'sites': {}, 'rows_gramian': NROWS, 'rows_perturbation': NPERT_ROWS, 'rel_norms': list(REL_NORMS)}
lm = M.lm_head.weight.detach().float()
lm_basis = torch.linalg.svd(lm, full_matrices=False)[2][:64].T                 # D x 64 top logit directions
for site in SITES:
    ts = time.time()
    G = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    C = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    GA = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    GB = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = 0
    for s in range(0, NROWS, 8):
        idx = ROWS[s:s + 8].to(DEV)
        ce, xs, _ = run(idx, site=site, want_grad=True)
        ce[:, SKIP:].sum().backward()
        g = xs.grad[:, SKIP:-1].reshape(-1, D).double()
        a = xs.detach()[:, SKIP:-1].reshape(-1, D).double()
        G += g.T @ g; C += (a - a.mean(0)).T @ (a - a.mean(0)); n += g.shape[0]
        (GA if s < NROWS // 2 else GB).add_(g.T @ g)
        M.zero_grad(set_to_none=True)
    G /= n; C /= n
    eg, rg = spectrum(G)
    ec, rc = spectrum(C)
    Q = torch.linalg.eigh(G)[1].flip(1)                                          # descending
    r90 = rg['r90']
    P_obs = Q[:, :r90].float()
    QA = torch.linalg.eigh(GA)[1].flip(1)[:, :r90]
    transfer = float((QA.T @ GB @ QA).trace() / GB.trace())
    lm_overlap = float(((lm_basis.T.to(DEV) @ Q[:, :8].float()) ** 2).sum() / 8)
    # ---- causal test: perturb inside observable subspace vs complement vs random r90-subspace
    idxp = ROWS[NROWS:NROWS + NPERT_ROWS].to(DEV)
    with torch.no_grad():
        base_ce, xs_base, _ = run(idxp, site=site)
        base = float(base_ce[:, SKIP:].mean())
        xnorm = xs_base[:, SKIP:].norm(dim=-1).mean()
    gen = torch.Generator(device='cpu').manual_seed(site)
    causal = {}
    for rel in REL_NORMS:
        rec = {'observable': [], 'complement': [], 'random_r90': []}
        atten = {'observable': None, 'complement': None}
        for draw in range(NDRAW):
            z = torch.randn(NPERT_ROWS, idxp.shape[1], D, generator=gen).to(DEV)
            d_obs = z @ P_obs @ P_obs.T
            d_cmp = z - d_obs
            Rr = torch.linalg.qr(torch.randn(D, r90, generator=gen))[0].to(DEV)
            d_rnd = z @ Rr @ Rr.T
            for name, dv in (('observable', d_obs), ('complement', d_cmp), ('random_r90', d_rnd)):
                dv = dv / dv.norm(dim=-1, keepdim=True).clamp_min(1e-9) * (rel * xnorm)
                with torch.no_grad():
                    ce, _, devs = run(idxp, site=site, delta=dv, trace=(draw == 0 and name != 'random_r90'))
                rec[name].append(float(ce[:, SKIP:].mean()) - base)
                if devs:
                    with torch.no_grad():
                        _, _, devs0 = run(idxp, site=site, trace=True)
                    atten[name] = [round(float((a - b)[:, SKIP:].norm(dim=-1).mean() / (rel * xnorm)), 4)
                                   for a, b in zip(devs, devs0)]
        causal[str(rel)] = {k: {'mean_dCE': round(sum(v) / len(v), 5), 'min': round(min(v), 5),
                                'max': round(max(v), 5)} for k, v in rec.items()}
        causal[str(rel)]['attenuation_by_block'] = atten
    results['sites'][str(site)] = {
        'gramian_spectrum': rg, 'activation_spectrum': rc,
        'gramian_top8_eigs': [float(v) for v in eg[:8]], 'gramian_trace': float(eg.sum()),
        'observable_transfer_A_to_B': round(transfer, 4), 'lm_head_overlap_top8': round(lm_overlap, 4),
        'base_ce_perturbation_rows': round(base, 4), 'mean_stream_norm': round(float(xnorm), 3),
        'causal': causal, 'seconds': round(time.time() - ts, 1)}
    print(f'site {site}: r90 obs {r90} vs act {rc["r90"]} (r50 {rg["r50"]}/{rc["r50"]}, r99 {rg["r99"]}/{rc["r99"]}) '
          f'| transfer {transfer:.3f} | lm overlap {lm_overlap:.3f} | ' + ' '.join(
              f'rel{rel}: obs {causal[str(rel)]["observable"]["mean_dCE"]:+.4f} cmp '
              f'{causal[str(rel)]["complement"]["mean_dCE"]:+.4f} rnd {causal[str(rel)]["random_r90"]["mean_dCE"]:+.4f}'
              for rel in REL_NORMS) + f' | {time.time() - ts:.0f}s', flush=True)
    del G, C, GA, GB, Q
    torch.cuda.empty_cache()

S = results['sites']
pa = all(S[k]['gramian_spectrum']['r90'] <= 0.5 * S[k]['activation_spectrum']['r90'] and
         S[k]['observable_transfer_A_to_B'] >= 0.80 for k in S)
ratio = {k: {rel: S[k]['causal'][rel]['observable']['mean_dCE'] / max(S[k]['causal'][rel]['complement']['mean_dCE'], 1e-9)
             for rel in S[k]['causal'] if rel != 'attenuation_by_block'} for k in S}
pb = all(v >= 3.0 for k in ratio for v in ratio[k].values())
pc = all(S[k]['causal'][rel]['random_r90']['mean_dCE'] < S[k]['causal'][rel]['observable']['mean_dCE'] and
         S[k]['causal'][rel]['random_r90']['mean_dCE'] <= 2 * max(S[k]['causal'][rel]['complement']['mean_dCE'], 1e-9)
         for k in S for rel in S[k]['causal'] if rel != 'attenuation_by_block')
results.update({'observable_over_complement_ratio': ratio, 'model_weights_sha256': RECEIPT.weights_sha256,
                'self_reviewed': True, 'pred_a_quotient_small_and_stable': bool(pa),
                'pred_b_causal_at_real_magnitudes': bool(pb), 'pred_c_random_subspace_null': bool(pc),
                'runtime_s': round(time.time() - T0, 1)})
if os.path.exists(OUT):
    print(f'{OUT} exists; refusing to overwrite'); raise SystemExit(2)
json.dump(results, open(OUT, 'w'), indent=1)
print(f'(a) quotient small (r90 obs <= 0.5 r90 act) and A->B transfer >= 0.80 at every site: {"HELD" if pa else "FAILED"}')
print(f'(b) observable/complement dCE ratio >= 3 at every site and norm: {"HELD" if pb else "FAILED"}  {ratio}')
print(f'(c) random-r90 subspace below observable and within 2x complement: {"HELD" if pc else "FAILED"}')
print(f'wrote {OUT} ({time.time() - T0:.0f}s)')
