# RUNG 10, STEP 3: is m16's per-document coefficient PRIVATE, or inferable from the other owners' responses?
#
# §2098: the six m16 source circuits' deletion-response block has a document-stable two-direction basis
# (A->B transfer 0.8779). §2099: its per-document loading tracks neither sentence-boundary density (rho 0.035)
# nor base CE (0.14). The other lane's programs share ONE document code across owners (global CP) or give
# each owner a private code that must be inferred from anchor arms, and both failed at m16->* on held-out
# documents (calibrated NRMSE 2.4-3.2 at m=8-16 arms). Those failures were inside a CP grammar. This asks the
# grammar-free upper bound: fit a ridge from the OTHER five owners' per-document loadings (each owner's block
# projected on its own top-2 source directions, fitted on half A) to m16's two loadings, on half A, and score
# R^2 on half B. If that fails, m16's document code is PRIVATE to m16 -- no cheap owner's response carries it --
# and the m16 interface is a calibration cost in any program; if it holds, the code is shared and the CP
# failure was the grammar, not the information.
#
# METHOD. Same replayed training input (229 FIT documents), same SHA-salted split as §2098 (halves 114/115).
# For each owner o: U_o = block unfolded to (n_o sources) x (2*49*docs); top-2 left singular vectors P_o from
# half A; loading L_o(d) in R^2 = P_o^T U_o[:, phase, target, d] summed over (phase, target) with the valid
# mask -> per-document 2-vector. Predictor X(d) = concat of the five non-m16 owners' loadings (10 dims) plus
# their block RMS (5 dims) = 15 features; target Y(d) = m16's 2 loadings and its block RMS. Ridge (lam = 1e-2 n),
# standardised on A, scored by held-out R^2 on B per target, reported for both the loadings and the RMS.
#
# REGISTERED PREDICTIONS:
#   (a) THE CODE IS NOT PRIVATE: held-out R^2 for m16's block RMS from the other owners' 15 features >= 0.30.
#       The bar is a design bar stated in advance: every CP-constrained calibrated arm on the other lane scored
#       m16 at NRMSE > 2 (R^2 < 0), so ANY positive held-out R^2 is new information and 0.30 is the level at
#       which a program could price the code as "inferred from cheap owners" rather than measured. If FALSE,
#       m16's per-document coefficient is private: not in the text (§2099) and not in the other owners'
#       responses; the m16 interface must be measured per document or left uncompressed.
#   (b) AND IT IS NOT CAPACITY: with the document assignment of the target PERMUTED within half A (200 draws),
#       the 95th percentile of held-out R^2 is <= 0.05 and the observed R^2 exceeds it.
#   (c) THE TWO LOADINGS ARE NOT A SINGLE GAIN: the two m16 loadings' held-out R^2 differ by >= 0.15 OR their
#       correlation across documents is <= 0.7 -- i.e. m16's document code is genuinely two-dimensional. If
#       FALSE, the two directions share one per-document gain and m16's code is effectively one scalar, which
#       makes (a)'s question a one-number question and is reported as such.
#
# Descriptive: the reverse direction (other owners' RMS from m16's features) and per-owner univariate rho.
#
# Writes m16_code_from_other_owners_results.json.
import hashlib
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(os.path.dirname(BQ), 'polynomial_causal')
sys.path.insert(0, BQ)
sys.path.insert(0, PC)
sys.path.insert(0, os.path.dirname(os.path.dirname(BQ)))
os.chdir(BQ)

INPUT = os.path.join(PC, 'causal_response_factorization_v1_training_input.pt')
RECEIPT = os.path.join(PC, 'causal_response_factorization_v1_training_terminal', 'receipt.json')
PRIOR = os.path.join(BQ, 'm16_response_block_split_results.json')
BAR_A, BAR_B, NPERM, SALT = 0.30, 0.05, 200, 'm16-block-split'
if os.environ.get('BQLIB_DRYRUN') == '1':
    miss = [f for f in (INPUT, RECEIPT, PRIOR) if not os.path.exists(f)]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    p = json.load(open(PRIOR))
    if not p.get('pred_a_rank2_subspace_transfers'):
        print('DRYRUN FAIL: S2098 basis not document-stable'); raise SystemExit(1)
    print(f'DRYRUN OK: S2098 transfer {p["per_owner"]["m16"]["transfer_top2_A_to_B"]}; bars (a) {BAR_A} (b) {BAR_B}')
    raise SystemExit(0)

from pathlib import Path                                                  # noqa: E402

import torch                                                              # noqa: E402

from causal_response_factorization_v1_training_input import replay_training_input   # noqa: E402

T0 = time.time()
TERM = json.load(open(RECEIPT))
V, DIGEST = replay_training_input(
    Path(INPUT), expected_analysis_authority_sha256=TERM['authority_logical_sha256'],
    expected_artifact_sha256=TERM['payload']['input_sha256'], require_production=True)
R = V.response.clone()
R[~V.valid] = 0.0
OWNERS = list(V.owner_components)
GROUPS = V.source_groups
DOCS = V.document_ids.tolist()
ND = len(DOCS)
keyed = sorted((hashlib.sha256(f'{SALT}|{d}'.encode()).hexdigest(), i) for i, d in enumerate(DOCS))
A_IDX = torch.tensor(sorted(i for _, i in keyed[:len(keyed) // 2]))
B_IDX = torch.tensor(sorted(i for _, i in keyed[len(keyed) // 2:]))
print(f'replayed {DIGEST[:12]} | {ND} documents | halves {len(A_IDX)}/{len(B_IDX)}', flush=True)


def loadings(owner):
    """Per-document 2-vector on the owner's top-2 source directions (fitted on half A) plus block RMS."""
    rows = (GROUPS == OWNERS.index(owner)).nonzero().squeeze(1)
    UA = R[:, rows][..., A_IDX].permute(1, 0, 2, 3).reshape(len(rows), -1)
    P = torch.linalg.svd(UA, full_matrices=False)[0][:, :2]                       # sources x 2
    blk = R[:, rows]                                                              # 2 x n x 49 x docs
    proj = torch.einsum('sk,psTd->kd', P, blk)                                    # 2 x docs (summed cells)
    rms = (blk ** 2).mean(dim=(0, 1, 2)).sqrt()                                   # docs
    return proj.T.contiguous(), rms


LOAD = {o: loadings(o) for o in OWNERS}
X = torch.cat([torch.cat([LOAD[o][0], LOAD[o][1][:, None]], 1) for o in OWNERS if o != 'm16'], 1)   # docs x 15
Ym = torch.cat([LOAD['m16'][0], LOAD['m16'][1][:, None]], 1)                                          # docs x 3
TARGETS = ('m16_loading_1', 'm16_loading_2', 'm16_block_rms')


def ridge_r2(X, y, fit, ev):
    Xf, Xe = X[fit], X[ev]
    mu, sd = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True).clamp_min(1e-9)
    Xf, Xe = (Xf - mu) / sd, (Xe - mu) / sd
    yf, ye = y[fit], y[ev]
    lam = 1e-2 * len(Xf)
    w = torch.linalg.solve(Xf.T @ Xf + lam * torch.eye(X.shape[1], dtype=X.dtype), Xf.T @ (yf - yf.mean()))
    pred = Xe @ w + yf.mean()
    return float(1 - ((pred - ye) ** 2).sum() / ((ye - ye.mean()) ** 2).sum().clamp_min(1e-12))


r2 = {t: ridge_r2(X, Ym[:, j], A_IDX, B_IDX) for j, t in enumerate(TARGETS)}
g = torch.Generator().manual_seed(0)
draws = []
for _ in range(NPERM):
    perm = A_IDX[torch.randperm(len(A_IDX), generator=g)]
    y = Ym[:, 2].clone(); y[A_IDX] = Ym[perm, 2]
    draws.append(ridge_r2(X, y, A_IDX, B_IDX))
null = sorted(draws)
null_95 = null[int(0.95 * NPERM) - 1]
corr12 = float(torch.corrcoef(Ym[:, :2].T)[0, 1])
reverse = {o: ridge_r2(Ym, LOAD[o][1], A_IDX, B_IDX) for o in OWNERS if o != 'm16'}


def spearman(x, y):
    rx = x.argsort().argsort().double(); ry = y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


uni = {o: round(spearman(LOAD[o][1], LOAD['m16'][1]), 4) for o in OWNERS if o != 'm16'}
pa = r2['m16_block_rms'] >= BAR_A
pb = null_95 <= BAR_B and r2['m16_block_rms'] > null_95
pc = abs(r2['m16_loading_1'] - r2['m16_loading_2']) >= 0.15 or abs(corr12) <= 0.7
out = {'training_input_sha256': DIGEST, 'documents': ND, 'features': 15, 'targets': list(TARGETS),
       'heldout_r2': {k: round(v, 4) for k, v in r2.items()},
       'null_r2_block_rms': {'draws': NPERM, 'median': round(null[NPERM // 2], 4), 'p95': round(null_95, 4),
                             'max': round(null[-1], 4)},
       'm16_loading_correlation': round(corr12, 4),
       'reverse_r2_owner_rms_from_m16': {k: round(v, 4) for k, v in reverse.items()},
       'univariate_spearman_owner_rms_vs_m16_rms': uni,
       'bars': {'a': BAR_A, 'b_null_p95': BAR_B},
       'pred_a_code_not_private': bool(pa), 'pred_b_not_capacity': bool(pb),
       'pred_c_two_dimensional_code': bool(pc), 'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open('m16_code_from_other_owners_results.json', 'w'), indent=1)
print('held-out R^2: ' + ' | '.join(f'{k} {v:+.4f}' for k, v in r2.items()))
print(f'null R^2 (permuted target) median {null[NPERM // 2]:+.4f} p95 {null_95:+.4f} | loading corr {corr12:+.3f}')
print('reverse (owner RMS from m16): ' + ' '.join(f'{k} {v:+.3f}' for k, v in reverse.items()))
print('univariate rho(owner RMS, m16 RMS): ' + ' '.join(f'{k} {v:+.3f}' for k, v in uni.items()))
print(f'(a) R^2 {r2["m16_block_rms"]:+.4f} >= {BAR_A}: {"HELD" if pa else "FAILED"}')
print(f'(b) null p95 {null_95:+.4f} <= {BAR_B} and beaten: {"HELD" if pb else "FAILED"}')
print(f'(c) two-dimensional code: {"HELD" if pc else "FAILED"}')
print(f'wrote m16_code_from_other_owners_results.json ({time.time() - T0:.0f}s)')
