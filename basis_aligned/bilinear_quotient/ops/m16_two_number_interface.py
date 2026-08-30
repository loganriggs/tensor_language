# THE m16 TWO-NUMBER INTERFACE, PRICED (rung 34; the reviews' candidate C).
#
# §2098-§2100: the m16 deletion-response block has a document-stable two-direction source basis; its per-document
# coefficient is not in the text and only ~10% inferable from other owners. §2100's consequence: measure the two
# numbers per document from m16's OWN arms and price it. This builds that interface and tests it end to end on the
# training input's prospective document halves.
#
# MODEL (fit on half A only): X_d[s,p,t] ~ sum_k u_k[s] * g_k[p,t] * a_k(d), k = 1,2 — u from the §2098 SVD on
# half A, g_k the rank-1 (phase,target) profile of component k on half A, a_k(d) two per-document numbers. On half
# B, a_k(d) is estimated (i) from ALL of the document's valid m16 cells and (ii) from TWO physical arms only (one
# (phase,source) row of 49 targets per phase, at the two heaviest-|u| sources) — 2 intervention forwards per
# document. Score: held-out per-document R^2 of the reconstructed m16 block on half B.
#
# LITERAL PRICE: persistent 6*2 + 2*2*49 = 208 values; per-document 2 values; calibration 2 arms of 98.
#
# REGISTERED PREDICTIONS:
#   (a) A TWO-NUMBER CODE CARRIES THE BLOCK: median per-document R^2 on half B, coefficients from all cells, >= 0.5.
#       If FALSE the block is not a two-number-per-document object and the interface dies here.
#   (b) TWO ARMS SUFFICE: median R^2 with coefficients from the 2 arms >= 0.8 x the all-cells median.
#   (c) THE CODE IS NEEDED: the mean-code baseline (a_k = half-A mean) and a permuted-code null (a_k taken from a
#       random other document) both have median R^2 <= 0.1.
#
# Writes m16_two_number_interface_results.json. Self-reviewed.
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
SALT = 'm16-block-split'
if os.environ.get('BQLIB_DRYRUN') == '1':
    miss = [f for f in (INPUT, RECEIPT, PRIOR) if not os.path.exists(f)]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    p = json.load(open(PRIOR))
    print(f"DRYRUN OK: S2098 transfer {p['per_owner']['m16']['transfer_top2_A_to_B']}; "
          f"two-number interface, price 208 persistent + 2/doc + 2 arms")
    raise SystemExit(0)

from pathlib import Path                                                  # noqa: E402

import torch                                                              # noqa: E402

from causal_response_factorization_v1_training_input import replay_training_input   # noqa: E402

T0 = time.time()
TERM = json.load(open(RECEIPT))
V, DIGEST = replay_training_input(
    Path(INPUT), expected_analysis_authority_sha256=TERM['authority_logical_sha256'],
    expected_artifact_sha256=TERM['payload']['input_sha256'], require_production=True)
R = V.response.clone(); R[~V.valid] = 0.0
VAL = V.valid
OWNERS = list(V.owner_components); GROUPS = V.source_groups; DOCS = V.document_ids.tolist()
keyed = sorted((hashlib.sha256(f'{SALT}|{d}'.encode()).hexdigest(), i) for i, d in enumerate(DOCS))
A_IDX = torch.tensor(sorted(i for _, i in keyed[:len(keyed) // 2]))
B_IDX = torch.tensor(sorted(i for _, i in keyed[len(keyed) // 2:]))
rows = (GROUPS == OWNERS.index('m16')).nonzero().squeeze(1)
NS = len(rows)
X = R[:, rows].permute(1, 0, 2, 3).contiguous()                          # (6, 2, 49, docs)
M = VAL[:, rows].permute(1, 0, 2, 3).contiguous()
print(f'replayed {DIGEST[:12]} | m16 sources {NS} | halves {len(A_IDX)}/{len(B_IDX)}', flush=True)

# fit on half A: u (6x2), then rank-1 (p,t)-profile per component
XA = X[..., A_IDX].reshape(NS, -1)
U6 = torch.linalg.svd(XA, full_matrices=False)[0][:, :2]                 # (6, 2)
profiles = []
for k in range(2):
    W = torch.einsum('s,sptd->ptd', U6[:, k], X[..., A_IDX]).reshape(-1, len(A_IDX))   # (98, docsA)
    uu, ss, vv = torch.linalg.svd(W, full_matrices=False)
    profiles.append(uu[:, 0] * ss[0] / max(len(A_IDX) ** 0.5, 1.0))      # (98,), scale convention absorbed by a_k
GPT2 = torch.stack(profiles, 1).reshape(2, 49, 2)                        # (phase, target, k)
BASIS = torch.einsum('sk,ptk->sptk', U6, GPT2).reshape(-1, 2)            # (6*2*49, 2) design for a_k
top_src = U6.abs().sum(1).argsort(descending=True)[:2].tolist()
CAL = [(0, top_src[0]), (1, top_src[1])]                                 # (phase, source) arms


def coeffs(xd, mask):
    mm = mask.reshape(-1)
    Bm = BASIS[mm]
    if int(mm.sum()) < 4 or torch.linalg.matrix_rank(Bm) < 2:
        return None
    return torch.linalg.lstsq(Bm, xd.reshape(-1)[mm]).solution


def r2(xd, msk, a):
    pred = (BASIS @ a).reshape(xd.shape)
    num = float(((pred - xd)[msk] ** 2).sum()); den = float((xd[msk] ** 2).sum())
    return 1 - num / max(den, 1e-12)


gen = torch.Generator().manual_seed(34)
B_list = B_IDX.tolist()
mean_a_rows = []
for d in A_IDX.tolist():
    a = coeffs(X[..., d], M[..., d])
    if a is not None:
        mean_a_rows.append(a)
A_MEAN = torch.stack(mean_a_rows).mean(0)
full_r2, arm_r2, mean_r2, perm_r2 = [], [], [], []
for d in B_list:
    xd = X[..., d]; md = M[..., d]
    a_full = coeffs(xd, md)
    if a_full is None:
        continue
    calm = torch.zeros_like(md)
    for (p, src) in CAL:
        calm[src, p, :] = md[src, p, :]
    a_cal = coeffs(xd, calm)
    other = B_list[int(torch.randint(0, len(B_list), (1,), generator=gen))]
    a_perm = coeffs(X[..., other], M[..., other])
    full_r2.append(r2(xd, md, a_full))
    if a_cal is not None:
        arm_r2.append(r2(xd, md, a_cal))
    mean_r2.append(r2(xd, md, A_MEAN))
    if a_perm is not None:
        perm_r2.append(r2(xd, md, a_perm))


def med(v):
    w = sorted(x for x in v if x == x)
    return w[len(w) // 2] if w else float('nan')


mf, ma, mm_, mp = med(full_r2), med(arm_r2), med(mean_r2), med(perm_r2)
pa = mf >= 0.5
pb = ma >= 0.8 * mf
pc = mm_ <= 0.1 and mp <= 0.1
out = {'training_input_sha256': DIGEST, 'documents_scored': len(full_r2), 'calibration_arms': CAL,
       'price': {'persistent_values': 6 * 2 + 2 * 2 * 49, 'per_document_values': 2, 'calibration_arms': 2},
       'median_r2': {'all_cells': round(mf, 4), 'two_arms': round(ma, 4), 'mean_code': round(mm_, 4),
                     'permuted_code': round(mp, 4)},
       'pred_a_two_numbers_carry_the_block': bool(pa), 'pred_b_two_arms_suffice': bool(pb),
       'pred_c_code_is_needed': bool(pc), 'self_reviewed': True, 'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open(os.path.join(BQ, 'm16_two_number_interface_results.json'), 'w'), indent=1)
print(f'median held-out R^2: all-cells {mf:+.4f} | two-arms {ma:+.4f} | mean-code {mm_:+.4f} | permuted {mp:+.4f}')
print(f"(a) all-cells {mf:+.4f} >= 0.5: {'HELD' if pa else 'FAILED'}")
print(f"(b) two-arms {ma:+.4f} >= 0.8 x all-cells: {'HELD' if pb else 'FAILED'}")
print(f"(c) mean-code and permuted <= 0.1: {'HELD' if pc else 'FAILED'}")
print(f'wrote m16_two_number_interface_results.json ({time.time() - T0:.0f}s)')
