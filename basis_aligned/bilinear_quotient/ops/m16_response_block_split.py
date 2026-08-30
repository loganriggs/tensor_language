# THE m16 TARGET, first registered measurement: is the m16 deletion-response block a DOCUMENT-STABLE
# low-rank object, or does its low rank live inside each document half and not across them?
#
# BACKLOG rung 10 (shared with the polynomial_causal lane). CPU only; reads the OTHER lane's published
# training-role artifact by content hash (causal_response_factorization_v1_training_input.pt, 229 FIT
# training documents, replayed against the training terminal receipt). No validation or EVAL value is
# touched: the 114 validation documents live behind that lane's lifecycle and are not on this path.
#
# WHAT IS ALREADY MEASURED (design pass, 2026-08-30, in-sample on all 229 documents, NOT registered):
#   the six m16 source circuits' rows have RMS 0.4923 against 0.1803 overall (7.7x a8, 43x a16) --
#   the amplitude story of the other lane's 1350 review, reproduced; their 6 x (2*49*229) unfolding has
#   top-1 energy share 0.6565 and top-2 0.8775, against a null of random 6-row blocks from the other 43
#   sources at median 0.63 / 95th 0.87. So mlp16's RANK-1 output core (§713/§715: one direction carries
#   90% of its 0.88-nat benefit) does NOT make its deletion-response block rank-1 across its six census
#   circuits; two families are visible in the source cosines ({r.1.1.1, r.1.2, r.1.2.0, r.1.2.1} and
#   {r.1.1.2, r.6.2.2}). Rank-2 gets 0.88 -- in sample. The candidate library on the other lane gave m16
#   a private rank of 1, 2 or 4 and still failed m16->* on held-out documents at NRMSE 2.4-3.2, which is
#   only consistent with rank-2 if the rank-2 subspace is DOCUMENT-SPECIFIC. That is the registered
#   question, and it is decided here on a prospective split of the training documents so nothing seen
#   above is scored again.
#
# METHOD. Documents are split by SHA-256 of "m16-block-split|<document id>" into halves A and B (a new
# salt; the split is not the other lane's train/validation split). For every owner block, and for 200
# random 6-row blocks drawn from the 43 non-m16 sources (the null), fit the top-k right-singular
# subspace of the source x (phase, target, document) unfolding on half A's columns and measure the
# fraction of half B's energy that subspace captures ("transfer"), k = 1, 2. The source partition is
# the sign split of the second left-singular vector on each half.
#
# REGISTERED PREDICTIONS:
#   (a) THE RANK-2 SUBSPACE TRANSFERS: m16's cross-half top-2 transfer >= 0.78, i.e. within 0.10 of the
#       in-sample top-2 share 0.8775 measured above (a bar set from that measured number, not a round
#       one). If FALSE, the m16 block's low rank is a within-half artefact: its response directions are
#       document-specific, no fixed low-rank program can fit m16->* on new documents, and the other
#       lane's failure at m16 is structural rather than a normalisation problem.
#   (b) AND THE TRANSFER IS SPECIFIC: m16's top-2 transfer exceeds the 95th percentile of the 200-draw
#       null. If FALSE, any six rows transfer this well and (a) says nothing about m16.
#   (c) THE TWO FAMILIES ARE THE SAME ON BOTH HALVES: the second-singular-vector sign partition of the
#       six sources agrees between A and B exactly (up to global sign). If FALSE, "two families" was a
#       reading of one half's noise. Also reported, per LESSON 42 style: both halves' in-sample top-1/
#       top-2 shares, which must bracket the all-document 0.6565 / 0.8775 above.
#
# Writes m16_response_block_split_results.json.
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
IN_SAMPLE_TOP1, IN_SAMPLE_TOP2 = 0.6565, 0.8775
BAR_A = round(IN_SAMPLE_TOP2 - 0.10, 4)
NDRAW = 200
SALT = 'm16-block-split'
if os.environ.get('BQLIB_DRYRUN') == '1':
    miss = [f for f in (INPUT, RECEIPT) if not os.path.exists(f)]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    r = json.load(open(RECEIPT))
    if r.get('kind') != 'receipt' or 'input_sha256' not in r.get('payload', {}):
        print('DRYRUN FAIL: training receipt is not a success receipt'); raise SystemExit(1)
    print(f'DRYRUN OK: training input bound to receipt {r["payload"]["input_sha256"][:12]}; '
          f'bar (a) {BAR_A}; {NDRAW} null draws')
    raise SystemExit(0)

import torch                                                              # noqa: E402

from causal_response_factorization_v1_training_input import replay_training_input   # noqa: E402

T0 = time.time()
TERM = json.load(open(RECEIPT))
V, DIGEST = replay_training_input(
    INPUT, expected_analysis_authority_sha256=TERM['authority_logical_sha256'],
    expected_artifact_sha256=TERM['payload']['input_sha256'], require_production=True)
R = V.response.clone()
R[~V.valid] = 0.0
OWNERS = list(V.owner_components)
GROUPS = V.source_groups
M16 = OWNERS.index('m16')
DOCS = V.document_ids.tolist()
print(f'replayed {DIGEST[:12]} {tuple(R.shape)} | owners {OWNERS} | groups '
      f'{torch.bincount(GROUPS).tolist()}', flush=True)

keyed = sorted((hashlib.sha256(f'{SALT}|{d}'.encode()).hexdigest(), i) for i, d in enumerate(DOCS))
A_IDX = torch.tensor(sorted(i for _, i in keyed[:len(keyed) // 2]))
B_IDX = torch.tensor(sorted(i for _, i in keyed[len(keyed) // 2:]))


def unfold(rows, docs):
    """sources x (phase, target, document) for the given source rows and document columns."""
    return R[:, rows][..., docs].permute(1, 0, 2, 3).reshape(len(rows), -1)


def shares(U, k):
    s = torch.linalg.svdvals(U)
    return float((s[:k] ** 2).sum() / (s ** 2).sum())


def transfer(rows, k):
    """Fit top-k right subspace on half A, report the energy fraction it captures on half B."""
    UA = unfold(rows, A_IDX)
    UB = unfold(rows, B_IDX)
    # right singular subspace lives in the (phase,target,document) axis, which differs between halves;
    # the shared axis is the SOURCE side, so transfer the top-k LEFT subspace (source loadings).
    Ua, _, _ = torch.linalg.svd(UA, full_matrices=False)
    P = Ua[:, :k]
    captured = float(((P.T @ UB) ** 2).sum() / (UB ** 2).sum())
    return captured, shares(UA, k), shares(UB, k)


def partition(rows, docs):
    U = unfold(rows, docs)
    Ua, _, _ = torch.linalg.svd(U, full_matrices=False)
    sgn = (Ua[:, 1] >= 0).tolist()
    if not sgn[0]:
        sgn = [not b for b in sgn]
    return sgn


rows_of = {o: (GROUPS == g).nonzero().squeeze(1) for g, o in enumerate(OWNERS)}
per_owner = {}
for o, rows in rows_of.items():
    t1, a1, b1 = transfer(rows, 1)
    t2, a2, b2 = transfer(rows, 2)
    per_owner[o] = {'n_sources': int(len(rows)),
                    'rms_all': round(float((R[:, rows] ** 2).mean().sqrt()), 4),
                    'top1_share_A': round(a1, 4), 'top1_share_B': round(b1, 4),
                    'top2_share_A': round(a2, 4), 'top2_share_B': round(b2, 4),
                    'transfer_top1_A_to_B': round(t1, 4), 'transfer_top2_A_to_B': round(t2, 4)}
    print(f'  {o:>4} n={len(rows):2d} top-2 share A {a2:.4f} B {b2:.4f}  transfer(A->B) top-1 '
          f'{t1:.4f} top-2 {t2:.4f}', flush=True)

g = torch.Generator().manual_seed(0)
others = (GROUPS != M16).nonzero().squeeze(1)
draws = []
for _ in range(NDRAW):
    pick = others[torch.randperm(len(others), generator=g)[:6]]
    draws.append(transfer(pick, 2)[0])
null = sorted(draws)
null_95 = null[int(0.95 * NDRAW) - 1]
m16_rows = rows_of['m16']
part_A = partition(m16_rows, A_IDX)
part_B = partition(m16_rows, B_IDX)
m16_t2 = per_owner['m16']['transfer_top2_A_to_B']
pa = m16_t2 >= BAR_A
pb = m16_t2 > null_95
pc = part_A == part_B
tags = list(V.source_tags)
out = {'training_input_sha256': DIGEST, 'documents': len(DOCS), 'half_A': int(len(A_IDX)),
       'half_B': int(len(B_IDX)), 'split_salt': SALT,
       'in_sample_all_docs': {'top1': IN_SAMPLE_TOP1, 'top2': IN_SAMPLE_TOP2},
       'bar_a': BAR_A, 'per_owner': per_owner,
       'm16_sources': [tags[i] for i in m16_rows.tolist()],
       'm16_partition_A': part_A, 'm16_partition_B': part_B,
       'null_transfer_top2': {'draws': NDRAW, 'median': round(null[NDRAW // 2], 4),
                              'p95': round(null_95, 4), 'max': round(null[-1], 4)},
       'pred_a_rank2_subspace_transfers': bool(pa),
       'pred_b_transfer_specific_to_m16': bool(pb),
       'pred_c_families_agree_across_halves': bool(pc),
       'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open('m16_response_block_split_results.json', 'w'), indent=1)
print(f'\nm16 top-2 transfer A->B {m16_t2:.4f} | bar {BAR_A} | null median {null[NDRAW // 2]:.4f} '
      f'p95 {null_95:.4f} max {null[-1]:.4f}')
print(f'(a) transfers >= {BAR_A}: {"HELD" if pa else "FAILED"}')
print(f'(b) beats null p95 {null_95:.4f}: {"HELD" if pb else "FAILED"}')
print(f'(c) families agree: A {part_A} B {part_B}: {"HELD" if pc else "FAILED"}')
if not pa:
    print('    READING: the m16 block\'s low rank is document-specific; no fixed low-rank program fits '
          'm16->* on new documents -- the other lane\'s m16 failure is structural, not normalisation.')
print(f'wrote m16_response_block_split_results.json ({time.time() - T0:.0f}s)')
