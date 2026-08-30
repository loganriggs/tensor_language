# RUNG 10, STEP 2: does m16's PER-DOCUMENT GAIN track the document's SENTENCE-BOUNDARY density?
#
# §2098 settled the shape of the shared m16 failure: the six m16 source circuits' deletion-response block has a
# document-STABLE rank-2 source subspace (0.8779 transferred across a prospective document split, equal to the
# in-sample share), so the other lane's programs -- which gave m16 a private rank of 1, 2 or 4 and still failed
# m16->* on held-out documents at NRMSE 2.4-3.2 -- were not short of directions. What they could not predict is
# the per-document COEFFICIENT on that fixed basis: m16's rows are 2.7x the tensor RMS and vary by document.
# §715 named mlp16's rank-1 output core as a sentence-boundary -> continuation writer (fires on . ) ? ! :). If
# the deletion response of m16's circuits is that core's work, a document's m16 gain should scale with how
# often the core fires in it -- its sentence-boundary density -- and that is a SURFACE feature any program can
# read for free, i.e. exactly the "observable document feature" the other lane's explanation_1405 §14.3 asks
# for in place of a free per-document code.
#
# LAWFUL PATH TO THE TOKENS. The FIT documents ARE this lane's rows: the other lane's FIT authority registers
# census_state_diverse.pt (c785f3d9...) and curated_rows.pt (faaf89f3...) as parents, curated['docid'] maps
# each of the 1000 rows to its source document, and the training input's document_ids are those ids. Only the
# first 256 positions of a row enter the response (positions_per_row 256). census_state_diverse['basev'] holds
# the live model's per-position base CE on the same 256,000 positions, so per-document base CE (the text
# difficulty §2083/§2085 found drives gating) comes for free as the control covariate. No validation or EVAL
# document is touched: everything here is keyed by the 229 training document ids from the replayed input.
#
# QUANTITIES, per training document d:
#   gain_d      RMS of the m16 source block R[:, m16 rows, :, d] over valid cells (the block's amplitude for d)
#   bnd_d       fraction of the document's tokens (positions 0..255 of its FIT rows) that decode to a string
#               whose last non-space character is one of . ? ! : )      (§715's firing set, comma excluded)
#   ce_d        mean base CE over the same positions
#   the same gain for every other owner block, as the specificity comparison
#
# REGISTERED PREDICTIONS:
#   (a) GAIN TRACKS BOUNDARIES: Spearman rho(gain_m16, bnd) >= 0.50 over the 229 documents. The 0.50 bar is the
#       one §2083 registered for its own "base CE explains the spread" claim (|r| >= 0.5) -- the same program's
#       precedent for "a document feature explains a per-window quantity", reused rather than invented.
#       If FALSE, m16's per-document gain is not (mainly) how often the boundary core fires, and the surface
#       route to a free document code for m16 is closed at this feature.
#   (b) AND IT IS m16's FEATURE: rho(gain_m16, bnd) exceeds rho(gain_o, bnd) for EVERY other owner o in
#       {a8, a16, a3, m14, m13}. If FALSE, boundary density is a generic amplitude covariate of the response
#       tensor (like the source-side low rank §2098's null exposed) and says nothing about m16's mechanism.
#   (c) NULL AND CONFOUND, and (a) may not be read without both: (i) with documents' boundary densities
#       PERMUTED (1000 draws), the 95th percentile of |rho| is below the observed rho; (ii) the partial Spearman
#       correlation of gain_m16 with bnd controlling for base CE stays >= 0.30 -- boundary density must carry
#       information beyond "hard text has big responses". If (ii) fails while (a) holds, the finding is
#       difficulty, not boundaries, and is reported as such.
#
# Descriptive, NOT registered: the same correlations per m16 source family ({r.1.1.1, r.1.2, r.1.2.0, r.1.2.1}
# vs {r.1.1.2, r.6.2.2}), and rho(gain_m16, ce) itself.
#
# Writes m16_gain_vs_boundary_results.json.
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
CENSUS = os.path.join(BQ, 'census_state_diverse.pt')
CURATED = os.path.join(BQ, 'curated_rows.pt')
CENSUS_SHA = 'c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b'
CURATED_SHA = 'faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd'
PRIOR = os.path.join(BQ, 'm16_response_block_split_results.json')
BAR_A, BAR_C_PARTIAL, NPERM = 0.50, 0.30, 1000
BOUNDARY = ('.', '?', '!', ':', ')')
if os.environ.get('BQLIB_DRYRUN') == '1':
    miss = [f for f in (INPUT, RECEIPT, CENSUS, CURATED, PRIOR) if not os.path.exists(f)]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    p = json.load(open(PRIOR))
    if not p.get('pred_a_rank2_subspace_transfers'):
        print('DRYRUN FAIL: S2098 did not establish a document-stable basis'); raise SystemExit(1)
    print(f"DRYRUN OK: S2098 transfer {p['per_owner']['m16']['transfer_top2_A_to_B']}; bars (a) {BAR_A} "
          f"(c.ii) {BAR_C_PARTIAL}; {NPERM} permutations")
    raise SystemExit(0)

from pathlib import Path                                                  # noqa: E402

import tiktoken                                                           # noqa: E402
import torch                                                              # noqa: E402

from causal_response_factorization_v1_training_input import replay_training_input   # noqa: E402

T0 = time.time()


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


if sha(CENSUS) != CENSUS_SHA or sha(CURATED) != CURATED_SHA:
    print('PARENT MISMATCH: census/curated rows are not the FIT authority parents; refusing.')
    raise SystemExit(2)
TERM = json.load(open(RECEIPT))
V, DIGEST = replay_training_input(
    Path(INPUT), expected_analysis_authority_sha256=TERM['authority_logical_sha256'],
    expected_artifact_sha256=TERM['payload']['input_sha256'], require_production=True)
R = V.response.clone()
R[~V.valid] = 0.0
VALID = V.valid
OWNERS = list(V.owner_components)
GROUPS = V.source_groups
DOCS = V.document_ids.tolist()
ST = torch.load(CENSUS, map_location='cpu', weights_only=False)
CU = torch.load(CURATED, map_location='cpu', weights_only=False)
ROWS, DOCID, BASEV = CU['rows'], CU['docid'], ST['basev'].float().view(1000, 256)
if not torch.equal(ROWS, ST['rows']):
    print('census/curated rows differ; refusing.'); raise SystemExit(2)
ENC = tiktoken.get_encoding('gpt2')
print(f'replayed {DIGEST[:12]} | {len(DOCS)} training documents | owners {OWNERS}', flush=True)

# token -> boundary flag, cached per vocabulary id
BND = {}


def is_boundary(tok):
    if tok not in BND:
        s = ENC.decode([tok]).rstrip()
        BND[tok] = bool(s) and s[-1] in BOUNDARY
    return BND[tok]


gain = {o: [] for o in OWNERS}
fam = {'A': [], 'B': []}
bnd, ce, nrows = [], [], []
m16_rows = (GROUPS == OWNERS.index('m16')).nonzero().squeeze(1)
tags = list(V.source_tags)
FAM_A = [i for i, r in enumerate(m16_rows.tolist()) if tags[r] in ('r.1.1.1', 'r.1.2', 'r.1.2.0', 'r.1.2.1')]
FAM_B = [i for i in range(len(m16_rows)) if i not in FAM_A]
for d_index, d in enumerate(DOCS):
    rows = (DOCID == d).nonzero().squeeze(1)
    toks = ROWS[rows, :256].reshape(-1).tolist()
    bnd.append(sum(is_boundary(t) for t in toks) / len(toks))
    ce.append(float(BASEV[rows].mean()))
    nrows.append(int(len(rows)))
    for g, o in enumerate(OWNERS):
        src = (GROUPS == g)
        blk = R[:, src, :, d_index]
        msk = VALID[:, src, :, d_index]
        gain[o].append(float((blk[msk] ** 2).mean().sqrt()) if int(msk.sum()) else float('nan'))
    blk = R[:, m16_rows, :, d_index]
    msk = VALID[:, m16_rows, :, d_index]
    for name, idx in (('A', FAM_A), ('B', FAM_B)):
        b, m = blk[:, idx], msk[:, idx]
        fam[name].append(float((b[m] ** 2).mean().sqrt()) if int(m.sum()) else float('nan'))


def ranks(x):
    x = torch.tensor(x, dtype=torch.float64)
    o = x.argsort()
    r = torch.empty_like(x); r[o] = torch.arange(len(x), dtype=torch.float64)
    # average ties
    _, inv, cnt = torch.unique(x, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(cnt), dtype=torch.float64).index_add_(0, inv, r)
    return sums[inv] / cnt[inv].double()


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12))


def partial_spearman(x, y, z):
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    rxy, rxz, ryz = [float(torch.corrcoef(torch.stack([a, b]))[0, 1]) for a, b in ((rx, ry), (rx, rz), (ry, rz))]
    return (rxy - rxz * ryz) / max(((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5, 1e-12)


keep = [i for i in range(len(DOCS)) if gain['m16'][i] == gain['m16'][i]]
G = {o: [gain[o][i] for i in keep] for o in OWNERS}
B = [bnd[i] for i in keep]
CE = [ce[i] for i in keep]
rho = {o: spearman(G[o], B) for o in OWNERS}
rho_ce = {o: spearman(G[o], CE) for o in OWNERS}
rho_fam = {k: spearman([fam[k][i] for i in keep], B) for k in fam}
partial = partial_spearman(G['m16'], B, CE)
g = torch.Generator().manual_seed(0)
null = sorted(abs(spearman(G['m16'], [B[j] for j in torch.randperm(len(B), generator=g).tolist()]))
              for _ in range(NPERM))
null_95 = null[int(0.95 * NPERM) - 1]
pa = rho['m16'] >= BAR_A
pb = all(rho['m16'] > rho[o] for o in OWNERS if o != 'm16')
pc_i = rho['m16'] > null_95
pc_ii = partial >= BAR_C_PARTIAL
pc = pc_i and pc_ii
out = {'training_input_sha256': DIGEST, 'documents_scored': len(keep), 'boundary_chars': list(BOUNDARY),
       'rows_per_document': {'min': min(nrows), 'max': max(nrows)},
       'boundary_density': {'min': round(min(B), 4), 'median': round(sorted(B)[len(B) // 2], 4),
                            'max': round(max(B), 4)},
       'spearman_gain_vs_boundary': {o: round(v, 4) for o, v in rho.items()},
       'spearman_gain_vs_base_ce': {o: round(v, 4) for o, v in rho_ce.items()},
       'spearman_boundary_vs_base_ce': round(spearman(B, CE), 4),
       'm16_partial_spearman_given_base_ce': round(partial, 4),
       'm16_family_rho': {k: round(v, 4) for k, v in rho_fam.items()},
       'm16_families': {'A': [tags[m16_rows[i]] for i in FAM_A], 'B': [tags[m16_rows[i]] for i in FAM_B]},
       'null_abs_rho': {'draws': NPERM, 'median': round(null[NPERM // 2], 4), 'p95': round(null_95, 4),
                        'max': round(null[-1], 4)},
       'bars': {'a': BAR_A, 'c_partial': BAR_C_PARTIAL},
       'pred_a_gain_tracks_boundaries': bool(pa), 'pred_b_specific_to_m16': bool(pb),
       'pred_c_null_and_confound': bool(pc), 'pred_c_i_beats_null': bool(pc_i),
       'pred_c_ii_partial_given_ce': bool(pc_ii), 'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open('m16_gain_vs_boundary_results.json', 'w'), indent=1)
print('rho(gain, boundary) by owner: ' + ' '.join(f'{o} {v:+.3f}' for o, v in rho.items()))
print('rho(gain, base CE)  by owner: ' + ' '.join(f'{o} {v:+.3f}' for o, v in rho_ce.items()))
print(f'rho(boundary, base CE) {spearman(B, CE):+.3f} | m16 partial rho | CE {partial:+.3f} | families '
      f'A {rho_fam["A"]:+.3f} B {rho_fam["B"]:+.3f} | null |rho| p95 {null_95:.3f}')
print(f'(a) m16 rho {rho["m16"]:.4f} >= {BAR_A}: {"HELD" if pa else "FAILED"}')
print(f'(b) m16 exceeds every other owner: {"HELD" if pb else "FAILED"}')
print(f'(c) beats null p95 {null_95:.3f}: {"HELD" if pc_i else "FAILED"} | partial >= {BAR_C_PARTIAL}: '
      f'{"HELD" if pc_ii else "FAILED"}')
print(f'wrote m16_gain_vs_boundary_results.json ({time.time() - T0:.0f}s)')
