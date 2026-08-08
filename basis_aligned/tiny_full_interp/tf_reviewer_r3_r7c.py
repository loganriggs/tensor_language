"""R7c -- the apples-to-apples class comparison, on the EMBEDDING TABLE ALONE
with the body held exact at fp32 for every point, so the three description
classes are directly comparable.  Run after --o4, R7b (sparse dictionary) and
the analyst's own sections are all in the JSON."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = f'{HERE}/tf_reviewer_round_3_compression.json'
STRUCT = ('cluster_', 'lowrank_', 'pq_', 'anchor', 'transform_pca',
          'sparsedict_', 'writecluster_', 'readcluster_', 'vq_')
HYBRID = ('feature_residual_', 'corpusstat_residual_')


def cl(s):
    if s.startswith(HYBRID):
        return 1
    if s.startswith(STRUCT):
        return 2
    return 0


def front(ps):
    o = []
    for b, k, s in sorted(ps):
        if not o or k < o[-1][1] - 1e-12:
            o.append((b, k, s))
    return o


def hull(pts):
    """Lower convex hull in (bits, KL).  This -- not the raw staircase
    -- is the baseline a competing scheme has to beat, because any point on the
    chord between two achievable schemes is itself achievable by splitting the
    table and coding the halves differently.  Interpolating the staircase
    instead flatters the challenger at every budget between two recoding
    points, which is how a prototype code can look like a winner when it is
    not."""
    p = sorted((b, k) for b, k, *_ in pts if k > 0)
    h = []
    for q in p:
        while len(h) >= 2:
            (x1, y1), (x2, y2) = h[-2], h[-1]
            if (y2 - y1) * (q[0] - x1) >= (q[1] - y1) * (x2 - x1):
                h.pop()
            else:
                break
        h.append(q)
    return h


def kl_at(c, b):
    h = hull(c)
    for i in range(len(h) - 1):
        if h[i][0] <= b <= h[i + 1][0]:
            (x0, y0), (x1, y1) = h[i], h[i + 1]
            t = (b - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


def main():
    rev = json.load(open(P))
    d0 = json.load(open(f'{HERE}/tf_vanilla_d1_w128_b8192_s0_compress.json'))
    emb = {0: [], 1: [], 2: []}
    for sec in ('C_embedding', 'D_anchor', 'G_codes', 'K_features',
                'L_corpus_stats'):
        for r in d0.get(sec, {}).get('rows', []):
            if r.get('bits_embedding') is None:
                continue
            emb[cl(r['scheme'])].append((r['bits_embedding'], r['kl'],
                                         r['scheme']))
    for r in rev.get('R7b_sparse_dictionary_the_untried_structural_family',
                     {}).get('rows', []):
        emb[2].append((r['bits'], r['kl'], r['scheme']))
    for r in rev.get('O4_clustering', {}).get('rows', []):
        if r['role'] == 'both':
            emb[2].append((r['bits'], r['kl'], r['scheme']))
    fa = front(emb[0])
    out = {'recoding_front': [{'scheme': s, 'bits': b, 'kl': k}
                              for b, k, s in fa], 'comparisons': {}}
    for c, nm in ((1, 'hybrid_structure_plus_coded_residual'),
                  (2, 'pure_structure')):
        rowsx = []
        print('==', nm)
        for b, k, s in front(emb[c]):
            ka = kl_at(fa, b)
            rowsx.append({'scheme': s, 'bits': b, 'kl': k,
                          'recoding_kl_at_same_bits': ka,
                          'kl_penalty_x': (k / ka) if ka else None})
            print('  %-32s %6.3f Mbit KL %.5f  recoding %s -> %s'
                  % (s, b / 1e6, k, ('%.5f' % ka) if ka else '--',
                     ('%.2fx' % (k / ka)) if ka else '--'))
        out['comparisons'][nm] = rowsx
        ok = [r for r in rowsx if r['kl_penalty_x']]
        if ok:
            out['comparisons'][nm + '_summary'] = {
                'best_penalty_x': min(r['kl_penalty_x'] for r in ok),
                'worst_penalty_x': max(r['kl_penalty_x'] for r in ok),
                'penalty_below_1_only_at_bits_under': max(
                    [r['bits'] for r in ok if r['kl_penalty_x'] < 1.0] or [0])}
            print('  summary', out['comparisons'][nm + '_summary'])
    rev['R7c_embedding_only_class_fronts'] = out
    json.dump(rev, open(P, 'w'), indent=1)


if __name__ == '__main__':
    main()
