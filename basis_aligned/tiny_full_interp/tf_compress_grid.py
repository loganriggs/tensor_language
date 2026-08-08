"""COMPRESSIBILITY ACROSS THE GRID — is "structure does not compress" a
small-model artifact, or a property of this architecture family?

FINDING 12 §7b established the one honest scalar for this question:

    ratio = bits(naive per-row uniform quantisation + entropy coding)
            -------------------------------------------------------
            bits(the best description we can build)      at the SAME score

measured at exactly one cell (depth 1, width 128), where it is 1.13–1.54 across
25 frontier points, median 1.20.  A single cell cannot distinguish "the family
does not compress" from "the smallest model does not compress".  This module
computes the same scalar at every cell of the grid.

THREE THINGS THIS MODULE DOES THAT `tf_compress_run.py` DOES NOT
----------------------------------------------------------------
1. **Depth-general.** `tf_compress.D1Desc` asserts `cfg.depth == 1` and
   re-implements the depth-1 forward.  Here the decoder writes described tables
   straight back into the model's own parameters and calls the model's own
   forward, so it is exact at any depth by construction (positive control:
   the identity description returns KL exactly 0.0, not 1e-6).
2. **Comparability across cells is measured, not assumed** (the adversarial
   review item).  Different widths have different weight counts and different
   KL scales, so a fixed absolute KL is a different *difficulty* at every cell.
   Every ratio is therefore reported at BOTH an absolute score level and a
   level normalised by that cell's own headroom over the unigram floor, and the
   naive baseline's fixed-overhead share (row scales + histograms, which
   amortise differently at different widths) is reported per cell so a trend in
   the ratio cannot be a trend in the denominator's overhead.
3. **Structure and recoding are separated.**  The best-of-everything ratio
   answers "is fp32 a bad file format".  The question here is whether a
   description made out of an INTERPRETATION compresses, so a second ratio is
   computed with the numerator restricted to structural families (low rank,
   row clustering, product quantisation, exact anchor rows + a coded tail) and
   recodings excluded.

Held CE is the primary score (Logan's standing correction); KL from the model
is secondary.  Quantisation is labelled RECODING everywhere, never explanation.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import tf_corpus
import tf_fold
import tf_model as M
from tf_compress import (Bits, bits_dense, bits_index, entropy_bits, q_cluster,
                         q_lowrank, q_pq, q_scalar, q_scalar_entropy,
                         q_stratified, q_transform)

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===========================================================================
# THE DEPTH-GENERAL DECODER
# ===========================================================================
class GDesc:
    """A description is a set of replacement tables; the decoder is the model's
    own forward pass reading them.  Every table it needs is charged.

    Using the model's forward rather than a re-implementation is what makes
    this depth-general AND makes the positive control exact: with no
    replacements the description IS the model, so KL must be 0.0 to the bit.
    """

    def __init__(self, stem, device=DEV, n_seq=64, T=256, batch=8):
        model, cfg, ck = tf_fold.load_checkpoint(stem, device)
        assert cfg.variant == 'vanilla' and cfg.n_slots == 1, \
            'the grid scalar is defined on the plain model'
        self.stem, self.model, self.cfg, self.dev = stem, model, cfg, device
        self.V, self.L, self.Ws = cfg.vocab, cfg.depth, model.Ws
        self.n_seq, self.T, self.batch = n_seq, T, batch
        self.params = {'wte': model.wte.weight}
        for li, blk in enumerate(model.h):
            for nm in ('c_q', 'c_k', 'c_q2', 'c_k2', 'c_v', 'c_proj',
                       'Left', 'Right', 'Down'):
                self.params[f'l{li}.{nm}'] = getattr(blk, nm).weight
            self.params[f'l{li}.Down_bias'] = blk.Down_bias
        self.base = {k: v.detach().float().clone()
                     for k, v in self.params.items()}
        self.emb_keys = ['wte']
        self.body_keys = [k for k in self.base if k != 'wte']
        self.n_emb = self.base['wte'].numel()
        self.n_body = sum(self.base[k].numel() for k in self.body_keys)
        self.n_params = self.n_emb + self.n_body
        self._ref = None

    # ------------------------------------------------------------- plumbing
    def _install(self, P):
        with torch.no_grad():
            for k, p in self.params.items():
                p.copy_((P.get(k, self.base[k])).to(p.dtype))

    def batches(self, split='held'):
        arr = tf_corpus.load_split(self.V, split, self.n_seq, tok=self.cfg.tok)
        x = torch.from_numpy(arr[:, :self.T + 1]).to(self.dev)
        for a in range(0, x.shape[0], self.batch):
            bb = x[a:a + self.batch]
            yield bb[:, :-1], bb[:, 1:]

    @torch.no_grad()
    def cache_ref(self):
        """Reference log-probabilities in FLOAT32.  FINDING 12's review (R2)
        found the original cached them in fp16, which put the measurement floor
        at ~1e-5 and produced a NEGATIVE control KL.  fp32 costs a few hundred
        MB here and removes the floor entirely."""
        if self._ref is None:
            self._install({})
            ref = []
            for x, y in self.batches():
                lp = F.log_softmax(self.model(x).float(), -1)
                ref.append((x, y, lp, lp.exp()))
            self._ref = ref
        return self._ref

    @torch.no_grad()
    def score(self, P=None):
        ref = self.cache_ref()
        self._install(P or {})
        kl = ce = 0.0
        n = 0
        per = []
        for x, y, lpr, pr in ref:
            lp = F.log_softmax(self.model(x).float(), -1)
            k = (pr * (lpr - lp)).sum(-1)
            kl += float(k.sum())
            ce += float(F.cross_entropy(lp.reshape(-1, self.V),
                                        y.reshape(-1), reduction='sum'))
            per.append(k.mean(-1))
            n += y.numel()
        self._install({})
        per = torch.cat(per)
        return {'kl': kl / n, 'ce': ce / n, 'ntok': n,
                'kl_se': float(per.std(unbiased=True) / math.sqrt(len(per)))}

    @torch.no_grad()
    def positive_control(self):
        s = self.score({})
        return {'kl_identity': s['kl'], 'ce_identity': s['ce'],
                'exact': s['kl'] == 0.0}


# ===========================================================================
# THE SCHEME FAMILY — identical at every cell
# ===========================================================================
def _freq_order(V, tok):
    """Token frequency on the ESTIMATION split (never held).  Used only as an
    importance ordering; the corpus is declared free to the decoder, exactly as
    in FINDING 12."""
    arr = tf_corpus.load_split(V, 'est', 4000, tok=tok)
    c = np.bincount(arr.reshape(-1), minlength=V).astype(np.float64)
    return torch.from_numpy(np.argsort(-c)).long(), torch.from_numpy(c).float()


def emb_schemes(E, V, d, order, freq, dev):
    """(name, kind, reconstruction, Bits) for the embedding table.

    kind is 'naive' | 'recode' | 'structure'.  'structure' means a description
    that asserts something about the table's organisation (a low-rank basis, a
    set of prototypes, a subspace codebook, a set of important rows); 'recode'
    means the same numbers in a better file format.
    """
    out = []
    for b in (2, 3, 4, 5, 6, 8):
        R, bt = q_scalar_entropy(E, b)
        out.append((f'emb_naive{b}', 'naive', R, bt))
    # THE STRENGTHENED DENOMINATOR (adversarial review).  A per-ROW scale costs
    # 32 bits per row whatever the width, which is 1.0 bits/weight of pure
    # overhead at width 32 and 0.125 at width 256 -- so the registered
    # per-row denominator is systematically WEAKER at small width, and a trend
    # in the ratio could be nothing but that.  The per-tensor-scale variant has
    # no such overhead; the strengthened denominator is the better of the two.
    for b in (2, 3, 4, 5, 6, 8):
        R, bt = q_scalar_entropy(E, b, group='tensor')
        out.append((f'emb_naiveG{b}', 'naiveG', R, bt))
    # ---- recodings: per-column scales + per-column entropy, rotated or not
    for bpr in (2, 3, 4, 5, 6):
        for rot in ('none', 'pca'):
            R, bt = q_transform(E, bpr * d, rot=rot, entropy=True, wrow=freq)
            out.append((f'emb_T{rot}{bpr}', 'recode', R, bt))
    # ---- recoding: frequency-stratified precision (the smooth anchor)
    for (bh, bl) in ((8, 2), (8, 3), (6, 2), (6, 3), (12, 3)):
        for nh in (512, 2048):
            R, bt = q_stratified(E, bh, bl, nh, order.to(dev))
            out.append((f'emb_strat{bh}_{bl}_n{nh}', 'recode', R, bt))
    # ---- structure: low rank
    for r in sorted({max(2, d // 16), max(2, d // 8), d // 4, d // 2}):
        if r < 1 or r >= d:
            continue
        for b in (8, 32):
            R, bt = q_lowrank(E, r, b)
            out.append((f'emb_lr{r}b{b}', 'structure', R, bt))
    # ---- structure: row prototypes (frequency-weighted Lloyd)
    for k in (512, 1024, 2048, 4096):
        for b in (8, 32):
            R, bt = q_cluster(E, k, b=b, weights=freq)
            out.append((f'emb_clu{k}b{b}', 'structure', R, bt))
    # ---- structure: product quantisation (subspace codebooks)
    for m in (2, 4, 8, 16, 32):
        if d % m or d // m < 2:
            continue
        for nb in (6, 8):
            R, bt = q_pq(E, m, nb)
            out.append((f'emb_pq{m}x{nb}', 'structure', R, bt))
    # ---- structure: EXACT ANCHOR ROWS for the top tokens + a coded tail
    #      (the parent program's frontier winner, ported in FINDING 13)
    for B in (256, 1024):
        for k in (512, 2048):
            idx = order[:B].to(dev)
            mask = torch.ones(V, dtype=torch.bool, device=dev)
            mask[idx] = False
            tail = torch.nonzero(mask).squeeze(1)
            Rt, bt = q_cluster(E[tail], k, b=8, weights=freq.to(dev)[tail])
            R = E.clone()
            R[tail] = Rt
            bb = Bits(anchor_ids=bits_index(B, V),
                      anchor_rows=bits_dense(B * d, 32)).merge(bt, 'tail_')
            out.append((f'emb_anch{B}_clu{k}', 'structure', R, bb))
    # ---- structure PLUS AN HONEST REMAINDER.  Without this the structural
    #      family simply cannot reach small KL at any price, and "structure
    #      never reaches the level" would be confounded with "structure is
    #      expensive".  Prototypes / a low-rank basis, then the residual coded.
    #      If the structure is real the residual is cheaper to code than the
    #      raw table, and the pair beats naive quantisation outright.
    for k in (512, 2048):
        C, a = None, None
        Rc, btc = q_cluster(E, k, b=32, weights=freq)
        for b in (2, 3, 4):
            Rr, btr = q_scalar_entropy(E - Rc, b)
            bb = Bits().merge(btc, 'clu_').merge(btr, 'res_')
            out.append((f'emb_clu{k}_res{b}', 'structure', Rc + Rr, bb))
    for r in sorted({max(2, d // 4), max(2, d // 2)}):
        if r >= d:
            continue
        Rl, btl = q_lowrank(E, r, 32)
        for b in (2, 3, 4):
            Rr, btr = q_scalar_entropy(E - Rl, b)
            bb = Bits().merge(btl, 'lr_').merge(btr, 'res_')
            out.append((f'emb_lr{r}_res{b}', 'structure', Rl + Rr, bb))
    return out


def body_schemes(D, dev):
    """(name, kind, {key: reconstruction}, Bits) for the body tables."""
    out = []
    for b in (3, 4, 5, 6, 8, 12):
        rec, bt = {}, Bits()
        for k in D.body_keys:
            R, bk = q_scalar_entropy(D.base[k], b)
            rec[k] = R
            bt.merge(bk, k + '_')
        out.append((f'body_naive{b}', 'naive', rec, bt))
    for b in (3, 4, 5, 6, 8, 12):
        rec, bt = {}, Bits()
        for k in D.body_keys:
            R, bk = q_scalar_entropy(D.base[k], b, group='tensor')
            rec[k] = R
            bt.merge(bk, k + '_')
        out.append((f'body_naiveG{b}', 'naiveG', rec, bt))
    for frac in (0.25, 0.5):
        rec, bt = {}, Bits()
        for k in D.body_keys:
            W = D.base[k]
            if W.dim() < 2 or min(W.shape) < 4:
                R, bk = q_scalar_entropy(W, 8)
            else:
                r = max(1, int(frac * min(W.shape)))
                R, bk = q_lowrank(W, r, 8)
            rec[k] = R
            bt.merge(bk, k + '_')
        out.append((f'body_lr{frac}', 'structure', rec, bt))
    return out


# ===========================================================================
# THE FRONTIER AND THE SCALAR
# ===========================================================================
def _interp_bits(points, level, key):
    """Bits needed by a family to reach `level` of `key` (score, lower=better),
    by linear interpolation in (score, log bits).  Returns None if the family
    never reaches the level within its measured range."""
    pts = sorted(points, key=lambda p: p['bits'])
    # the family's own Pareto staircase: as bits grow the score must fall
    st, best = [], float('inf')
    for p in pts:
        if p[key] < best:
            best = p[key]
            st.append(p)
    if not st:
        return None
    if st[0][key] <= level:
        return st[0]['bits']          # already there at its cheapest point
    for a, b in zip(st, st[1:]):
        if b[key] <= level <= a[key]:
            if a[key] == b[key]:
                return b['bits']
            t = (a[key] - level) / (a[key] - b[key])
            return math.exp(math.log(a['bits'])
                            + t * (math.log(b['bits']) - math.log(a['bits'])))
    return None                        # never reaches it


def run_cell(stem, n_seq=64, T=256, quick=False, out_suffix='_cgrid'):
    t0 = time.time()
    D = GDesc(stem, n_seq=(16 if quick else n_seq), T=(128 if quick else T))
    rep = {'stem': stem, 'depth': D.L, 'width': D.cfg.width,
           'vocab': D.V, 'tok': D.cfg.tok,
           'n_params': D.n_params, 'n_emb': D.n_emb, 'n_body': D.n_body,
           'embedding_share_of_params': D.n_emb / D.n_params,
           'fp32_bits': 32 * D.n_params,
           'registered_predictions':
               json.load(open(f'{HERE}/tf_depth_ladder_predictions.json'))
               ['task_2_compressibility_across_the_grid']}
    rep['positive_control'] = D.positive_control()
    assert rep['positive_control']['exact'], rep['positive_control']
    base = D.score({})
    rep['model'] = {'held_ce': base['ce'], 'ntok': base['ntok']}
    bl = json.load(open(f'{HERE}/tf_baselines_b{D.V}.json'))
    rep['baselines'] = {'unigram_floor_held_ce': bl['unigram_floor_held_ce'],
                        'bigram_held_ce': bl['bigram_held_ce']}
    # the cell's own difficulty scale: everything the model knows over unigram
    head = bl['unigram_floor_held_ce'] - base['ce']
    rep['headroom_over_unigram_nats'] = head

    order, freq = _freq_order(D.V, D.cfg.tok)
    E = D.base['wte']
    d = E.shape[1]
    embs = emb_schemes(E, D.V, d, order, freq.to(D.dev), D.dev)
    bodies = body_schemes(D, D.dev)
    print(f'  {len(embs)} embedding x {len(bodies)} body schemes '
          f'({time.time()-t0:.0f}s to build)', flush=True)

    pts = []
    for en, ek, ER, eb in embs:
        for bn, bk, BR, bb in bodies:
            # keep the cross product honest but affordable: pair every
            # embedding coder with the naive body ladder, keep the two naive
            # scale-groupings internally consistent, and pair the structural
            # body only with the naive embedding ladder
            if (ek == 'naiveG') != (bk == 'naiveG'):
                continue
            if bk == 'structure' and ek != 'naive':
                continue
            P = dict(BR)
            P['wte'] = ER
            s = D.score(P)
            if ek == 'naive' and bk == 'naive':
                kind = 'naive'
            elif ek == 'naiveG' and bk == 'naiveG':
                kind = 'naiveG'
            elif 'structure' in (ek, bk):
                kind = 'structure'
            else:
                kind = 'recode'
            pts.append({'name': f'{en}+{bn}', 'kind': kind,
                        'emb_kind': ek, 'body_kind': bk,
                        'bits': eb.total + bb.total,
                        'emb_bits': eb.total, 'body_bits': bb.total,
                        'kl': s['kl'], 'ce': s['ce'], 'kl_se': s['kl_se'],
                        'bits_per_weight': (eb.total + bb.total) / D.n_params})
    print(f'  {len(pts)} descriptions scored ({time.time()-t0:.0f}s)',
          flush=True)
    rep['points'] = pts

    # ---- the naive baseline's fixed-overhead share (the comparability check)
    ov = {}
    for b in (2, 4, 8):
        _, eb = q_scalar_entropy(E, b)
        tot = eb.total
        ov[f'emb_b{b}'] = {'total_bits': tot, 'scale_bits': eb.items['scales'],
                           'scale_share': eb.items['scales'] / tot,
                           'bits_per_weight': tot / D.n_emb}
    rep['naive_overhead'] = ov

    # ---- the scalar, at matched score, absolute AND headroom-normalised
    naive = [p for p in pts if p['kind'] == 'naive']
    naiveG = [p for p in pts if p['kind'] == 'naiveG']
    strong = naive + naiveG          # the strengthened denominator
    allp = pts
    struct = [p for p in pts if p['kind'] in ('naive', 'structure')]
    struct_only = [p for p in pts if p['kind'] == 'structure']

    levels = {}
    for f in (0.002, 0.005, 0.01, 0.03, 0.1):
        levels[f'kl_{f}xheadroom'] = ('kl', f * head, f)
        levels[f'ce_{f}xheadroom'] = ('ce', base['ce'] + f * head, f)
    for a in (0.005, 0.02, 0.08, 0.3):
        levels[f'kl_abs{a}'] = ('kl', a, None)

    rows = {}
    for nm, (key, lev, frac) in levels.items():
        bn = _interp_bits(naive, lev, key)
        bg = _interp_bits(strong, lev, key)
        ba = _interp_bits(allp, lev, key)
        bs = _interp_bits(struct, lev, key)
        bso = _interp_bits(struct_only, lev, key)
        rows[nm] = {
            'key': key, 'level': lev, 'headroom_fraction': frac,
            'naive_bits': bn, 'naive_strong_bits': bg, 'best_bits': ba,
            'best_incl_structure_bits': bs, 'structure_only_bits': bso,
            'ratio_best_over_naive': (bn / ba) if (bn and ba) else None,
            'ratio_best_over_naive_strong': (bg / ba) if (bg and ba) else None,
            'ratio_structure_only': (bn / bso) if (bn and bso) else None,
            'ratio_structure_only_strong': (bg / bso) if (bg and bso) else None,
            'structure_reaches_this_level': bso is not None,
            'naive_bits_per_weight': (bn / D.n_params) if bn else None,
            'best_bits_per_weight': (ba / D.n_params) if ba else None,
            'best_scheme': min(
                [p for p in allp if p[key] <= lev],
                key=lambda p: p['bits'])['name']
            if any(p[key] <= lev for p in allp) else None,
            'best_structural_scheme': min(
                [p for p in struct_only if p[key] <= lev],
                key=lambda p: p['bits'])['name']
            if any(p[key] <= lev for p in struct_only) else None}
    rep['matched_score_ratios'] = rows
    def col(k):
        return [v[k] for v in rows.values() if v.get(k)]
    rr, rg = col('ratio_best_over_naive'), col('ratio_best_over_naive_strong')
    rs, rsg = col('ratio_structure_only'), col('ratio_structure_only_strong')
    rep['summary'] = {
        'ratio_median': float(np.median(rr)) if rr else None,
        'ratio_min': float(np.min(rr)) if rr else None,
        'ratio_max': float(np.max(rr)) if rr else None,
        'ratio_strong_median': float(np.median(rg)) if rg else None,
        'ratio_strong_min': float(np.min(rg)) if rg else None,
        'ratio_strong_max': float(np.max(rg)) if rg else None,
        'ratio_structure_only_median': float(np.median(rs)) if rs else None,
        'ratio_structure_only_strong_median':
            float(np.median(rsg)) if rsg else None,
        'levels_structure_can_reach':
            int(sum(v['structure_reaches_this_level'] for v in rows.values())),
        'n_levels': len(rows)}
    rep['seconds'] = round(time.time() - t0, 1)
    jp = f'{HERE}/{stem}{out_suffix}.json'
    json.dump(rep, open(jp, 'w'), indent=2)
    print(f'== {jp}  ratio median {rep["summary"]["ratio_median"]}  strong {rep["summary"]["ratio_strong_median"]}  '
          f'structure-only {rep["summary"]["ratio_structure_only_median"]}  '
          f'({rep["seconds"]}s)', flush=True)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n-seq', type=int, default=64)
    a = ap.parse_args()
    run_cell(a.stem, n_seq=a.n_seq, quick=a.quick)
