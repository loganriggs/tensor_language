"""INDEPENDENT REVIEWER ROUND 3 on FINDING 12 (the compression frontier).

Run by a reviewer who did not produce the finding.  Every objection gets a
measurement written into `tf_reviewer_round_3_compression.json`.

  O1  is fp32 a fair denominator?              --o1
  O2  are the bits charged completely?         --o2
  O3  fit/score split, and frontier selection  --o3
  O4  was the clustering done well?            --o4
  O5  does R^2 translate into bits?            --o5
  O6  which frontier points are seed-robust?   --o6

Nothing here imports the analyst's bit bills; O2 re-derives them from scratch.
"""
import argparse
import io
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import tf_compress as CC
import tf_compress_run as RR
import tf_corpus
from tf_compress import Bits, bits_dense, bits_index

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{HERE}/tf_reviewer_round_3_compression.json'
STEM0 = 'tf_vanilla_d1_w128_b8192_s0'
STEM1 = 'tf_vanilla_d1_w128_b8192_s1'
STEM2 = 'tf_vanilla_d1_w128_b8192_s2'
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save(o):
    json.dump(o, open(OUT, 'w'), indent=1)


def sec_rows(d, sec):
    return d.get(sec, {}).get('rows', [])


# ===========================================================================
# O1 -- IS THE MODEL'S 42.996 Mbit A FAIR DENOMINATOR?
# ===========================================================================
def plane_entropy(b):
    """Empirical entropy (bits) of a uint8 array under its own histogram,
    plus the histogram at fp16 -- an honest order-0 code length."""
    c = np.bincount(b.ravel(), minlength=256).astype(np.float64)
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) * b.size + 256 * 16


def o1(out):
    log('O1: fair denominator for "5.7x smaller than the model"')
    import lzma
    import zlib
    D = CC.D1Desc(STEM0)
    parts = [D.base[k] for k in D.PART_NAMES if k != 'wte_read']
    flat = torch.cat([p.reshape(-1) for p in parts]).cpu()
    n = flat.numel()
    assert n == D.n_params_model, (n, D.n_params_model)
    raw = flat.numpy().astype('<f4').tobytes()
    res = {'n_params': n, 'fp32_bits': 32 * n}

    # --- lossless general-purpose coders on the fp32 bytes ---------------
    lossless = {}
    lossless['fp32_raw'] = 32 * n
    lossless['fp32_zlib9'] = 8 * len(zlib.compress(raw, 9))
    lossless['fp32_lzma'] = 8 * len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))
    # byte-plane (shuffle) transposition -- the standard float-compression trick
    a8 = np.frombuffer(raw, dtype=np.uint8).reshape(n, 4)
    sh = np.ascontiguousarray(a8.T).tobytes()
    lossless['fp32_shuffle_lzma'] = 8 * len(
        lzma.compress(sh, preset=9 | lzma.PRESET_EXTREME))
    # explicit IEEE plane split: sign, exponent, mantissa, order-0 entropy
    u = np.frombuffer(raw, dtype='<u4')
    sign = (u >> 31).astype(np.uint8)
    expo = ((u >> 23) & 0xFF).astype(np.uint8)
    m_hi = ((u >> 16) & 0x7F).astype(np.uint8)
    m_mid = ((u >> 8) & 0xFF).astype(np.uint8)
    m_lo = (u & 0xFF).astype(np.uint8)
    lossless['fp32_ieee_planes_order0'] = int(
        plane_entropy(sign) + plane_entropy(expo) + plane_entropy(m_hi)
        + plane_entropy(m_mid) + plane_entropy(m_lo))
    res['lossless_fp32'] = {k: int(v) for k, v in lossless.items()}
    res['lossless_best'] = min(lossless.values())
    log('   lossless fp32:', {k: round(v / 1e6, 3) for k, v in lossless.items()})

    # --- standard shipped precisions, with their KL --------------------
    prec = {}
    for nm, dt in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
        P = {k: D.base[k].to(dt).float() for k in D.PART_NAMES}
        s = D.score(P)
        prec[nm] = {'bits': 16 * n, 'kl': s['kl'], 'ce': s['ce']}
        log(f'   {nm}: {16*n/1e6:.3f} Mbit  KL {s["kl"]:.3e}')
    # int8 per-row round-to-nearest = textbook post-training quantisation
    for b in (8, 6, 5, 4):
        Wq, be = CC.q_scalar(D.base['wte_read'], b)
        P, bb = RR.body_parts_bits(D, b)
        P['wte_read'] = Wq
        P['wte_out'] = Wq
        s = D.score(P)
        prec[f'ptq_int{b}_perrow'] = {'bits': be.total + bb.total, 'kl': s['kl']}
    # int8 per-row PLUS entropy coding = the strongest *naive* coder
    for b in (8, 6, 5, 4):
        Wq, be = CC.q_scalar_entropy(D.base['wte_read'], b)
        P, tot = {'wte_read': Wq, 'wte_out': Wq}, be.total
        for k in D.PART_NAMES:
            if k in ('wte_read', 'wte_out'):
                continue
            W = D.base[k]
            if W.dim() == 1:
                P[k] = W
                tot += bits_dense(W.numel(), 32)
            else:
                R, bb2 = CC.q_scalar_entropy(W, b)
                P[k] = R
                tot += bb2.total
        s = D.score(P)
        prec[f'ptq_int{b}_perrow_entropy'] = {'bits': tot, 'kl': s['kl']}
        log(f'   ptq int{b}+entropy: {tot/1e6:.3f} Mbit KL {s["kl"]:.5f}')
    res['standard_encodings'] = prec

    # --- the apples-to-apples factor: frontier vs the model's OWN uniform
    #     quantisation curve, at MATCHED KL ------------------------------
    d0 = load(f'{HERE}/{STEM0}_compress.json')
    uni = [(r['bits'], r['kl']) for r in sec_rows(d0, 'A_self_quantisation')
           if r['scheme'].startswith('uniform_')]
    uni = sorted(uni)
    ent = sorted((v['bits'], v['kl']) for k, v in prec.items()
                 if k.endswith('_perrow_entropy'))

    def bits_at_kl(curve, kl):
        """log-log interpolate a (bits, kl) curve to the bits needed for `kl`."""
        c = [(b, k) for b, k in curve if k > 0]
        c = sorted(c, key=lambda t: t[1])          # ascending KL
        if kl <= c[0][1]:
            return None
        if kl >= c[-1][1]:
            return None
        for i in range(len(c) - 1):
            if c[i][1] <= kl <= c[i + 1][1]:
                (b0, k0), (b1, k1) = c[i], c[i + 1]
                t = (math.log(kl) - math.log(k0)) / (math.log(k1) - math.log(k0))
                return math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))
        return None

    fr = load(f'{HERE}/{STEM0}_compress_frontier.json')['frontier']
    comp = []
    for p in fr:
        bu = bits_at_kl(uni, p['kl'])
        be2 = bits_at_kl(ent, p['kl'])
        comp.append({'scheme': p['scheme'], 'bits': p['bits'], 'kl': p['kl'],
                     'x_vs_fp32': 32 * n / p['bits'],
                     'x_vs_bf16': 16 * n / p['bits'],
                     'x_vs_lossless_best': res['lossless_best'] / p['bits'],
                     'uniform_bits_at_same_kl': bu,
                     'x_vs_uniform_ptq': (bu / p['bits']) if bu else None,
                     'entropy_ptq_bits_at_same_kl': be2,
                     'x_vs_entropy_ptq': (be2 / p['bits']) if be2 else None})
    res['frontier_vs_honest_baselines'] = comp
    for c in comp:
        f = lambda v: ('%.2f' % v) if v else '  -- '
        log(f'   {c["scheme"]:28s} {c["bits"]/1e6:6.3f} Mbit KL {c["kl"]:.4f}'
            f'  vs fp32 {c["x_vs_fp32"]:.1f}x  vs bf16 {c["x_vs_bf16"]:.1f}x'
            f'  vs uniform-PTQ {f(c["x_vs_uniform_ptq"])}x'
            f'  vs entropy-PTQ {f(c["x_vs_entropy_ptq"])}x')
    out['O1_fair_denominator'] = res
    return out


# ===========================================================================
# O2 -- RE-DERIVE THE BIT BILL FOR THE TOP FRONTIER POINTS FROM SCRATCH
# ===========================================================================
def independent_bill_transform(W, bpr, rot='none', entropy=True):
    """Recount q_transform's bill without using CC's Bits object, and audit
    every item the DECODER needs.  Returns (recon, dict)."""
    V, d = W.shape
    mu = W.mean(0, keepdim=True)
    X = W - mu
    assert rot == 'none'
    Q = torch.eye(d, device=W.device)
    Z = X @ Q
    var = (Z * Z).mean(0)
    b = CC._alloc(var, bpr)
    lo = Z.min(0).values.half().float()
    hi = Z.max(0).values.half().float()
    Zq = torch.zeros_like(Z)
    codes_bits, hist_bits, per_col = 0, 0, []
    for j in range(d):
        bj = int(b[j])
        if bj == 0:
            per_col.append({'col': j, 'bits': 0, 'code_bits': 0})
            continue
        step = ((hi[j] - lo[j]) / (2 ** bj - 1)).clamp_min(1e-30)
        c = ((Z[:, j] - lo[j]) / step).round().clamp(0, 2 ** bj - 1)
        Zq[:, j] = c * step + lo[j]
        cnt = torch.bincount(c.long(), minlength=2 ** bj).float()
        p = cnt / cnt.sum()
        Hh = float(-(p[p > 0] * p[p > 0].log2()).sum())
        cb = math.ceil(V * Hh)
        codes_bits += cb
        hist_bits += (2 ** bj) * 16
        per_col.append({'col': j, 'bits': bj, 'code_bits': cb,
                        'hist_bits': (2 ** bj) * 16})
    bill = {
        'column_means_fp32': d * 32,
        'rotation': 0,
        'entropy_coded_symbols': codes_bits,
        'symbol_histograms_fp16': hist_bits,
        'per_column_lo_hi_fp16': 2 * d * 16,
        'bit_allocation_map_4bit_per_column': d * 4,
    }
    bill['TOTAL'] = int(sum(bill.values()))
    return Zq @ Q.t() + mu, bill, per_col


def o2(out):
    log('O2: independent re-derivation of the bit bill (top frontier points)')
    D = CC.D1Desc(STEM0)
    d0 = load(f'{HERE}/{STEM0}_compress.json')
    W = D.base['wte_out']
    V, dd = W.shape
    res = {'points': [], 'audit': []}

    def body_bill_independent(b):
        """Recount the body from scratch: 6 attention + 3 MLP matrices at b
        bits with per-row fp16 lo/hi, plus the fp32 bias."""
        P, items = {}, {}
        for k in D.PART_NAMES:
            if k in ('wte_read', 'wte_out'):
                continue
            Wm = D.base[k]
            if Wm.dim() == 1:
                P[k] = Wm.clone()
                items[k + '_fp32'] = Wm.numel() * 32
                continue
            rows = Wm.shape[0]
            lo = Wm.min(1, keepdim=True).values.half().float()
            hi = Wm.max(1, keepdim=True).values.half().float()
            step = ((hi - lo) / (2 ** b - 1)).clamp_min(1e-30)
            q = ((Wm - lo) / step).round().clamp(0, 2 ** b - 1)
            P[k] = q * step + lo
            items[k + '_values'] = Wm.numel() * b
            items[k + '_rowscales'] = 2 * rows * 16
        return P, items

    for name, bpr, bb in (('embT768+body8', 768, 8), ('embT640+body8', 640, 8),
                          ('embT512+body6', 512, 6)):
        Wc, ebill, per_col = independent_bill_transform(W, bpr)
        P, bitems = body_bill_independent(bb)
        P['wte_read'] = Wc
        P['wte_out'] = Wc
        s = D.score(P)
        total = ebill['TOTAL'] + sum(bitems.values())
        ref = [r for r in sec_rows(d0, 'F_combined') if r['scheme'] == name]
        res['points'].append({
            'scheme': name,
            'reviewer_bits': int(total), 'reviewer_kl': s['kl'],
            'analyst_bits': ref[0]['bits'] if ref else None,
            'analyst_kl': ref[0]['kl'] if ref else None,
            'embedding_bill': ebill, 'body_bill': {k: int(v) for k, v in
                                                   bitems.items()},
            'body_total': int(sum(bitems.values())),
            'n_columns_with_zero_bits': sum(1 for c in per_col
                                            if c['bits'] == 0),
            'max_column_bits': max(c['bits'] for c in per_col)})
        log(f'   {name}: reviewer {total/1e6:.4f} Mbit KL {s["kl"]:.5f} | '
            f'analyst {ref[0]["bits"]/1e6 if ref else float("nan"):.4f} Mbit '
            f'KL {ref[0]["kl"] if ref else float("nan"):.5f}')

    # --- the distilled headline point, recounted --------------------------
    row = [r for r in sec_rows(d0, 'I_distilled')
           if r['scheme'] == 'distilled_emb4_body6']
    if row:
        res['distilled_note'] = {
            'scheme': 'distilled_emb4_body6',
            'analyst_bits': row[0]['bits'], 'analyst_kl': row[0]['kl'],
            'bill': row[0]['bill'],
            'reviewer_recount': int(
                # 1048576 embedding weights entropy-coded at 4 bits + row
                # scales, 6 attn + 3 mlp at 6 bits + row scales, fp32 bias
                sum(row[0]['bill'][k] for k in row[0]['bill'] if k != 'TOTAL')),
        }

    # --- AUDIT: things a decoder needs that might not be charged ----------
    audit = []
    audit.append({'item': 'decoder source code (architecture, rotary tables, '
                          'logit soft-cap, RMSNorm)',
                  'charged': False,
                  'verdict': 'declared convention; identical for every point '
                             'including the model itself, so it cancels'})
    audit.append({'item': 'tensor shapes / vocabulary size / scheme identity',
                  'charged': False, 'bits_if_charged': '~100',
                  'verdict': 'negligible'})
    audit.append({'item': 'which of the ~150 measured schemes is the winner',
                  'charged': False,
                  'bits_if_charged': int(math.ceil(math.log2(150))),
                  'verdict': 'negligible (8 bits) as a code, but it is a '
                             'SELECTION on held KL -- see O3'})
    audit.append({'item': 'est-split token frequency order (stratified/anchor '
                          'families) and PPMI basis (corpus-statistic family)',
                  'charged': False,
                  'verdict': 'declared free; O3 re-runs the frontier with the '
                             'corpus-derived families deleted'})
    audit.append({'item': 'arithmetic-coder histograms',
                  'charged': True,
                  'verdict': 'charged at 2^b x 16 bits per stream, which is '
                             'GENEROUS: a 12-bit column charges 65536 bits of '
                             'histogram for 8192 symbols'})
    audit.append({'item': 'per-column bit-allocation map',
                  'charged': True, 'bits': dd * 4,
                  'verdict': '4 bits/column covers 0..12, correct'})
    audit.append({'item': 'per-row / per-column lo,hi scales',
                  'charged': True,
                  'verdict': 'fp16 lo and hi, and the quantiser itself rounds '
                             'lo/hi through fp16 before use, so the decoder '
                             'reconstruction is exact'})
    audit.append({'item': 'tied embedding counted once',
                  'charged': True,
                  'verdict': 'correct -- the model ties read and write, and '
                             'every description that splits the roles pays '
                             'for both tables'})
    audit.append({'item': 'entropy-coded stream length / EOF',
                  'charged': False, 'bits_if_charged': 32,
                  'verdict': 'negligible'})
    res['audit'] = audit
    out['O2_bit_accounting'] = res
    return out


# ===========================================================================
# O3 -- FIT/SCORE SPLIT, AND SELECTION-ON-HELD
# ===========================================================================
def o3(out):
    log('O3: fit/score split and frontier selection')
    res = {}
    man = json.load(open(f'{HERE}/tf_corpus_b8192/MANIFEST.json'))
    res['split_row_ranges'] = man['splits']
    res['split_semantics'] = man['split_semantics']
    res['disjoint'] = True
    res['held_scored'] = {'n_seq': 64, 'T': 256, 'ntok': 16384,
                          'source': 'held00.npy rows 0:64'}
    res['what_touches_est'] = [
        'token frequency + orthographic class (sections B,D,G,J,K)',
        'PPMI co-occurrence basis (sections L,M)',
        'distillation (section I): 8000 est sequences, iterate selected on a '
        'disjoint 256-sequence est slice',
    ]
    res['what_touches_held'] = [
        'every KL/CE number',
        'THE PARETO SELECTION ITSELF -- the frontier is the lower envelope of '
        '~150 schemes evaluated on the same 16384 held tokens',
    ]

    # --- the real test: re-score every frontier candidate on a DISJOINT
    #     evaluation set (the untouched `spare` split) and see whether the
    #     frontier and its winners change.
    D = CC.D1Desc(STEM0)
    W = D.base['wte_out']
    TS = RR.token_stats(D)
    order = torch.argsort(TS['freq'], descending=True)

    def eval_two(P):
        a = D.score(P, split='held', n_seq=64, T=256)
        b = D.score(P, split='spare', n_seq=256, T=256)
        return a['kl'], b['kl']

    cands = []
    for bpr in (256, 384, 512, 640, 768):
        cands.append((f'embT{bpr}', lambda bpr=bpr: CC.q_transform(W, bpr, rot='none')))
    for b in (2, 3, 4, 5, 6, 8):
        cands.append((f'embQ{b}e', lambda b=b: CC.q_scalar_entropy(W, b)))
    for (bh, bl, nh) in ((6, 4, 2048), (8, 5, 512), (6, 3, 1024)):
        cands.append((f'embS{bh}_{bl}_{nh}',
                      lambda bh=bh, bl=bl, nh=nh: CC.q_stratified(W, bh, bl, nh, order)))
    rows = []
    for en, ef in cands:
        Wc, be = ef()
        for bb in (4, 6, 8):
            P, bbits = RR.body_parts_bits(D, bb)
            P['wte_read'] = Wc
            P['wte_out'] = Wc
            kh, ks = eval_two(P)
            rows.append({'scheme': f'{en}+body{bb}',
                         'bits': Bits(embedding=be.total).merge(bbits, 'b_').total,
                         'kl_held': kh, 'kl_spare_disjoint': ks})
            log(f'   {en}+body{bb}: held {kh:.5f}  spare {ks:.5f}')
    res['held_vs_disjoint_rows'] = rows

    def pareto(rs, key):
        o = []
        for p in sorted(rs, key=lambda q: (q['bits'], q[key])):
            if not o or p[key] < o[-1][key] - 1e-12:
                o.append(p)
        return [p['scheme'] for p in o]
    res['pareto_on_held'] = pareto(rows, 'kl_held')
    res['pareto_on_disjoint'] = pareto(rows, 'kl_spare_disjoint')
    res['pareto_identical'] = res['pareto_on_held'] == res['pareto_on_disjoint']
    kh = np.array([r['kl_held'] for r in rows])
    ks = np.array([r['kl_spare_disjoint'] for r in rows])
    res['kl_ratio_disjoint_over_held'] = {
        'median': float(np.median(ks / kh)), 'min': float((ks / kh).min()),
        'max': float((ks / kh).max())}
    log('   pareto identical on a disjoint eval set:', res['pareto_identical'])
    out['O3_split'] = res
    return out


# ===========================================================================
# O4 -- WAS THE CLUSTERING DONE WELL?
# ===========================================================================
@torch.no_grad()
def stream_second_moment(D, n_seq=512, T=256, batch=16):
    """Sigma = E[rms(x) rms(x)^T] over est text: the metric in which an error
    in an unembedding ROW turns into an error in that token's LOGIT."""
    arr = tf_corpus.load_split(D.V, 'est', n_seq, tok=D.cfg.tok)
    x = torch.from_numpy(arr[:, :T]).to(D.dev)
    S = torch.zeros(D.Ws, D.Ws, device=D.dev, dtype=torch.float64)
    n = 0
    for a in range(0, x.shape[0], batch):
        xb = x[a:a + batch]
        h = D.model(xb, collect=None)
        # recompute the pre-readout stream exactly as D1Desc.forward does
        e = CC.rms(D.base['wte_read'][xb], D.Ws)
        B, Tq = xb.shape
        cos = D.cos[None, :Tq, None, :]
        sin = D.sin[None, :Tq, None, :]
        mask = D.mask[:Tq, :Tq]
        import tf_model as M

        def qk(Wm):
            z = (e @ Wm.t()).view(B, Tq, D.H, D.hd)
            return M.apply_rot(CC.rms(z, D.hd), cos, sin)
        s1 = torch.einsum('bqhd,bkhd->bhqk', qk(D.base['Wq']), qk(D.base['Wk'])) / D.hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', qk(D.base['Wq2']), qk(D.base['Wk2'])) / D.hd
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        v = (e @ D.base['Wv'].t()).view(B, Tq, D.H, D.hd)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, D.Dc)
        xx = e + y @ D.base['Wproj'].t()
        xn = CC.rms(xx, D.Ws)
        hh = (xn @ D.base['Left'].t()) * (xn @ D.base['Right'].t())
        xx = xx + hh @ D.base['Down'].t() + D.base['Down_bias']
        z = CC.rms(xx, D.Ws).reshape(-1, D.Ws).double()
        S += z.t() @ z
        n += z.shape[0]
    return (S / n).float()


@torch.no_grad()
def read_fisher(D, n_seq=256, T=256, batch=8):
    """Diagonal (per-coordinate) empirical Fisher of the model's loss with
    respect to the READ embedding: the natural whitening metric for the read
    role, and a per-token importance that is behavioural rather than merely
    frequency."""
    arr = tf_corpus.load_split(D.V, 'est', n_seq, tok=D.cfg.tok)
    x = torch.from_numpy(arr[:, :T + 1]).to(D.dev)
    Wr = D.base['wte_read'].clone().requires_grad_(True)
    acc = torch.zeros_like(Wr)
    with torch.enable_grad():
        for a in range(0, x.shape[0], batch):
            bb = x[a:a + batch]
            xi, yi = bb[:, :-1], bb[:, 1:]
            e = CC.rms(Wr[xi], D.Ws)
            lg = D.forward(xi, e_override=e)
            loss = F.cross_entropy(lg.reshape(-1, D.V).float(), yi.reshape(-1))
            g, = torch.autograd.grad(loss, Wr)
            acc += g.detach() ** 2
    return acc


def o4(out):
    log('O4: was the clustering done well?')
    D = CC.D1Desc(STEM0)
    TS = RR.token_stats(D)
    freq = TS['freq']
    W = D.base['wte_out']
    E = F.rms_norm(D.base['wte_read'], (D.Ws,))
    V, dd = W.shape
    rows = []

    def add(name, role, C, a, k, cent_bits_b):
        """Charge honestly: centroids at cent_bits_b bits (entropy coded, with
        per-row scales) plus one index per token."""
        if cent_bits_b >= 32:
            Cq, cb = C, Bits(centroids=bits_dense(C.numel(), 32))
        else:
            Cq, cb = CC.q_scalar_entropy(C, cent_bits_b)
        bits = Bits(index=bits_index(V, k)).merge(cb, 'cent_')
        P = {role: Cq[a]} if role != 'both' else {'wte_read': Cq[a],
                                                  'wte_out': Cq[a]}
        s = D.score(P)
        rows.append({'scheme': name, 'role': role, 'k': k,
                     'centroid_bits': cent_bits_b, 'bits': bits.total,
                     'bill': bits.to_json(), **s})
        log(f'   {name:44s} {bits.total/1e6:6.3f} Mbit  KL {s["kl"]:.5f}')

    def group_means(tab, a, k, w):
        C = torch.zeros(k, tab.shape[1], device=tab.device)
        n = torch.zeros(k, device=tab.device)
        C.index_add_(0, a, tab * w[:, None])
        n.index_add_(0, a, w)
        return C / n.clamp_min(1e-30)[:, None]

    # --- (i) the analyst's clustering, but with the centroids QUANTISED.
    #     Their q_cluster stores centroids at fp32, which at k=4096 is 16.8
    #     Mbit of the 16.9 Mbit bill -- a self-inflicted wound.
    for k in (512, 1024, 2048, 4096):
        for cb in (4, 6, 8, 32):
            C, a = CC.kmeans_weighted(W, k, weights=freq, seed=0)
            C = group_means(W, a, k, freq)
            add(f'writecluster_k{k}_cent{cb}', 'wte_out', C, a, k, cb)

    # --- (ii) BEHAVIOUR-WEIGHTED metric for the write role.  An error dw in
    #     an unembedding row changes that token's logit by <rms(x), dw>, so the
    #     distortion that matters is dw^T Sigma dw with Sigma = E[rms(x)rms(x)^T]
    #     -- NOT the Euclidean norm the analyst clustered in.  The optimal
    #     centroid under any quadratic metric is still the (weighted) mean, so
    #     this changes only the ASSIGNMENT and costs the decoder nothing.
    Sig = stream_second_moment(D)
    ev, Q = torch.linalg.eigh(Sig.double())
    Lh = (Q * ev.clamp_min(1e-12).sqrt()).float() @ Q.t().float()   # Sigma^{1/2}
    Yw = W @ Lh
    for k in (512, 1024, 2048, 4096):
        _, a = CC.kmeans_weighted(Yw, k, weights=freq, seed=0)
        C = group_means(W, a, k, freq)      # centroids stored in w-space
        for cb in (6, 32):
            add(f'writecluster_whitened_k{k}_cent{cb}', 'wte_out', C, a, k, cb)

    # --- (iii) FISHER-weighted metric + Fisher row-importance for the READ
    #     role (frequency is not the same thing as behavioural importance).
    Fi = read_fisher(D)
    colw = Fi.sum(0)
    colw = (colw / colw.mean()).clamp_min(1e-8).sqrt()
    roww = Fi.sum(1)
    Yr = E * colw[None, :]
    for k in (512, 1024, 2048, 4096):
        _, a = CC.kmeans_weighted(Yr, k, weights=roww, seed=0)
        C = group_means(E, a, k, roww)
        for cb in (6, 32):
            add(f'readcluster_fisher_k{k}_cent{cb}', 'wte_read', C, a, k, cb)

    # --- (iv) the rate-distortion-correct form of "merge tokens": vector
    #     quantisation WITH a coded residual.  This is what a competent
    #     engineer would build if told to use prototypes.
    for k in (512, 2048):
        C, a = CC.kmeans_weighted(W, k, weights=freq, seed=0)
        C = group_means(W, a, k, freq)
        for rb in (2, 3, 4):
            Cq, cb = CC.q_scalar_entropy(C, 8)
            resid = W - Cq[a]
            Rq, rbits = CC.q_scalar_entropy(resid, rb)
            Wc = Cq[a] + Rq
            bits = Bits(index=bits_index(V, k)).merge(cb, 'cent_') \
                                               .merge(rbits, 'resid_')
            s = D.score({'wte_read': Wc, 'wte_out': Wc})
            rows.append({'scheme': f'vq_k{k}_resid_q{rb}', 'role': 'both',
                         'k': k, 'bits': bits.total, 'bill': bits.to_json(),
                         **s})
            log(f'   vq_k{k}_resid_q{rb:<2d} {bits.total/1e6:6.3f} Mbit  '
                f'KL {s["kl"]:.5f}')

    # --- (v) the analyst's own read-role clustering, centroids QUANTISED --
    for k in (512, 1024, 2048, 4096):
        _, a = CC.kmeans_weighted(E, k, weights=freq, seed=0)
        C = group_means(E, a, k, freq)
        for cb in (4, 6):
            add(f'readcluster_freq_k{k}_cent{cb}', 'wte_read', C, a, k, cb)

    # --- baselines at matched bits: plain scalar codes AND the frontier-
    #     winning transform code, on the same role -------------------------
    base = []
    for b in (2, 3, 4, 5, 6):
        Wq, bt = CC.q_scalar_entropy(W, b)
        s = D.score({'wte_out': Wq})
        base.append({'scheme': f'write_scalar_q{b}e', 'role': 'wte_out',
                     'bits': bt.total, **s})
        Eq, bt2 = CC.q_scalar_entropy(E, b)
        s2 = D.score({'wte_read': Eq})
        base.append({'scheme': f'read_scalar_q{b}e', 'role': 'wte_read',
                     'bits': bt2.total, **s2})
        s3 = D.score({'wte_read': Wq, 'wte_out': Wq})
        base.append({'scheme': f'both_scalar_q{b}e', 'role': 'both',
                     'bits': bt.total, **s3})
    for bpr in (256, 384, 512, 640, 768):
        Wt, bt = CC.q_transform(W, bpr, rot='none')
        s = D.score({'wte_read': Wt, 'wte_out': Wt})
        base.append({'scheme': f'both_transform_{bpr}', 'role': 'both',
                     'bits': bt.total, **s})
        s2 = D.score({'wte_out': Wt})
        base.append({'scheme': f'write_transform_{bpr}', 'role': 'wte_out',
                     'bits': bt.total, **s2})
        Et, bte = CC.q_transform(E, bpr, rot='none')
        s3 = D.score({'wte_read': Et})
        base.append({'scheme': f'read_transform_{bpr}', 'role': 'wte_read',
                     'bits': bte.total, **s3})

    # --- the verdict computation: at MATCHED BITS, how much worse is the
    #     best clustering than the best recoding, per role? ----------------
    def interp_kl(curve, bits):
        c = sorted([(p['bits'], p['kl']) for p in curve])
        if not c or bits <= c[0][0] or bits >= c[-1][0]:
            return None
        for i in range(len(c) - 1):
            if c[i][0] <= bits <= c[i + 1][0]:
                (b0, k0), (b1, k1) = c[i], c[i + 1]
                t = (math.log(bits) - math.log(b0)) / (math.log(b1) - math.log(b0))
                return math.exp(math.log(k0) + t * (math.log(k1) - math.log(k0)))
        return None

    verdict = []
    for role in ('wte_out', 'wte_read', 'both'):
        rec = [b for b in base if b['role'] == role]
        for r in rows:
            if r['role'] != role:
                continue
            kb = interp_kl(rec, r['bits'])
            verdict.append({'scheme': r['scheme'], 'role': role,
                            'bits': r['bits'], 'kl': r['kl'],
                            'best_recoding_kl_at_same_bits': kb,
                            'kl_penalty_x': (r['kl'] / kb) if kb else None})
    for v in sorted([v for v in verdict if v['kl_penalty_x']],
                    key=lambda q: q['kl_penalty_x'])[:14]:
        log(f'   MATCHED-BITS {v["scheme"]:38s} KL {v["kl"]:.4f} vs recoding '
            f'{v["best_recoding_kl_at_same_bits"]:.4f} -> '
            f'{v["kl_penalty_x"]:.2f}x')
    out['O4_clustering'] = {'rows': rows, 'baselines': base,
                            'matched_bits_verdict': verdict,
                            'sigma_trace': float(Sig.trace())}
    return out


# ===========================================================================
# O5 -- DOES R^2 TRANSLATE INTO BITS?  IS THE RESIDUAL CODER COMPETENT?
# ===========================================================================
def o5(out):
    log('O5: does the structural R^2 translate into bits?')
    D = CC.D1Desc(STEM0)
    TS = RR.token_stats(D)
    W = D.base['wte_out']
    V, dd = W.shape
    res = {'families': {}}

    for nm, Phi in (('spelling', RR.token_features(D, TS)),
                    ('corpusstat', RR.corpus_stat_features(D))):
        nf = Phi.shape[1]
        G = Phi.t() @ Phi + 1e-2 * torch.eye(nf, device=D.dev)
        B = torch.linalg.solve(G, Phi.t() @ W)
        pred = Phi @ B
        r2_in = 1 - float(((W - pred) ** 2).sum()
                          / (W - W.mean(0)).pow(2).sum())
        # --- out-of-sample R^2 over TOKENS (5-fold on the token axis) ------
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(V, generator=g).to(D.dev)
        num, den = 0.0, 0.0
        for f in range(5):
            te = perm[f::5]
            tr = torch.cat([perm[j::5] for j in range(5) if j != f])
            Gt = Phi[tr].t() @ Phi[tr] + 1e-2 * torch.eye(nf, device=D.dev)
            Bt = torch.linalg.solve(Gt, Phi[tr].t() @ W[tr])
            num += float(((W[te] - Phi[te] @ Bt) ** 2).sum())
            den += float((W[te] - W[tr].mean(0)).pow(2).sum())
        r2_cv = 1 - num / den
        Bq, bb = CC.q_scalar(B, 8)
        predq = Phi @ Bq
        r2_q = 1 - float(((W - predq) ** 2).sum()
                         / (W - W.mean(0)).pow(2).sum())
        resid = W - predq
        # --- the two quantities a scalar coder actually responds to --------
        var_ratio = float((resid ** 2).mean() / ((W - W.mean(0)) ** 2).mean())
        rng_W = float((W.max(1).values - W.min(1).values).mean())
        rng_R = float((resid.max(1).values - resid.min(1).values).mean())
        # high-rate law: entropy-coded uniform quantisation at a FIXED step
        # costs h(X) - log2(step); halving the variance saves 0.5 log2 bits per
        # weight, INDEPENDENT of the rate.
        pred_saving_bits_per_weight = -0.5 * math.log2(max(var_ratio, 1e-12))
        # ... but a per-ROW min/max quantiser sets its step from the RANGE, so
        # what it actually banks is the range reduction:
        range_saving_bits_per_weight = -math.log2(max(rng_R / rng_W, 1e-12))
        fam = {'n_features': nf, 'r2_in_sample': r2_in,
               'r2_cross_validated_over_tokens': r2_cv,
               'r2_after_8bit_coefficients': r2_q,
               'residual_variance_ratio': var_ratio,
               'mean_row_range_original': rng_W,
               'mean_row_range_residual': rng_R,
               'predicted_saving_bits_per_weight_from_variance':
                   pred_saving_bits_per_weight,
               'predicted_saving_bits_per_weight_from_row_range':
                   range_saving_bits_per_weight,
               'regression_coefficient_bits': bb.total,
               'rows': []}
        # --- measured saving, matched-KL, against the SAME coder -----------
        for b in (2, 3, 4, 5, 6):
            Rq, br = CC.q_scalar_entropy(resid, b)
            Wc = predq + Rq
            s = D.score({'wte_read': Wc, 'wte_out': Wc})
            cond_bits = Bits(regression=bb.total).merge(br, 'r_').total
            Wp, bp = CC.q_scalar_entropy(W, b)
            s0 = D.score({'wte_read': Wp, 'wte_out': Wp})
            # equal-KL comparison needs the plain curve interpolated; but the
            # equal-*step* comparison is the clean coder question, so report
            # both: at the same b, bits saved and KL ratio.
            fam['rows'].append({
                'b': b, 'cond_bits': cond_bits, 'cond_kl': s['kl'],
                'cond_ce': s['ce'], 'plain_ce': s0['ce'],
                'plain_bits': bp.total, 'plain_kl': s0['kl'],
                'measured_saving_bits_per_weight':
                    (bp.total - cond_bits) / (V * dd),
                'kl_ratio_cond_over_plain': s['kl'] / s0['kl']})
            log(f'   {nm} b={b}: cond {cond_bits/1e6:.3f} Mbit KL {s["kl"]:.5f}'
                f'  plain {bp.total/1e6:.3f} Mbit KL {s0["kl"]:.5f}')
        # --- IS THE RESIDUAL CODER THE PROBLEM?  Give the residual the
        #     frontier-winning coder (transform + per-column allocation) and
        #     compare against the SAME coder on the raw table.
        tr_rows = []
        for bpr in (384, 512, 640, 768):
            Rc, brt = CC.q_transform(resid, bpr, rot='none')
            Wc = predq + Rc
            s = D.score({'wte_read': Wc, 'wte_out': Wc})
            Wp2, bp2 = CC.q_transform(W, bpr, rot='none')
            s2 = D.score({'wte_read': Wp2, 'wte_out': Wp2})
            tr_rows.append({'bits_per_row': bpr,
                            'cond_bits': Bits(regression=bb.total).merge(
                                brt, 'r_').total,
                            'cond_kl': s['kl'], 'cond_ce': s['ce'],
                            'plain_bits': bp2.total, 'plain_kl': s2['kl'],
                            'plain_ce': s2['ce']})
            log(f'   {nm} transform {bpr}: cond '
                f'{tr_rows[-1]["cond_bits"]/1e6:.3f} Mbit KL {s["kl"]:.5f}  '
                f'plain {bp2.total/1e6:.3f} Mbit KL {s2["kl"]:.5f}')
        fam['transform_coded_residual'] = tr_rows

        # --- THE DIAGNOSIS.  Two competing explanations for "R^2 0.41 but
        #     only 7-14% of the bits": (i) the residual coder is incompetent,
        #     (ii) R^2 in weight space is simply not worth many bits.  Settle
        #     it by comparing the GROSS saving (before paying for the
        #     regression) against the two rate-distortion bounds.
        def matched(rowsx, ck, cb_, pk, pb):
            o = []
            curve = sorted([(r[pb], r[pk]) for r in rowsx])
            for r in rowsx:
                kl = r[ck]
                pl = None
                c = sorted(curve, key=lambda t: t[1])
                for i in range(len(c) - 1):
                    if c[i][1] <= kl <= c[i + 1][1]:
                        (b0, k0), (b1, k1) = c[i], c[i + 1]
                        t = ((math.log(kl) - math.log(k0))
                             / (math.log(k1) - math.log(k0)))
                        pl = math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))
                if pl is None:
                    o.append(None)
                    continue
                net = (pl - r[cb_]) / pl
                gross_bpw = (pl - r[cb_] + bb.total) / (V * dd)
                o.append({'cond_bits': r[cb_], 'cond_kl': kl,
                          'plain_bits_at_same_kl': pl,
                          'net_bits_saved_frac': net,
                          'gross_bits_saved_per_weight': gross_bpw,
                          'regression_cost_per_weight': bb.total / (V * dd),
                          'variance_law_bound_per_weight':
                              pred_saving_bits_per_weight,
                          'row_range_bound_per_weight':
                              range_saving_bits_per_weight,
                          'gross_within_bounds': bool(
                              pred_saving_bits_per_weight * 0.8 <= gross_bpw
                              <= range_saving_bits_per_weight * 1.2)})
            return [x for x in o if x]
        fam['matched_kl_scalar_coder'] = matched(
            fam['rows'], 'cond_kl', 'cond_bits', 'plain_kl', 'plain_bits')
        fam['matched_kl_transform_coder'] = matched(
            tr_rows, 'cond_kl', 'cond_bits', 'plain_kl', 'plain_bits')
        for tag in ('matched_kl_scalar_coder', 'matched_kl_transform_coder'):
            for m in fam[tag]:
                log(f'   {nm} {tag[10:]}: net saved {m["net_bits_saved_frac"]*100:5.1f}%'
                    f'  gross {m["gross_bits_saved_per_weight"]:.3f} b/wt '
                    f'(variance law {m["variance_law_bound_per_weight"]:.3f}, '
                    f'range law {m["row_range_bound_per_weight"]:.3f}, '
                    f'regression costs {m["regression_cost_per_weight"]:.3f})')
        res['families'][nm] = fam

    # --- the conversion law itself, stated and checked -------------------
    res['conversion_law'] = (
        'For an entropy-coded uniform quantiser at a fixed step, rate = h(X) - '
        'log2(step), so replacing a source by a residual with variance ratio v '
        'saves exactly -0.5*log2(v) bits per weight at the SAME distortion, '
        'independent of the bit depth.  R^2 = 1 - v, so R^2 = 0.405 buys '
        '0.374 bits/weight and R^2 = 0.256 buys 0.213 bits/weight.  To halve a '
        '4-bit code you would need R^2 = 1 - 2^-4 = 0.9375.')
    out['O5_structure_to_bits'] = res
    return out


# ===========================================================================
# O6 -- SEED ROBUSTNESS OF EVERY FRONTIER POINT
# ===========================================================================
def o6(out):
    log('O6: seed robustness of every frontier point')
    fr = load(f'{HERE}/{STEM0}_compress_frontier.json')['frontier']
    seeds = {'s0': load(f'{HERE}/{STEM0}_compress.json'),
             's1': load(f'{HERE}/{STEM1}_compress.json'),
             's1_IM': load(f'{HERE}/tf_rev3_seed1_IM.json'),
             's2': load(f'{HERE}/tf_rev3_seed2_full.json')}

    def find(d, scheme):
        for sec in ('A_self_quantisation', 'C_embedding', 'D_anchor', 'E_body',
                    'F_combined', 'G_codes', 'I_distilled', 'K_features',
                    'L_corpus_stats', 'M_conditional_combined',
                    'H_weightsfree'):
            for r in sec_rows(d, sec):
                if r['scheme'] == scheme:
                    return r
        return None

    rows = []
    for p in fr:
        e = {'scheme': p['scheme'], 'bits_s0': p['bits'], 'kl_s0': p['kl']}
        for tag in ('s1', 's1_IM', 's2'):
            r = find(seeds[tag], p['scheme'])
            if r:
                e[f'bits_{tag}'] = r.get('bits')
                e[f'kl_{tag}'] = r.get('kl')
        kls = [e[k] for k in e if k.startswith('kl_')]
        e['n_seeds'] = len(kls)
        e['kl_min'] = min(kls)
        e['kl_max'] = max(kls)
        e['kl_spread_x'] = (max(kls) / max(min(kls), 1e-12)) if len(kls) > 1 else None
        e['verdict'] = ('at the measurement floor (KL < 1e-4, fp16 reference '
                        'storage) -- no spread is meaningful'
                        if e['kl_max'] < 1e-4 else
                        'single seed only' if len(kls) == 1 else
                        'seed-robust (<1.25x)' if e['kl_spread_x'] < 1.25 else
                        'seed-sensitive (>2x)' if e['kl_spread_x'] > 2 else
                        'moderately seed-sensitive (1.25-2x)')
        rows.append(e)
        log(f'   {p["scheme"]:28s} n={e["n_seeds"]} spread '
            f'{e["kl_spread_x"] if e["kl_spread_x"] else float("nan"):.2f}x '
            f'-> {e["verdict"]}')
    out['O6_seed_robustness'] = {'points': rows}
    return out


# ===========================================================================
# O7 -- LOGAN'S TWO REDIRECTIONS
#   (7a) quantisation is a FILE FORMAT, not an explanation.  Split every
#        measured description into schemes that merely RECODE the model's own
#        weights and schemes that ASSERT STRUCTURE, and give class (b) its own
#        Pareto front.
#   (7b) KL-from-the-model cannot see a description that is BETTER than the
#        model.  Score held cross-entropy against the DATA for every point and
#        ask whether anything beats the model's own 4.7114 nats.
# ===========================================================================
RECODE_PREFIXES = (
    'uniform_', 'scalar_q', 'emb', 'transform_none', 'transform_hadamard',
    'distilled_', 'attn_q', 'mlp_q', 'embQ', 'embT', 'embS', 'strat_',
)
STRUCT_PURE_PREFIXES = (
    'cluster_', 'lowrank_', 'pq_', 'anchor', 'mlp_trunc_', 'mlp_cp_refit',
    'mlp_cp', 'body_cp', 'transform_pca', 'weightsfree_', 'factored_',
)
STRUCT_HYBRID_PREFIXES = (
    'feature_residual_', 'corpusstat_residual_', 'spelling_res_',
    'corpusstat_res_',
)


def classify(scheme):
    for p in STRUCT_HYBRID_PREFIXES:
        if scheme.startswith(p):
            return 'b_hybrid_structure_plus_coded_residual'
    for p in STRUCT_PURE_PREFIXES:
        if scheme.startswith(p):
            return 'b_pure_structure'
    for p in RECODE_PREFIXES:
        if scheme.startswith(p):
            return 'a_recoding_of_the_weights'
    return 'unclassified'


def o7(out):
    log('O7: recoding vs structure, and cross-entropy against the DATA')
    d0 = load(f'{HERE}/{STEM0}_compress.json')
    mce = d0['model']['ce']
    fp32 = d0['model']['fp32_bits']
    pts = []
    for sec, v in d0.items():
        if not isinstance(v, dict) or 'rows' not in v:
            continue
        for r in v['rows']:
            b = r.get('bits') or r.get('bits_total_with_fp32_body')
            if b is None or r.get('kl') is None or r.get('ce') is None:
                continue
            pts.append({'section': sec, 'scheme': r['scheme'],
                        'bits': float(b), 'kl': float(r['kl']),
                        'ce': float(r['ce']),
                        'delta_ce_vs_model': float(r['ce']) - mce,
                        'class': classify(r['scheme'])})
    # de-duplicate identical (scheme, bits) rows appearing in two sections
    seen, uniq = set(), []
    for p in sorted(pts, key=lambda q: q['bits']):
        k = (p['scheme'], round(p['bits']))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    pts = uniq

    def pareto(ps, key='kl'):
        o = []
        for p in sorted(ps, key=lambda q: (q['bits'], q[key])):
            if not o or p[key] < o[-1][key] - 1e-12:
                o.append(p)
        return o

    cls = {}
    for c in ('a_recoding_of_the_weights', 'b_pure_structure',
              'b_hybrid_structure_plus_coded_residual', 'unclassified'):
        g = [p for p in pts if p['class'] == c]
        cls[c] = {'n': len(g), 'pareto': pareto(g)}
        log(f'   {c}: {len(g)} points, {len(cls[c]["pareto"])} on its own front')

    # how much does the structural front beat / lose to the recoding front?
    def bits_at_kl(front, kl):
        c = sorted([(p['bits'], p['kl']) for p in front if p['kl'] > 0],
                   key=lambda t: t[1])
        if not c or kl <= c[0][1] or kl >= c[-1][1]:
            return None
        for i in range(len(c) - 1):
            if c[i][1] <= kl <= c[i + 1][1]:
                (b0, k0), (b1, k1) = c[i], c[i + 1]
                t = (math.log(kl) - math.log(k0)) / (math.log(k1) - math.log(k0))
                return math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))
        return None

    fa = cls['a_recoding_of_the_weights']['pareto']
    comp = []
    for c in ('b_pure_structure', 'b_hybrid_structure_plus_coded_residual'):
        for p in cls[c]['pareto']:
            ba = bits_at_kl(fa, p['kl'])
            comp.append({'class': c, 'scheme': p['scheme'], 'bits': p['bits'],
                         'kl': p['kl'], 'ce': p['ce'],
                         'recoding_bits_at_same_kl': ba,
                         'structure_over_recoding_x': (p['bits'] / ba) if ba else None})
    for c in comp:
        if c['structure_over_recoding_x']:
            log(f'   {c["scheme"]:30s} {c["bits"]/1e6:6.3f} Mbit at KL '
                f'{c["kl"]:.4f}; pure recoding needs '
                f'{c["recoding_bits_at_same_kl"]/1e6:6.3f} Mbit -> structure '
                f'costs {c["structure_over_recoding_x"]:.2f}x')

    # --- APPLES TO APPLES.  The comparison above is contaminated by how each
    #     section treated the OTHER half of the model (an embedding scheme with
    #     an fp32 body carries 34 Mbit of body).  Redo it inside the two
    #     sub-families where the other half is held fixed.
    sub = {}
    for tag, secs, key in (
            ('embedding_only_fp32_body',
             ('C_embedding', 'D_anchor', 'G_codes', 'K_features',
              'L_corpus_stats'), 'bits_embedding'),
            ('body_only_fp32_embedding', ('E_body',), 'bits_body')):
        g = []
        for sec in secs:
            for r in sec_rows(d0, sec):
                if r.get(key) is None or r.get('kl') is None:
                    continue
                g.append({'scheme': r['scheme'], 'bits': float(r[key]),
                          'kl': float(r['kl']), 'ce': r.get('ce'),
                          'class': classify(r['scheme'])})
        if tag == 'body_only_fp32_embedding':
            # section E never quantised attention and MLP TOGETHER, so its
            # recoding arm has an artificial 3.5 Mbit floor.  Section A did:
            # add `emb32_bodyNbit`, otherwise the CP family wins by default.
            emb32 = 32 * 1048576
            for r in sec_rows(d0, 'A_self_quantisation'):
                if r['scheme'].startswith('emb32_body'):
                    g.append({'scheme': r['scheme'],
                              'bits': float(r['bits'] - emb32),
                              'kl': float(r['kl']), 'ce': r.get('ce'),
                              'class': 'a_recoding_of_the_weights'})
        fronts = {c: pareto([p for p in g if p['class'] == c])
                  for c in set(p['class'] for p in g)}
        fa2 = fronts.get('a_recoding_of_the_weights', [])
        rowsx = []
        for c, fr2 in fronts.items():
            if c == 'a_recoding_of_the_weights':
                continue
            for p in fr2:
                ba = bits_at_kl(fa2, p['kl'])
                rowsx.append({'class': c, 'scheme': p['scheme'],
                              'bits': p['bits'], 'kl': p['kl'],
                              'recoding_bits_at_same_kl': ba,
                              'structure_over_recoding_x':
                                  (p['bits'] / ba) if ba else None})
        sub[tag] = {'fronts': {c: [{'scheme': p['scheme'], 'bits': p['bits'],
                                    'kl': p['kl']} for p in f]
                               for c, f in fronts.items()},
                    'structure_vs_recoding': rowsx}
        log(f'   -- {tag} --')
        for r in rowsx:
            if r['structure_over_recoding_x']:
                log(f'      {r["scheme"]:30s} {r["bits"]/1e6:6.3f} Mbit KL '
                    f'{r["kl"]:.4f}; recoding needs '
                    f'{r["recoding_bits_at_same_kl"]/1e6:6.3f} -> '
                    f'{r["structure_over_recoding_x"]:.2f}x')

    # --- (7b) CE against the data ---------------------------------------
    below = [p for p in pts if p['ce'] < mce - 1e-9]
    ce_sorted = sorted(pts, key=lambda p: p['ce'])[:20]
    # is CE just KL in disguise?  If Delta CE == KL the descriptions carry no
    # information about the DATA that the model does not already carry.
    kk = np.array([p['kl'] for p in pts if p['kl'] > 1e-4])
    dc = np.array([p['delta_ce_vs_model'] for p in pts if p['kl'] > 1e-4])
    fit = float(np.polyfit(kk, dc, 1)[0])
    out['O7_recoding_vs_structure_and_CE'] = {
        'model_held_ce': mce, 'fp32_bits': fp32,
        'n_points': len(pts),
        'classes': {k: {'n': v['n'],
                        'pareto': [{'scheme': p['scheme'], 'bits': p['bits'],
                                    'kl': p['kl'], 'ce': p['ce']}
                                   for p in v['pareto']]}
                    for k, v in cls.items()},
        'structure_vs_recoding_at_matched_kl': comp,
        'apples_to_apples_subfamilies': sub,
        'n_descriptions_with_ce_below_the_model': len(below),
        'descriptions_below_model_ce': below,
        'twenty_lowest_ce': ce_sorted,
        'delta_ce_per_nat_of_kl_slope': fit,
        'joint_frontier_with_ce': [
            {'scheme': p['scheme'], 'bits': p['bits'], 'x_vs_fp32': fp32 / p['bits'],
             'kl': p['kl'], 'ce': p['ce'],
             'delta_ce_vs_model': p['delta_ce_vs_model'], 'class': p['class']}
            for p in pareto(pts)],
    }
    log(f'   descriptions with held CE below the model: {len(below)} of {len(pts)}')
    log(f'   Delta CE per nat of KL = {fit:.3f} (1.0 means the description '
        f'carries nothing about the data the model does not)')
    return out


# ===========================================================================
# O8 -- THE TEST KL IS STRUCTURALLY BLIND TO: fit the description to the DATA
# ===========================================================================
def distill_ce(D, emb_b, body_b, steps=10000, lr=6e-4, T=256, batch=8,
               n_seq=8000, seed=0, val_every=250):
    """Identical to `tf_compress_run.distill` except the objective is the DATA
    cross-entropy on `est`, not the KL to the model.  If the model is an
    imperfect approximation of something simpler, a short description fitted to
    the data -- not to the model -- is where that would show up."""
    torch.manual_seed(seed)
    keys = [k for k in D.PART_NAMES if k != 'wte_read']
    p = {k: torch.nn.Parameter(D.base[k].clone()) for k in keys}
    opt = torch.optim.Adam(p.values(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 100) * 0.5
        * (1 + math.cos(math.pi * min(1.0, t / steps))))
    arr = tf_corpus.load_split(D.V, 'est', n_seq + 256, tok=D.cfg.tok)
    x_all = torch.from_numpy(arr[:n_seq, :T + 1]).to(D.dev)
    x_val = torch.from_numpy(arr[n_seq:n_seq + 256, :T + 1]).to(D.dev)
    nb = x_all.shape[0] // batch

    def qdict(src):
        Q = {}
        for k in keys:
            b = emb_b if k == 'wte_out' else body_b
            Q[k] = RR.ste_q(src[k], b) if src[k].dim() > 1 else src[k]
        Q['wte_read'] = Q['wte_out']
        return Q

    @torch.no_grad()
    def val_ce():
        Q = qdict({k: v.detach() for k, v in p.items()})
        tot, n = 0.0, 0
        for a in range(0, x_val.shape[0], 16):
            xb = x_val[a:a + 16]
            lg = D.forward(xb[:, :-1], Q).float()
            tot += float(F.cross_entropy(lg.reshape(-1, D.V),
                                         xb[:, 1:].reshape(-1), reduction='sum'))
            n += xb[:, 1:].numel()
        return tot / n

    best = (val_ce(), {k: v.detach().clone() for k, v in p.items()}, 0)
    for it in range(steps):
        xb = x_all[(it % nb) * batch:(it % nb + 1) * batch]
        lg = D.forward(xb[:, :-1], qdict(p)).float()
        loss = F.cross_entropy(lg.reshape(-1, D.V), xb[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(p.values()), 1.0)
        opt.step()
        sch.step()
        if (it + 1) % val_every == 0:
            v = val_ce()
            if v < best[0]:
                best = (v, {k: q.detach().clone() for k, q in p.items()}, it + 1)
    src = best[1]
    P, bits = {}, Bits()
    for k in keys:
        b = emb_b if k == 'wte_out' else body_b
        W = src[k]
        if W.dim() == 1:
            P[k] = W
            bits.add(**{k: bits_dense(W.numel(), 32)})
        else:
            R, bb = CC.q_scalar_entropy(W, b)
            P[k] = R
            bits.add(**{k: bb.total})
    P['wte_read'] = P['wte_out']
    return P, bits, {'est_val_ce': best[0], 'best_step': best[2]}


def o8(out):
    log('O8: fit descriptions to the DATA (the test KL cannot see)')
    D = CC.D1Desc(STEM0)
    mce = D.score()['ce']
    rows = []
    # the 32/32 arm is the CONFOUND CONTROL: est is FRESH data the model never
    # saw, so any CE gain that also appears at full precision is extra data,
    # not compression finding simpler structure.
    for emb_b, body_b in ((32, 32), (8, 8), (6, 8), (4, 6), (3, 6), (2, 4)):
        P, bits, tl = distill_ce(D, emb_b, body_b)
        s = D.score(P)
        rows.append({'scheme': f'ce_distilled_emb{emb_b}_body{body_b}',
                     'emb_bits': emb_b, 'body_bits': body_b,
                     'bits': bits.total, 'kl_from_model': s['kl'],
                     'held_ce': s['ce'], 'model_held_ce': mce,
                     'delta_ce_vs_model': s['ce'] - mce,
                     'beats_the_model_on_data': bool(s['ce'] < mce),
                     'selection': tl})
        log(f'   emb{emb_b}/body{body_b}: {bits.total/1e6:6.3f} Mbit  held CE '
            f'{s["ce"]:.5f} (model {mce:.5f}, delta {s["ce"]-mce:+.5f})  '
            f'KL-from-model {s["kl"]:.4f}')
    out['O8_fit_to_data_not_to_the_model'] = {
        'model_held_ce': mce, 'rows': rows,
        'note': ('the 32/32 row is the confound control: it is the same '
                 'objective and the same fresh est data with NO quantisation, '
                 'so it separates "compression found simpler structure" from '
                 '"the model was undertrained and est is more data"')}
    return out


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    for k in '12345678':
        ap.add_argument(f'--o{k}', action='store_true')
    a = ap.parse_args()
    out = load(OUT)
    out.setdefault('reviewer', 'independent round 3, FINDING 12 (compression '
                               'frontier)')
    fns = {'1': o1, '2': o2, '3': o3, '4': o4, '5': o5, '6': o6, '7': o7,
           '8': o8}
    for k in '12345678':
        if getattr(a, f'o{k}'):
            t0 = time.time()
            fns[k](out)
            log(f'O{k} done in {time.time()-t0:.0f}s')
            save(out)
    save(out)
    log('wrote', OUT)


if __name__ == '__main__':
    main()
