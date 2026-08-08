"""Driver for the RUNG-5 COMPRESSION FRONTIER.  Registered predictions are
written to the output JSON BEFORE any measurement runs (`--register`), then
each section appends its measurements.

Sections
  A  reference   the model as its own description: weight-quantisation frontier
  B  split       memorisation vs structure -- coarsening the READ role and the
                 WRITE role of the embedding, learned clusters vs token
                 FEATURES vs a random-grouping control
  C  embedding   scalar quant / low rank / product quantisation / clustering
  D  anchor      exact rows for the top-B tokens + compressed tail (the parent
                 program's frontier-dominating hybrid, ported)
  E  body        attention + MLP: unit truncation (control), refitted CP,
                 low rank in the FOLDED coordinates, quantisation
  F  combined    the best joint description, and a second cell for confirmation

Usage
  python tf_compress_run.py --sections ABCDEF
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import tf_compress as CC
import tf_corpus
import tf_interp as I1
from tf_compress import Bits, bits_dense, bits_index

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False

STEM = 'tf_vanilla_d1_w128_b8192_s0'
CONFIRM = 'tf_vanilla_d1_w128_b8192_s1'

REGISTERED = {
  "written_before_any_measurement": True,
  "P1_read_vs_write_asymmetry": (
    "The embedding's READ role is far more compressible than its WRITE role. "
    "Concretely: coarsening the read table to 512 learned clusters (a 16x cut "
    "on the token axis) will leave KL < 0.10 nats, while coarsening the WRITE "
    "table to 512 clusters will leave KL > 1.0 nats. Reason offered in "
    "advance: the write role must NAME the next token, which is irreducibly "
    "per-token information; the read role only has to steer a 128-dimensional "
    "computation."),
  "P2_features_lose_to_identity": (
    "At a matched number of groups, groupings built from token FEATURES "
    "(frequency band x orthographic class x length band) will recover less "
    "than 30% of the KL that learned behavioural clusters recover, i.e. the "
    "per-token content of the read role is NOT a function of cheap surface "
    "features. Both must beat a random grouping at the same group count."),
  "P3_embedding_dominates_the_bill": (
    "The best point on the frontier will be dominated by embedding "
    "compression: at the knee, the embedding will account for < 60% of the "
    "bits (down from 78% in the fp32 model) and the total will be at least 6x "
    "below the model's own 43.0 Mbit, at KL < 0.05."),
  "P4_body_resists": (
    "The MLP will resist structural compression: no low-rank or refitted-CP "
    "scheme will beat plain scalar quantisation of Left/Right/Down at matched "
    "bits, anywhere on the curve."),
  "P5_anchor_hybrid_wins": (
    "Ported from ../qk_mdl (RESULTS_l0_mdl.md 3b/3c): exact rows for the "
    "top-B tokens by attribution plus a compressed tail will dominate a pure "
    "compressed scheme at matched bits, by at least 1.5x in bits at matched "
    "KL somewhere on the curve."),
  "P7_corpus_statistics_do_not_pay_for_the_embedding": (
    "REGISTERED BEFORE SECTION L RAN (sections A-K were already measured; L "
    "was added in response to the self-red-team objection that the strongest "
    "structural hypothesis had not been tried). Predicting each embedding row "
    "from that token's PPMI co-occurrence statistics on the est split will "
    "reach a higher regression R^2 than the orthographic features did (0.256) "
    "-- somewhere in 0.40-0.65 -- but will still save LESS THAN 25% of the "
    "bits at matched KL even with the statistics given away free, because what "
    "the model needs is the residual, not the statistic."),
  "P6_no_weights_free_table_survives": (
    "No weights-free table description (the V x V token-pair table and its "
    "factorisations) will land anywhere near the frontier: all of them cost "
    "more bits than the fp32 model AND have higher KL."),
}


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save(out, path):
    json.dump(out, open(path, 'w'), indent=1)


# ---------------------------------------------------------------------------
def token_stats(D):
    """Frequencies and orthographic features from the ESTIMATION split only."""
    est = tf_corpus.load_split(D.V, 'est', 4000, tok=D.cfg.tok)
    cnt = np.bincount(est.reshape(-1), minlength=D.V).astype(np.float64)
    freq = torch.tensor(cnt, device=D.dev, dtype=torch.float32)
    vocab = tf_corpus.load_vocab(D.V, tok=D.cfg.tok)
    dec = vocab['decoded']
    labs = [dec[i] if i < len(dec) else '' for i in range(D.V)]
    cls = [I1._tok_class(s) for s in labs]
    return {'count': cnt, 'freq': freq, 'labels': labs, 'cls': cls,
            'uni_logp': torch.log(torch.tensor(
                (cnt + 1.0) / (cnt + 1.0).sum(), device=D.dev,
                dtype=torch.float32))}


def unigram_ce(D, TS):
    lp = TS['uni_logp']
    tot, n = 0.0, 0
    for x, y, _, _ in D.cache_ref():
        tot += float(-lp[y].sum())
        n += y.numel()
    return tot / n


def body_parts_bits(D, b):
    """Charge the six attention matrices + the three MLP matrices + bias."""
    bt = Bits()
    P = {}
    for k in D.PART_NAMES:
        if k in ('wte_read', 'wte_out'):
            continue
        W = D.base[k]
        if W.dim() == 1:
            P[k] = W.clone()
            bt.add(**{k: bits_dense(W.numel(), 32)})
            continue
        R, bb = CC.q_scalar(W, b)
        P[k] = R
        bt.add(**{k: bb.total})
    return P, bt


# ===========================================================================
# SECTION A -- the model as its own description
# ===========================================================================
def section_A(D, out):
    log('A: model self-quantisation frontier')
    rows = []
    # (i) everything at b bits (tied embedding counted once)
    for b in (2, 3, 4, 5, 6, 8, 12, 16, 32):
        Wq, be = CC.q_scalar(D.base['wte_read'], b)
        P, bb = body_parts_bits(D, b)
        P['wte_read'] = Wq
        P['wte_out'] = Wq
        s = D.score(P)
        bits = Bits(embedding=be.total).merge(bb, 'body_')
        rows.append({'scheme': f'uniform_{b}bit', 'emb_bits': b, 'body_bits': b,
                     'bits': bits.total, 'bill': bits.to_json(), **s})
        log('   ', rows[-1]['scheme'], rows[-1]['bits'], round(s['kl'], 5))
    # (ii) embedding quantised, body exact -- isolates the embedding's share
    for b in (2, 3, 4, 6, 8):
        Wq, be = CC.q_scalar(D.base['wte_read'], b)
        P = {'wte_read': Wq, 'wte_out': Wq}
        s = D.score(P)
        bits = Bits(embedding=be.total, body=bits_dense(D.n_params_body, 32))
        rows.append({'scheme': f'emb{b}bit_body32', 'emb_bits': b,
                     'body_bits': 32, 'bits': bits.total,
                     'bill': bits.to_json(), **s})
    # (iii) body quantised, embedding exact
    for b in (2, 3, 4, 6, 8):
        P, bb = body_parts_bits(D, b)
        s = D.score(P)
        bits = Bits(embedding=bits_dense(D.n_params_embed, 32)).merge(bb, 'body_')
        rows.append({'scheme': f'emb32_body{b}bit', 'emb_bits': 32,
                     'body_bits': b, 'bits': bits.total,
                     'bill': bits.to_json(), **s})
    out['A_self_quantisation'] = {
        'n_params_embed': D.n_params_embed, 'n_params_body': D.n_params_body,
        'n_params_total': D.n_params_model,
        'fp32_bits': 32 * D.n_params_model, 'rows': rows}
    return out


# ===========================================================================
# SECTION B -- memorisation vs structure
# ===========================================================================
def feature_groups(D, TS, want_k):
    """Groupings from token FEATURES only, at roughly `want_k` groups.  Returns
    (assignment, actual_k, description)."""
    V = D.V
    cnt = TS['count']
    order = np.argsort(-cnt)
    rank = np.empty(V, dtype=np.int64)
    rank[order] = np.arange(V)
    cls = TS['cls']
    ucls = sorted(set(cls))
    cid = np.array([ucls.index(c) for c in cls])
    L = np.array([min(len(s.replace('Ġ', ' ')), 8) for s in TS['labels']])
    nc, nl = len(ucls), 8
    per = max(1, int(round(want_k / (nc * nl))))
    fb = np.minimum(rank // max(1, V // max(per, 1)), per - 1)
    a = (cid * nl + (L - 1)) * per + fb
    _, a = np.unique(a, return_inverse=True)
    return (torch.tensor(a, device=D.dev), int(a.max() + 1),
            f'class({nc}) x len({nl}) x freq_band({per})')


def coarsen_score(D, table, assign, k, role, b=32, weights=None):
    """Replace every row of `table` by its (frequency-weighted) group mean, put
    it in `role`, score, and charge honestly."""
    V, d = table.shape
    w = torch.ones(V, device=table.device) if weights is None else weights.clamp_min(1e-6)
    C = torch.zeros(k, d, device=table.device)
    n = torch.zeros(k, device=table.device)
    C.index_add_(0, assign, table * w[:, None])
    n.index_add_(0, assign, w)
    C = C / n.clamp_min(1e-30)[:, None]
    if b < 32:
        C, bc = CC.q_scalar(C, b)
        cb = bc.total
    else:
        cb = bits_dense(C.numel(), 32)
    P = {role: C[assign]}
    s = D.score(P)
    bits = Bits(centroids=cb, index=bits_index(V, k))
    return s, bits


def section_B(D, out, TS):
    log('B: memorisation / structure split')
    V = D.V
    E = F.rms_norm(D.base['wte_read'], (D.Ws,))          # the READ table
    W = D.base['wte_out']                                 # the WRITE table
    freq = TS['freq']
    ks = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    res = {'read': [], 'write': [], 'read_features': [], 'read_random': [],
           'write_features': [], 'write_random': []}
    rng = np.random.default_rng(0)
    for k in ks:
        # --- learned behavioural clusters, frequency-weighted ---
        for role, tab, key in (('wte_read', E, 'read'), ('wte_out', W, 'write')):
            C, a = CC.kmeans_weighted(tab, k, weights=freq, seed=0)
            s, bits = coarsen_score(D, tab, a, k, role, weights=freq)
            res[key].append({'k': k, 'bits': bits.total, 'bill': bits.to_json(),
                             **s})
        log(f'   k={k:5d} read KL {res["read"][-1]["kl"]:.4f}  '
            f'write KL {res["write"][-1]["kl"]:.4f}')
        # --- random grouping control at the same k ---
        a = torch.tensor(rng.integers(0, k, size=V), device=D.dev)
        for role, tab, key in (('wte_read', E, 'read_random'),
                               ('wte_out', W, 'write_random')):
            s, bits = coarsen_score(D, tab, a, k, role, weights=freq)
            res[key].append({'k': k, 'bits': bits.total, **s})
    # --- feature groupings (their k is whatever the cross product gives) ---
    for want in (16, 48, 96, 192, 384, 768, 1536, 3072):
        a, k, desc = feature_groups(D, TS, want)
        for role, tab, key in (('wte_read', E, 'read_features'),
                               ('wte_out', W, 'write_features')):
            s, bits = coarsen_score(D, tab, a, k, role, weights=freq)
            res[key].append({'k': k, 'desc': desc, 'bits': bits.total, **s})
        log(f'   features k={k:5d} read KL {res["read_features"][-1]["kl"]:.4f} '
            f'write KL {res["write_features"][-1]["kl"]:.4f}')
    ce_uni = unigram_ce(D, TS)
    ce_model = D.score()['ce']
    out['B_split'] = {'unigram_ce': ce_uni, 'model_ce': ce_model,
                      'advantage_nats': ce_uni - ce_model, 'curves': res,
                      'note': ('read = the embedding as the layer-0 module '
                               'input (rms(wte)); write = the embedding as the '
                               'tied unembedding. Only one role is coarsened '
                               'at a time; the other is exact fp32.')}
    return out


# ===========================================================================
# SECTION C -- embedding schemes
# ===========================================================================
def section_C(D, out, TS):
    log('C: embedding compression schemes (tied)')
    W = D.base['wte_out']
    freq = TS['freq']
    rows = []

    def add(name, Wc, bits, extra=None):
        P = {'wte_read': Wc, 'wte_out': Wc}
        s = D.score(P)
        r = {'scheme': name, 'bits_embedding': bits.total,
             'bill': bits.to_json(),
             'bits_total_with_fp32_body': bits.total + 32 * D.n_params_body,
             **s}
        if extra:
            r.update(extra)
        rows.append(r)
        log(f'   {name:34s} {bits.total/1e6:7.3f} Mbit  KL {s["kl"]:.5f}')
        return r

    for b in (2, 3, 4, 5, 6, 8):
        Wc, bt = CC.q_scalar(W, b)
        add(f'scalar_q{b}', Wc, bt)
    for r in (4, 8, 16, 32, 48, 64, 96):
        Wc, bt = CC.q_lowrank(W, r)
        add(f'lowrank_r{r}', Wc, bt)
    for r in (16, 32, 64):
        Wc, bt = CC.q_lowrank(W, r, b=8)
        add(f'lowrank_r{r}_q8', Wc, bt)
    for m, nb in ((8, 8), (16, 8), (32, 8), (16, 6), (32, 6), (64, 8),
                  (16, 4), (32, 4), (64, 4), (128, 4), (128, 8)):
        Wc, bt = CC.q_pq(W, m, nb)
        add(f'pq_m{m}_b{nb}', Wc, bt)
    for k in (256, 512, 1024, 2048, 4096):
        Wc, bt = CC.q_cluster(W, k, weights=freq)
        add(f'cluster_k{k}', Wc, bt)
    out['C_embedding'] = {'rows': rows}
    return out


# ===========================================================================
# SECTION D -- the anchor-row hybrid, ported from ../qk_mdl
# ===========================================================================
def section_D(D, out, TS):
    log('D: anchor rows + compressed tail')
    W = D.base['wte_out']
    V, d = W.shape
    freq = TS['freq']
    order = torch.argsort(freq, descending=True)
    rows = []

    def add(name, Wc, bits, extra=None):
        P = {'wte_read': Wc, 'wte_out': Wc}
        s = D.score(P)
        r = {'scheme': name, 'bits_embedding': bits.total,
             'bill': bits.to_json(),
             'bits_total_with_fp32_body': bits.total + 32 * D.n_params_body, **s}
        if extra:
            r.update(extra)
        rows.append(r)
        log(f'   {name:38s} {bits.total/1e6:7.3f} Mbit  KL {s["kl"]:.5f}')
        return r

    # attribution: frequency (est split).  A sensitivity-based attribution is
    # run as a control below.
    for B in (0, 64, 256, 512, 1024, 2048):
        anchors = order[:B]
        for tail_name, tail_fn in (
                ('pq_m16_b8', lambda X: CC.q_pq(X, 16, 8)),
                ('pq_m8_b8', lambda X: CC.q_pq(X, 8, 8)),
                ('q4', lambda X: CC.q_scalar(X, 4)),
                ('q3', lambda X: CC.q_scalar(X, 3)),
                ('q2', lambda X: CC.q_scalar(X, 2)),
                ('cluster_k512', lambda X: CC.q_cluster(X, 512)),
        ):
            Wc = W.clone()
            mask = torch.ones(V, dtype=torch.bool, device=W.device)
            mask[anchors] = False
            tail_idx = torch.nonzero(mask).squeeze(1)
            Tail, bt = tail_fn(W[tail_idx])
            Wc[tail_idx] = Tail
            bits = Bits(anchor_rows=bits_dense(B * d, 32),
                        anchor_ids=bits_index(B, V) if B else 0)
            bits.merge(bt, 'tail_')
            add(f'anchor{B}_freq_tail_{tail_name}', Wc, bits,
                {'B': B, 'tail': tail_name, 'attribution': 'est_frequency'})
    # anchors at 8 bits instead of fp32 -- is exactness what matters, or rows?
    for B in (256, 1024):
        anchors = order[:B]
        Wc = W.clone()
        Aq, ba = CC.q_scalar(W[anchors], 8)
        Wc[anchors] = Aq
        mask = torch.ones(V, dtype=torch.bool, device=W.device)
        mask[anchors] = False
        ti = torch.nonzero(mask).squeeze(1)
        Tail, bt = CC.q_pq(W[ti], 16, 8)
        Wc[ti] = Tail
        bits = Bits(anchor_ids=bits_index(B, V)).merge(ba, 'anchor_') \
                                                .merge(bt, 'tail_')
        add(f'anchor{B}_q8_tail_pq_m16_b8', Wc, bits, {'B': B})
    out['D_anchor'] = {'rows': rows}
    return out


# ===========================================================================
# SECTION E -- the body: attention and MLP
# ===========================================================================
@torch.no_grad()
def cp_refit(D, h2, iters=60, seed=0):
    """Refit the bilinear MLP with h2 hidden units by ALS on the FOLDED tensor
    T[o,i,j] = sum_h Down[o,h] Left[h,i] Right[h,j], rather than truncating the
    trained units.  The neuron basis is a gauge; this asks whether the TENSOR
    has low CP rank, which is the basis-free question."""
    L, R, Dn = D.base['Left'], D.base['Right'], D.base['Down']
    hid, Ws = L.shape
    g = torch.Generator(device='cpu').manual_seed(seed)
    # init from the top-h2 trained units by |Down| column norm x factor norms
    sc = Dn.norm(dim=0) * L.norm(dim=1) * R.norm(dim=1)
    keep = torch.argsort(sc, descending=True)[:h2]
    A = L[keep].clone()            # (h2, Ws)
    B = R[keep].clone()
    Cm = Dn[:, keep].clone()       # (Ws, h2)
    # Target tensor is huge (Ws^3) but the ALS normal equations only need
    # Gram matrices, which are h2 x h2 / hid x hid -- never materialise T.
    GL, GR = L @ L.t(), R @ R.t()                       # (hid, hid)
    for _ in range(iters):
        # solve for Cm: minimise || sum_h C[:,h] a_h b_h^T - sum_g Dn[:,g] l_g r_g^T ||
        Gaa = (A @ A.t()) * (B @ B.t())                 # (h2, h2)
        Gax = (A @ L.t()) * (B @ R.t())                 # (h2, hid)
        Cm = torch.linalg.solve(Gaa + 1e-6 * torch.eye(h2, device=A.device),
                                Gax @ Dn.t()).t()
        # solve for A given B, Cm
        Gcc = (Cm.t() @ Cm) * (B @ B.t())
        rhs = ((Cm.t() @ Dn) * (B @ R.t())) @ L
        A = torch.linalg.solve(Gcc + 1e-6 * torch.eye(h2, device=A.device), rhs)
        # solve for B given A, Cm
        Gcc = (Cm.t() @ Cm) * (A @ A.t())
        rhs = ((Cm.t() @ Dn) * (A @ L.t())) @ R
        B = torch.linalg.solve(Gcc + 1e-6 * torch.eye(h2, device=A.device), rhs)
    return A, B, Cm


def section_E(D, out, TS):
    log('E: body compression')
    rows = []
    emb32 = bits_dense(D.n_params_embed, 32)

    def add(name, P, bits, extra=None):
        s = D.score(P)
        r = {'scheme': name, 'bits_body': bits.total, 'bill': bits.to_json(),
             'bits_total_with_fp32_embedding': bits.total + emb32, **s}
        if extra:
            r.update(extra)
        rows.append(r)
        log(f'   {name:34s} {bits.total/1e6:7.3f} Mbit  KL {s["kl"]:.5f}')
        return r

    attn_keys = ('Wq', 'Wk', 'Wq2', 'Wk2', 'Wv', 'Wproj')
    mlp_keys = ('Left', 'Right', 'Down')
    n_attn = sum(D.base[k].numel() for k in attn_keys)
    n_mlp = sum(D.base[k].numel() for k in mlp_keys) + D.base['Down_bias'].numel()

    # --- control: unit truncation in the NEURON basis (the known failure) ---
    L, R, Dn = D.base['Left'], D.base['Right'], D.base['Down']
    sc = Dn.norm(dim=0) * L.norm(dim=1) * R.norm(dim=1)
    order = torch.argsort(sc, descending=True)
    for h2 in (32, 64, 128, 256, 384):
        keep = order[:h2]
        P = {'Left': L[keep], 'Right': R[keep], 'Down': Dn[:, keep]}
        bits = Bits(LR=bits_dense(2 * h2 * D.Ws, 32),
                    Down=bits_dense(D.Ws * h2, 32),
                    bias=bits_dense(D.Ws, 32),
                    attn=bits_dense(n_attn, 32))
        add(f'mlp_trunc_units{h2}', P, bits, {'h2': h2})
    # --- refitted CP at the same widths (the basis-free question) ---
    for h2 in (32, 64, 128, 256, 384):
        A, B, Cm = cp_refit(D, h2)
        P = {'Left': A, 'Right': B, 'Down': Cm}
        bits = Bits(LR=bits_dense(2 * h2 * D.Ws, 32),
                    Down=bits_dense(D.Ws * h2, 32),
                    bias=bits_dense(D.Ws, 32),
                    attn=bits_dense(n_attn, 32))
        add(f'mlp_cp_refit{h2}', P, bits, {'h2': h2})
    # --- scalar quantisation of the MLP only ---
    for b in (2, 3, 4, 6, 8):
        P = {}
        tot = 0
        for k in mlp_keys:
            Rq, bb = CC.q_scalar(D.base[k], b)
            P[k] = Rq
            tot += bb.total
        bits = Bits(mlp=tot, bias=bits_dense(D.Ws, 32),
                    attn=bits_dense(n_attn, 32))
        add(f'mlp_q{b}', P, bits)
    # --- quantised refitted CP: does the structure help ON TOP of quantising ---
    for h2 in (128, 256, 384):
        for b in (4, 6, 8):
            A, B, Cm = cp_refit(D, h2)
            P, tot = {}, 0
            for k, Wm in (('Left', A), ('Right', B), ('Down', Cm)):
                Rq, bb = CC.q_scalar(Wm, b)
                P[k] = Rq
                tot += bb.total
            bits = Bits(mlp=tot, bias=bits_dense(D.Ws, 32),
                        attn=bits_dense(n_attn, 32))
            add(f'mlp_cp{h2}_q{b}', P, bits, {'h2': h2})
    # --- attention: quantisation and per-head low rank ---
    for b in (2, 3, 4, 6, 8):
        P, tot = {}, 0
        for k in attn_keys:
            Rq, bb = CC.q_scalar(D.base[k], b)
            P[k] = Rq
            tot += bb.total
        bits = Bits(attn=tot, mlp=bits_dense(n_mlp, 32))
        add(f'attn_q{b}', P, bits)
    # --- whole body quantised at b, MLP refitted to 256 CP terms ---
    for b in (3, 4, 6):
        A, B, Cm = cp_refit(D, 256)
        P, tot = {}, 0
        for k, Wm in list(zip(mlp_keys, (A, B, Cm))) + \
                     [(k, D.base[k]) for k in attn_keys]:
            Rq, bb = CC.q_scalar(Wm, b)
            P[k] = Rq
            tot += bb.total
        bits = Bits(body=tot, bias=bits_dense(D.Ws, 32))
        add(f'body_cp256_q{b}', P, bits)
    out['E_body'] = {'rows': rows, 'n_attn_params': n_attn,
                     'n_mlp_params': n_mlp}
    return out


# ===========================================================================
# SECTION F -- the combined frontier and the confirmation cell
# ===========================================================================
def combined_point(D, TS, emb_spec, body_spec, anchors=0):
    """Build one joint description and return (row, P, Bits)."""
    W = D.base['wte_out']
    V, d = W.shape
    freq = TS['freq']
    bits = Bits()
    Wc = W.clone()
    if anchors:
        order = torch.argsort(freq, descending=True)
        an = order[:anchors]
        mask = torch.ones(V, dtype=torch.bool, device=W.device)
        mask[an] = False
        ti = torch.nonzero(mask).squeeze(1)
        Tail, bt = emb_spec[1](W[ti])
        Wc[ti] = Tail
        bits.add(anchor_rows=bits_dense(anchors * d, 32),
                 anchor_ids=bits_index(anchors, V))
        bits.merge(bt, 'tail_')
    else:
        Wc, bt = emb_spec[1](W)
        bits.merge(bt, 'emb_')
    P = {'wte_read': Wc, 'wte_out': Wc}
    Pb, bb = body_spec[1](D)
    P.update(Pb)
    bits.merge(bb, 'body_')
    s = D.score(P)
    return {'scheme': f'{emb_spec[0]}+{body_spec[0]}'
                      + (f'+anchor{anchors}' if anchors else ''),
            'bits': bits.total, 'bill': bits.to_json(), **s}


def section_F(D, out, TS):
    """Cross the coders that actually won their own section: the embedding gets
    the best-in-class scalar code (entropy-coded uniform, per-column allocated
    transform code, frequency-stratified precision) and the body gets scalar
    quantisation at b bits, which beat every structural MLP scheme in E."""
    log('F: combined descriptions (best-in-class coders crossed)')
    W = D.base['wte_out']
    freq = TS['freq']
    order = torch.argsort(freq, descending=True)
    rows = []
    emb_specs = [
        ('embQ3e', lambda: CC.q_scalar_entropy(W, 3)),
        ('embQ4e', lambda: CC.q_scalar_entropy(W, 4)),
        ('embQ5e', lambda: CC.q_scalar_entropy(W, 5)),
        ('embT512', lambda: CC.q_transform(W, 512, rot='none')),
        ('embT640', lambda: CC.q_transform(W, 640, rot='none')),
        ('embT768', lambda: CC.q_transform(W, 768, rot='none')),
        ('embS6_4_2048', lambda: CC.q_stratified(W, 6, 4, 2048, order)),
        ('embS6_3_1024', lambda: CC.q_stratified(W, 6, 3, 1024, order)),
        ('embS8_5_512', lambda: CC.q_stratified(W, 8, 5, 512, order)),
    ]
    for en, ef in emb_specs:
        Wc, be = ef()
        for bb in (4, 5, 6, 8, 32):
            P, bbits = body_parts_bits(D, bb)
            P['wte_read'] = Wc
            P['wte_out'] = Wc
            s = D.score(P)
            bits = Bits(embedding=be.total).merge(bbits, 'body_')
            rows.append({'scheme': f'{en}+body{bb}', 'bits': bits.total,
                         'bill': bits.to_json(), 'emb_coder': en,
                         'body_bits': bb, **s})
            log(f'   {en}+body{bb:<2d} {bits.total/1e6:7.3f} Mbit  '
                f'KL {s["kl"]:.5f}')
    out['F_combined'] = {'rows': rows}
    return out


# ===========================================================================
# SECTION G -- better codes: transform coding, entropy coding, stratified bits
# ===========================================================================
def section_G(D, out, TS):
    log('G: transform / entropy / stratified codes for the embedding')
    W = D.base['wte_out']
    freq = TS['freq']
    order = torch.argsort(freq, descending=True)
    rows = []

    def add(name, Wc, bits, extra=None):
        P = {'wte_read': Wc, 'wte_out': Wc}
        s = D.score(P)
        r = {'scheme': name, 'bits_embedding': bits.total, 'bill': bits.to_json(),
             'bits_total_with_fp32_body': bits.total + 32 * D.n_params_body, **s}
        if extra:
            r.update(extra)
        rows.append(r)
        log(f'   {name:36s} {bits.total/1e6:7.3f} Mbit  KL {s["kl"]:.5f}')
        return r

    for b in (3, 4, 5, 6, 8):
        Wc, bt = CC.q_scalar_entropy(W, b)
        add(f'scalar_q{b}_entropy', Wc, bt)
    for rot in ('none', 'hadamard', 'pca'):
        for bpr in (256, 384, 512, 640, 768):
            Wc, bt = CC.q_transform(W, bpr, rot=rot, entropy=True)
            add(f'transform_{rot}_{bpr}bpr', Wc, bt, {'rot': rot, 'bpr': bpr})
    for (bh, bl, nh) in ((8, 3, 512), (8, 3, 1024), (8, 4, 512), (6, 3, 1024),
                         (6, 4, 1024), (8, 4, 1024), (6, 4, 2048), (8, 5, 512),
                         (6, 5, 1024), (5, 4, 2048)):
        Wc, bt = CC.q_stratified(W, bh, bl, nh, order)
        add(f'strat_hi{bh}_lo{bl}_n{nh}', Wc, bt,
            {'b_hi': bh, 'b_lo': bl, 'n_hi': nh})
    out['G_codes'] = {'rows': rows}
    return out


# ===========================================================================
# SECTION J -- the memorisation/structure split done in the currency that
# actually works: BITS PER TOKEN PER ROLE, not number of prototypes.
# ===========================================================================
def section_J(D, out, TS):
    log('J: per-role precision, size-matched grouping controls')
    W = D.base['wte_out']
    E = F.rms_norm(D.base['wte_read'], (D.Ws,))
    freq = TS['freq']
    res = {'read_precision': [], 'write_precision': [], 'both_precision': [],
           'feature_size_matched_random': [], 'write_unweighted': []}
    for b in (1, 2, 3, 4, 5, 6, 8):
        Rq, bt = CC.q_scalar_entropy(E, b)
        s = D.score({'wte_read': Rq})
        res['read_precision'].append({'b': b, 'bits_this_table': bt.total, **s})
        Wq, bt2 = CC.q_scalar_entropy(W, b)
        s2 = D.score({'wte_out': Wq})
        res['write_precision'].append({'b': b, 'bits_this_table': bt2.total, **s2})
        s3 = D.score({'wte_out': Wq, 'wte_read': Wq})
        res['both_precision'].append({'b': b, 'bits_this_table': bt2.total, **s3})
        log(f'   b={b}: read-only KL {s["kl"]:.5f}  write-only KL '
            f'{s2["kl"]:.5f}  tied KL {s3["kl"]:.5f}')
    # size-matched random control for the feature groupings: same group-size
    # histogram, random membership.  Without this, "features lose to random"
    # could just be "unbalanced groups lose to balanced groups".
    rng = np.random.default_rng(1)
    for want in (96, 324, 1056):
        a, k, desc = feature_groups(D, TS, want)
        sizes = torch.bincount(a, minlength=k).cpu().numpy()
        lab = np.concatenate([np.full(int(n), i) for i, n in enumerate(sizes)])
        rng.shuffle(lab)
        ar = torch.tensor(lab, device=D.dev)
        for role, tab, nm in (('wte_read', E, 'read'), ('wte_out', W, 'write')):
            sf, bf = coarsen_score(D, tab, a, k, role, weights=freq)
            sr, br = coarsen_score(D, tab, ar, k, role, weights=freq)
            res['feature_size_matched_random'].append(
                {'k': k, 'role': nm, 'desc': desc, 'bits': bf.total,
                 'feature_kl': sf['kl'], 'size_matched_random_kl': sr['kl']})
            log(f'   k={k} {nm}: features {sf["kl"]:.4f} vs size-matched '
                f'random {sr["kl"]:.4f}')
    # the write role clustered WITHOUT frequency weights: every row of the
    # unembedding sits in every softmax denominator, so frequency weighting is
    # the wrong importance for that role and could be handicapping it.
    for k in (256, 1024, 4096):
        C, a = CC.kmeans(W, k, iters=60)
        s, bt = coarsen_score(D, W, a, k, 'wte_out', weights=None)
        res['write_unweighted'].append({'k': k, 'bits': bt.total, **s})
        log(f'   write unweighted k={k}: KL {s["kl"]:.4f}')
    out['J_roles'] = res
    return out



# ===========================================================================
# SECTION K -- can the token's SPELLING pay for its embedding row?
# The decoder already knows every token's surface string (the tokenizer is part
# of the corpus specification, shared by the model and by every description, so
# it is not charged).  If the embedding were partly a function of orthography
# and frequency, predicting the row from those features and coding only the
# RESIDUAL would be strictly cheaper.  This is Logan's "the memorisation should
# compress too" in the currency that actually decides the frontier -- bits, not
# prototypes.
# ===========================================================================
def token_features(D, TS):
    V = D.V
    labs = TS['labels']
    cnt = TS['count']
    cls = TS['cls']
    ucls = sorted(set(cls))
    NH = 256
    F_ = np.zeros((V, 1 + len(ucls) + 8 + NH + 1), dtype=np.float32)
    lg = np.log1p(cnt)
    F_[:, 0] = (lg - lg.mean()) / (lg.std() + 1e-9)
    for i, s in enumerate(labs):
        F_[i, 1 + ucls.index(cls[i])] = 1.0
        L = min(max(len(s.replace('\u0120', ' ')), 1), 8)
        F_[i, 1 + len(ucls) + L - 1] = 1.0
        core = s
        grams = [core[j] for j in range(len(core))] + \
                [core[j:j + 2] for j in range(len(core) - 1)]
        for gmm in grams:
            F_[i, 1 + len(ucls) + 8 + (hash(gmm) % NH)] += 1.0
        F_[i, -1] = 1.0
    Phi = torch.tensor(F_, device=D.dev)
    Phi[:, 1 + len(ucls) + 8:-1] /= Phi[:, 1 + len(ucls) + 8:-1].std().clamp_min(1e-6)
    return Phi


def section_K(D, out, TS):
    log('K: orthography/frequency-conditional coding of the embedding')
    W = D.base['wte_out']
    Phi = token_features(D, TS)
    nf = Phi.shape[1]
    G = Phi.t() @ Phi + 1e-2 * torch.eye(nf, device=D.dev)
    B = torch.linalg.solve(G, Phi.t() @ W)
    pred = Phi @ B
    r2 = 1 - float(((W - pred) ** 2).sum() / (W - W.mean(0)).pow(2).sum())
    log(f'   feature regression R^2 on the embedding = {r2:.4f} '
        f'({nf} features)')
    Bq, bb = CC.q_scalar(B, 8)
    predq = Phi @ Bq
    rows = []
    for b in (2, 3, 4, 5, 6):
        Rq, br = CC.q_scalar_entropy(W - predq, b)
        Wc = predq + Rq
        bits = Bits(regression=bb.total).merge(br, 'residual_')
        s = D.score({'wte_read': Wc, 'wte_out': Wc})
        Wp, bp = CC.q_scalar_entropy(W, b)
        s0 = D.score({'wte_read': Wp, 'wte_out': Wp})
        rows.append({'scheme': f'feature_residual_q{b}', 'b': b,
                     'bits_embedding': bits.total, 'bill': bits.to_json(),
                     'bits_total_with_fp32_body': bits.total
                     + 32 * D.n_params_body, **s,
                     'control_plain_q': {'bits_embedding': bp.total, **s0}})
        log(f'   b={b}: feature-residual {bits.total/1e6:6.3f} Mbit KL '
            f'{s["kl"]:.5f}   plain {bp.total/1e6:6.3f} Mbit KL {s0["kl"]:.5f}')
    out['K_features'] = {'regression_r2': r2, 'n_features': nf, 'rows': rows,
                         'note': ('the token surface strings are free to the '
                                  'decoder (they are part of the corpus '
                                  'specification), which is generous to this '
                                  'scheme and therefore conservative for the '
                                  'negative result')}
    return out



# ===========================================================================
# SECTION L -- is the embedding just a compressed CORPUS STATISTIC?
# The strongest structural hypothesis available: the decoder is allowed to
# recompute the estimation split's co-occurrence table for itself (the corpus
# is a fixed public object in this program, the same convention that makes the
# token strings free), predict each embedding row from that token's empirical
# context statistics, and code only the residual.  If the embedding is
# "memorised bigram statistics", this must win.  Charged BOTH ways: with the
# statistics free, and with the projection basis charged at 8 bits.
# ===========================================================================
def corpus_stat_features(D, r=128, n_seq=8000, T=512):
    V = D.V
    arr = tf_corpus.load_split(V, 'est', n_seq, tok=D.cfg.tok)
    a = torch.from_numpy(arr).to(D.dev)
    Cm = torch.zeros(V, V, device=D.dev)
    for s0 in range(0, a.shape[0], 256):
        b = a[s0:s0 + 256]
        idx = (b[:, :-1].reshape(-1) * V + b[:, 1:].reshape(-1))
        Cm.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    # PPMI, the standard co-occurrence representation
    tot = Cm.sum()
    pr = Cm.sum(1, keepdim=True) / tot
    pc = Cm.sum(0, keepdim=True) / tot
    P = Cm / tot
    M = torch.log((P / (pr * pc).clamp_min(1e-12)).clamp_min(1e-12) + 1.0)
    U, S, Vh = torch.svd_lowrank(M, q=r + 16, niter=4)
    left = U[:, :r] * S[:r]                    # token as PREDECESSOR
    U2, S2, _ = torch.svd_lowrank(M.t(), q=r + 16, niter=4)
    right = U2[:, :r] * S2[:r]                 # token as SUCCESSOR
    Phi = torch.cat([left, right, torch.ones(V, 1, device=D.dev)], 1)
    Phi = Phi / Phi.std(0, keepdim=True).clamp_min(1e-6)
    return Phi


def section_L(D, out, TS):
    log('L: corpus-statistic-conditional coding of the embedding')
    W = D.base['wte_out']
    Phi = corpus_stat_features(D)
    nf = Phi.shape[1]
    G = Phi.t() @ Phi + 1e-2 * torch.eye(nf, device=D.dev)
    B = torch.linalg.solve(G, Phi.t() @ W)
    r2 = 1 - float(((W - Phi @ B) ** 2).sum() / (W - W.mean(0)).pow(2).sum())
    log(f'   corpus-statistic regression R^2 = {r2:.4f} ({nf} features)')
    Bq, bb = CC.q_scalar(B, 8)
    predq = Phi @ Bq
    rows = []
    for b in (2, 3, 4, 5, 6):
        Rq, br = CC.q_scalar_entropy(W - predq, b)
        Wc = predq + Rq
        s = D.score({'wte_read': Wc, 'wte_out': Wc})
        Wp, bp = CC.q_scalar_entropy(W, b)
        s0 = D.score({'wte_read': Wp, 'wte_out': Wp})
        free = Bits(regression=bb.total).merge(br, 'residual_')
        charged = Bits(regression=bb.total,
                       basis=CC.bits_dense(Phi.numel(), 8)).merge(br, 'residual_')
        rows.append({'scheme': f'corpusstat_residual_q{b}', 'b': b,
                     'bits_embedding': free.total,
                     'bits_embedding_basis_charged': charged.total,
                     'bill': free.to_json(),
                     'bits_total_with_fp32_body': free.total
                     + 32 * D.n_params_body, **s,
                     'control_plain_q': {'bits_embedding': bp.total, **s0}})
        log(f'   b={b}: stat-residual {free.total/1e6:6.3f} Mbit '
            f'(basis charged {charged.total/1e6:6.3f}) KL {s["kl"]:.5f}   '
            f'plain {bp.total/1e6:6.3f} Mbit KL {s0["kl"]:.5f}')
    out['L_corpus_stats'] = {'regression_r2': r2, 'n_features': nf,
                             'rows': rows}
    return out



# ===========================================================================
# SECTION M -- the CONDITIONAL codes crossed with a quantised body, so the
# frontier is not biased against the structural schemes by leaving their body
# at fp32 while the winners quantise theirs.
# ===========================================================================
def section_M(D, out, TS):
    log('M: conditional codes x quantised body')
    W = D.base['wte_out']
    rows = []
    coders = []
    PhiF = token_features(D, TS)
    PhiC = corpus_stat_features(D)
    for nm, Phi in (('spelling', PhiF), ('corpusstat', PhiC)):
        nf = Phi.shape[1]
        G = Phi.t() @ Phi + 1e-2 * torch.eye(nf, device=D.dev)
        B = torch.linalg.solve(G, Phi.t() @ W)
        Bq, bb = CC.q_scalar(B, 8)
        pred = Phi @ Bq
        for b in (3, 4, 5):
            Rq, br = CC.q_scalar_entropy(W - pred, b)
            coders.append((f'{nm}_res_q{b}', pred + Rq,
                           Bits(regression=bb.total).merge(br, 'residual_')))
    for en, Wc, be in coders:
        for bb2 in (5, 6, 8):
            P, bbits = body_parts_bits(D, bb2)
            P['wte_read'] = Wc
            P['wte_out'] = Wc
            sc = D.score(P)
            bits = Bits(embedding=be.total).merge(bbits, 'body_')
            rows.append({'scheme': f'{en}+body{bb2}', 'bits': bits.total,
                         'bill': bits.to_json(), **sc})
            log(f'   {en}+body{bb2:<2d} {bits.total/1e6:7.3f} Mbit  '
                f'KL {sc["kl"]:.5f}')
    out['M_conditional_combined'] = {'rows': rows}
    return out


# ===========================================================================
# SECTION I -- DISTILLED descriptions: fit the quantised tables on `est`
# ===========================================================================
def ste_q(W, b):
    """Straight-through uniform quantisation with per-row min/max scales --
    exactly the decoder of `q_scalar`, so the trained object IS the description
    that gets charged."""
    if b >= 32:
        return W
    W2 = W.reshape(W.shape[0], -1)
    lo = W2.min(dim=1, keepdim=True).values.half().float()
    hi = W2.max(dim=1, keepdim=True).values.half().float()
    step = ((hi - lo) / (2 ** b - 1)).clamp_min(1e-30)
    q = ((W2 - lo) / step).round().clamp(0, 2 ** b - 1)
    R = (q * step + lo).reshape(W.shape)
    return W + (R - W).detach()


def distill(D, emb_b, body_b, steps=10000, lr=6e-4, T=256, batch=8,
            n_seq=8000, seed=0, val_every=250):
    """Re-fit the description's tables on the ESTIMATION split against the true
    model's own distribution, with the quantiser in the loop.  Nothing is fitted
    on held: the iterate is selected on a DISJOINT est slice.  The returned
    tables are decoded EXACTLY as the post-hoc quantiser decodes them, so the
    bit bill is the same object.  Post-hoc quantisation is the initialisation,
    so distillation can only fail to help if the optimisation is bad -- which is
    why the best-iterate selection is on est, not on the last step."""
    torch.manual_seed(seed)
    keys = [k for k in D.PART_NAMES if k != 'wte_read']
    p = {k: torch.nn.Parameter(D.base[k].clone()) for k in keys}
    opt = torch.optim.Adam(p.values(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 100) * 0.5
        * (1 + math.cos(math.pi * min(1.0, t / steps))))
    arr = tf_corpus.load_split(D.V, 'est', n_seq + 256, tok=D.cfg.tok)
    x_all = torch.from_numpy(arr[:n_seq, :T]).to(D.dev)
    x_val = torch.from_numpy(arr[n_seq:n_seq + 256, :T]).to(D.dev)
    nb = x_all.shape[0] // batch

    def qdict(src):
        Q = {}
        for k in keys:
            b = emb_b if k == 'wte_out' else body_b
            Q[k] = ste_q(src[k], b) if src[k].dim() > 1 else src[k]
        Q['wte_read'] = Q['wte_out']
        return Q

    @torch.no_grad()
    def val_kl():
        Q = qdict({k: v.detach() for k, v in p.items()})
        tot, n = 0.0, 0
        for a in range(0, x_val.shape[0], 16):
            xb = x_val[a:a + 16]
            lr_ = F.log_softmax(D.model(xb).float(), -1)
            lg = F.log_softmax(D.forward(xb, Q).float(), -1)
            tot += float((lr_.exp() * (lr_ - lg)).sum())
            n += xb.numel()
        return tot / n

    best = (val_kl(), {k: v.detach().clone() for k, v in p.items()}, 0)
    for it in range(steps):
        x = x_all[(it % nb) * batch:(it % nb + 1) * batch]
        with torch.no_grad():
            lref = F.log_softmax(D.model(x).float(), -1)
            pref = lref.exp()
        lg = F.log_softmax(D.forward(x, qdict(p)).float(), -1)
        loss = (pref * (lref - lg)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(p.values()), 1.0)
        opt.step()
        sch.step()
        if (it + 1) % val_every == 0:
            v = val_kl()
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
    return P, bits, {'est_val_kl': best[0], 'best_step': best[2]}


def section_I(D, out, TS):
    log('I: distilled compressed descriptions (tables re-fit on est)')
    rows = []
    for emb_b, body_b in ((1, 4), (2, 3), (2, 4), (2, 6), (3, 3), (3, 4),
                          (3, 6), (4, 4), (4, 6), (4, 8), (5, 8), (6, 8)):
        # control: the SAME bit budget without distillation
        Wq, be = CC.q_scalar_entropy(D.base['wte_out'], emb_b)
        P0 = {'wte_read': Wq, 'wte_out': Wq}
        b0 = Bits(wte=be.total)
        for k in D.PART_NAMES:
            if k in ('wte_read', 'wte_out'):
                continue
            W = D.base[k]
            if W.dim() == 1:
                P0[k] = W
                b0.add(**{k: bits_dense(W.numel(), 32)})
            else:
                R, bb = CC.q_scalar_entropy(W, body_b)
                P0[k] = R
                b0.add(**{k: bb.total})
        s0 = D.score(P0)
        P, bits, tl = distill(D, emb_b, body_b)
        s = D.score(P)
        s.update(D.score_se(P))
        rows.append({'scheme': f'distilled_emb{emb_b}_body{body_b}',
                     'emb_bits': emb_b, 'body_bits': body_b,
                     'bits': bits.total, 'bill': bits.to_json(),
                     'distill_selection': tl, **s,
                     'control_posthoc': {'bits': b0.total, **s0}})
        log(f'   emb{emb_b}/body{body_b}: {bits.total/1e6:6.3f} Mbit  '
            f'distilled KL {s["kl"]:.5f}   post-hoc {b0.total/1e6:6.3f} Mbit '
            f'KL {s0["kl"]:.5f}')
    out['I_distilled'] = {'rows': rows}
    return out


# ===========================================================================
# SECTION H -- the weights-free table descriptions, priced
# ===========================================================================
def section_H(D, out, TS):
    """The rung-5 artifacts that motivated this file, put on the same axes."""
    log('H: weights-free table descriptions, priced in bits')
    import tf_interp as I
    Dep = I.Depth1Fold(D.stem)
    V = D.V
    rows = []
    # the model's own bigram: r0 (V, Ws) read out with W_U.  As a WEIGHTS-FREE
    # object it must be the V x V logit table.
    lad = I.ladder(Dep, n_seq=64, T=256, batch=8, extra=False)
    kl_bg = lad['model_bigram']['kl_from_model']
    for b, nm in ((32, 'fp32'), (16, 'fp16'), (8, 'int8')):
        rows.append({'scheme': f'weightsfree_VxV_bigram_table_{nm}',
                     'bits': V * V * b, 'kl': kl_bg,
                     'ce': lad['model_bigram']['ce'],
                     'note': 'the only weights-free artifact rung 5 had'})
    # its factored form -- still a table program, but two V x Ws matrices
    rows.append({'scheme': 'factored_r0_plus_WU_fp32',
                 'bits': 2 * V * D.Ws * 32, 'kl': kl_bg,
                 'ce': lad['model_bigram']['ce'],
                 'note': 'r0 row table + the unembedding; MDL makes the '
                         '"weights-free" distinction vacuous'})
    rows.append({'scheme': 'full_exact_ladder_control',
                 'bits': None, 'kl': lad['full_exact']['kl_from_model'],
                 'ce': lad['full_exact']['ce']})
    out['H_weightsfree'] = {'rows': rows, 'V': V}
    log('   V x V bigram table', V * V, 'entries, KL', round(kl_bg, 4))
    return out


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default=STEM)
    ap.add_argument('--sections', default='ABCDEFGHIJKLM')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    path = a.out or f'{HERE}/{a.stem}_compress.json'
    out = load_json(path)
    out['stem'] = a.stem
    out['registered_predictions'] = REGISTERED
    save(out, path)
    D = CC.D1Desc(a.stem)
    out['positive_control'] = D.self_check()
    log('gate', out['positive_control'])
    assert out['positive_control']['rel_logit_diff'] < 1e-5
    TS = token_stats(D)
    out['model'] = {'ce': D.score()['ce'],
                    'n_params': D.n_params_model,
                    'fp32_bits': 32 * D.n_params_model}
    for sec in a.sections:
        t0 = time.time()
        {'A': lambda: section_A(D, out),
         'B': lambda: section_B(D, out, TS),
         'C': lambda: section_C(D, out, TS),
         'D': lambda: section_D(D, out, TS),
         'E': lambda: section_E(D, out, TS),
         'F': lambda: section_F(D, out, TS),
         'G': lambda: section_G(D, out, TS),
         'H': lambda: section_H(D, out, TS),
         'I': lambda: section_I(D, out, TS),
         'J': lambda: section_J(D, out, TS),
         'K': lambda: section_K(D, out, TS),
         'L': lambda: section_L(D, out, TS),
         'M': lambda: section_M(D, out, TS)}[sec]()
        log(f'section {sec} done in {time.time()-t0:.0f}s')
        save(out, path)
    save(out, path)
    log('wrote', path)


if __name__ == '__main__':
    main()
