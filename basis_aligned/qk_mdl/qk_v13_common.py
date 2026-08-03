"""V13 (Logan update six): slots + group-lasso base with the N=6 window (V9
visibility rule: blocks read the last 6 blocks' writes, embedding visible only
l < 6, readout reads blocks 6..11) PLUS low-rank PER-EDGE ADAPTERS on every
visible module->module and module->readout edge:

    consumer l reads source i as  w_i + (w_i @ V_e) @ U_e^T,   e = edge (l, i)

rank r = 4 (sub-variant r = 1), applied as thin matmuls (A_e = U_e V_e^T is
never materialized in the forward). Init: U_e = 0, V_e ~ N(0, 0.02) -- the
adapter correction is exactly zero at init (function-identical to the plain
slots+window model) and there is no dead-edge fixed point (grad U_e != 0).
Per-edge group-lasso on ||U_e||_F * ||V_e||_F (Logan's product form), same 1e-4
coefficient as the read lasso, so unused edges die visibly.

Hypothesis to score: edge adapters let modules keep their own basis and
translate per consumer -> BETTER term concentration than the plain window
config (fewer terms to 95 percent at blocks 4/6/9, lower effective inputs) at
little parameter cost (114 edges x 2 x 264 x r).

Width 264, depth 12, slot 11, single lr 0.002, 6-epoch budget, batch 4,
matched control = qk_v264_vanilla (already trained by qk_v10v11_train.py).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_deeproute_train_2 as R2
import qk_v9_common as C
import qk_v10v11_common as W
from qk_deeproute_train import DEPTH

QK = Q.QK
DEV = 'cuda'

VIS_V13 = [C.window_vis(li, N=6) for li in range(DEPTH + 1)]
EDGES = [(li, si) for li in range(DEPTH + 1) for si in VIS_V13[li] if si != 0]


class V13Route(V8T.V8Route):
    """Slots + window-6 with per-edge low-rank read adapters."""

    def __init__(self, variant, depth, rank=4):
        super().__init__(variant, depth)
        Dm = self.wte.weight.shape[1]
        self.rank = rank
        self.vis = [list(v) for v in VIS_V13]
        self.edges = list(EDGES)
        self.edge_index = {e: j for j, e in enumerate(self.edges)}
        gen = torch.Generator().manual_seed(1313)
        self.ad_U = nn.ParameterList(
            [nn.Parameter(torch.zeros(Dm, rank)) for _ in self.edges])
        self.ad_V = nn.ParameterList(
            [nn.Parameter(torch.randn(Dm, rank, generator=gen) * 0.02)
             for _ in self.edges])
        self.dec_lasso = True                    # hooks into the patched penalty

    def dec_penalty(self):                       # sum_e ||U_e||_F * ||V_e||_F
        tot = None
        for Ue, Ve in zip(self.ad_U, self.ad_V):
            g = (Ue.pow(2).sum() + 1e-12).sqrt() * (Ve.pow(2).sum() + 1e-12).sqrt()
            tot = g if tot is None else tot + g
        return tot

    def adapt(self, li, si, s):
        """Adapted read of stream si by consumer li."""
        if si == 0:
            return s
        j = self.edge_index[(li, si)]
        Uj = self.ad_U[j].to(s.dtype)
        Vj = self.ad_V[j].to(s.dtype)
        return s + (s @ Vj) @ Uj.t()

    def assemble(self, li, streams, sub=None, coef_out=None):
        idxs = self.vis[li]
        get = lambda i: (sub[i] if (sub is not None and i in sub) else streams[i])
        h, cs = None, []
        for i in idxs:
            t = self.adapt(li, i, get(i))
            h = t if h is None else h + t
            if coef_out is not None:
                s = get(i)
                cs.append(t.detach().float().norm(dim=-1)
                          / (s.detach().float().norm(dim=-1) + 1e-12))
        if coef_out is not None:
            coef_out[li] = (idxs, torch.stack(cs, -1))
        return h

    def edge_product_norm(self, li, si):
        j = self.edge_index[(li, si)]
        return float(self.ad_U[j].detach().float().norm()
                     * self.ad_V[j].detach().float().norm())

    def edge_matrix(self, li, si):
        """A_e = U_e V_e^T, ANALYSIS only."""
        j = self.edge_index[(li, si)]
        return (self.ad_U[j].float() @ self.ad_V[j].float().t()).detach()


def make_v13(variant='V13', rank=4):
    C.register(variant)
    torch.manual_seed(Q.SEED)
    return V13Route(variant, DEPTH, rank=rank).to(DEV)


def load_v13(stem, variant, rank):
    ck = torch.load(f'{QK}/{stem}.pt', map_location=DEV, weights_only=False)
    m = make_v13(variant, rank)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


@torch.no_grad()
def v13_controls():
    """(a) edge census; (b) V13 at init == plain slots+window model at init
    (adapter corrections are exactly zero); (c) penalty at init is exactly 0."""
    n_edges = len(EDGES)
    per_edge = 2 * W.WIDTH * 4
    print(f"V13 edges: {n_edges} (expect 114), rank-4 adapter params "
          f"{n_edges * per_edge}", flush=True)
    assert n_edges == 114
    idx = Q.HELD[:2, :Q.T]
    ref = C.make_variant('V13ref', lambda li: C.window_vis(li, N=6))
    ref = ref.eval().float()
    out_ref = ref(idx)
    del ref
    torch.cuda.empty_cache()
    m = make_v13('V13', 4).eval().float()
    p0 = float(m.dec_penalty())
    d = (m(idx) - out_ref).abs().max().item()
    print(f"control V13(init)==slots+window(init): max |logit diff| {d:.2e}; "
          f"adapter penalty at init {p0:.2e} (U=0)", flush=True)
    assert d < 1e-3 and p0 < 1e-4
    del m, out_ref
    torch.cuda.empty_cache()


# ---------------- V13-aware exact-term decomposition (replaces R2's) ----------------
@torch.no_grad()
def term_decomposition_v13(model, base):
    """R2.term_decomposition with terms = ADAPTED per-source reads
    t_i = s_i + (s_i V)U^T (scalar-coefficient reconstruction does not apply)."""
    MID = R2.MID
    HELD, T, D_ = Q.HELD, Q.T, model.wte.weight.shape[1]
    r = {}
    b = HELD[:16, :T]
    col = {'entry_norm': [], 'attn_write': [], 'mlp_write': [], 'entry': []}
    model(b, collect=col)
    e = F.rms_norm(model.wte(b), (D_,))
    streams = [e]
    for l in range(DEPTH):
        streams.append(col['attn_write'][l])
        streams.append(col['mlp_write'][l])
    h_true = col['entry'][MID]
    idxs = model.vis[MID]
    terms = [model.adapt(MID, si, streams[si].float()) for si in idxs]
    recon = sum(terms)
    rel = float((recon - h_true).norm() / (h_true.norm() + 1e-12))
    r['n_terms'] = len(idxs)
    r['term_names'] = [R2.stream_name(si) for si in idxs]
    r['reassembly_rel_err'] = rel
    r['exact'] = bool(rel < 1e-5)
    en = torch.tensor([float(t.pow(2).sum()) for t in terms])
    order = torch.argsort(en, descending=True).tolist()
    k95 = len(idxs)
    for k in range(1, len(idxs) + 1):
        part = sum(terms[j] for j in order[:k])
        ve = 1 - float((part - h_true).pow(2).sum() / h_true.pow(2).sum())
        if ve >= 0.95:
            k95 = k
            break
    r['k95_energy'] = k95
    r['term_energy_ranked'] = [[R2.stream_name(idxs[j]), round(float(en[j]), 2)]
                               for j in order]
    keep = [idxs[j] for j in order[:k95]]

    def override_topk(i0):
        b2 = HELD[i0:i0 + 8, :T]
        col2 = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
        model(b2, collect=col2)
        e2 = F.rms_norm(model.wte(b2), (D_,))
        st = [e2]
        for l in range(DEPTH):
            st.append(col2['attn_write'][l])
            st.append(col2['mlp_write'][l])
        h = None
        for si in idxs:
            if si not in keep:
                continue
            t = model.adapt(MID, si, st[si].float())
            h = t if h is None else h + t
        return {MID: h}

    ce_topk = R2.ce_with(model, entry_override_fn=override_topk)
    hsum, n = torch.zeros(D_, device=DEV, dtype=torch.float64), 0
    for i in range(0, R2.ABL_N, 8):
        b2 = HELD[i:i + 8, :T]
        col2 = {'entry_norm': [], 'attn_write': [], 'mlp_write': [], 'entry': []}
        model(b2, collect=col2)
        hsum += col2['entry'][MID].double().sum((0, 1))
        n += b2.numel()
    hmean = (hsum / n).float()

    def override_mean(i0):
        B = HELD[i0:i0 + 8].shape[0]
        return {MID: hmean[None, None, :].expand(B, T, D_)}

    ce_floor = R2.ce_with(model, entry_override_fn=override_mean)
    d_topk, d_floor = ce_topk - base, ce_floor - base
    r['dce_topk95'] = round(d_topk, 5)
    r['dce_entry_mean_floor'] = round(d_floor, 5)
    r['topk95_recovered'] = (round(1 - d_topk / d_floor, 4)
                             if d_floor > 1e-6 else None)
    return r
