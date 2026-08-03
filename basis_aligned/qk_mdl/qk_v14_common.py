"""V14 (Logan update seven): WHY does the windowed model build a mid-stack token
relay? Hypothesis: the token line exists FOR ATTENTION -- the model wants
current-token identity chiefly to form queries/keys/values; in the windowed
model the MLPs relay it because attention has no other way to get it.

Two variants on the windowed N=6 base (width 384, depth 12, machinery verbatim
from qk_window_train.py; matched controls = the existing qk_window_N6 model,
held CE 5.7247, and qk_window_vanilla):

  V14a  N=6 window + VALUE-LERP to block-0 values: bilin18's carried value
        vector restored -- block 0 caches its per-head value tensor v0; every
        later block uses v = (1 - lamb_{l,h}) * v_l + lamb_{l,h} * v0 with one
        learned lamb per (block, head), init 0.5. Positive control: lamb == 0
        reproduces the plain N6 forward exactly at init.
  V14b  N=6 window + ATTENTION-ONLY token line: the rms-normed embedding is
        never part of any entry stream sum; it is added ONLY into the
        attention read (h_att = rms_norm(x_entry + e_norm) at EVERY block).
        The mlp path reads rms_norm(x_entry + a_l) with no embedding, the
        readout reads the last-6 writes only. Positive control: an as_n6 debug
        flag reroutes the embedding per the N6 rule (entry sums for l < 6, no
        attention line) and must reproduce the plain N6 forward exactly.

Score: does the mid-stack token-determined fraction of the mlp writes collapse
from the windowed ~0.98 toward vanilla's ~0.52 (relay dissolved because
attention gets the token directly), and does CE hold vs plain N6?
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_window_train as QW

QK = Q.QK
DEV = 'cuda'
DEPTH = QW.DEPTH
WINDOW = 6
V, T = Q.V, Q.T

# shared-GPU guard: patient waits (the v13 set / w1152 gate may still hold the GPU)
if not hasattr(Q, '_v14_guard_wrapped'):
    Q._v14_guard_wrapped = True
    _orig_guard = Q.gpu_guard

    def _patient_guard(min_free=4500, tries=45, sleep=20):
        _orig_guard(min_free=min(min_free, 6500), tries=360, sleep=60)
    Q.gpu_guard = _patient_guard


class V14aMini(QW.WindowMini):
    """N=6 window + per-(block, head) value-lerp to block-0 values."""

    def __init__(self, depth=DEPTH, window=WINDOW):
        super().__init__(depth, window)
        # AFTER matched construction (no extra RNG): lamb init 0.5
        self.lamb = nn.Parameter(torch.full((depth, self.NH), 0.5))
        self.variant = 'V14a'

    def forward(self, idx, collect=None, mlp_sub=None, lamb_zero=False,
                v0_sub=None):
        B, Tq = idx.shape
        Dm, NHm, HDm = self.D, self.NH, self.HD
        e_norm = F.rms_norm(self.wte(idx), (Dm,))
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        writes, v0 = [], None
        for l, blk in enumerate(self.h):
            parts = self.entry_parts(l, e_norm, writes)
            x = parts[0]
            for p in parts[1:]:
                x = x + p
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach().float().cpu())
            h = F.rms_norm(x, (Dm,))

            def qkf(lin):
                z = lin(h).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qkf(blk.c_q), qkf(blk.c_k)
            q2, k2 = qkf(blk.c_q2), qkf(blk.c_k2)
            v = blk.c_v(h).view(B, Tq, NHm, HDm)
            if l == 0:
                v0 = v if v0_sub is None else v0_sub.to(v.dtype).expand_as(v)
            elif not lamb_zero:
                lam = self.lamb[l].view(1, 1, NHm, 1).to(v.dtype)
                v = (1 - lam) * v + lam * v0
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                hn = F.rms_norm(x, (Dm,))
                mw = blk.Down(blk.Left(hn) * blk.Right(hn)) + blk.Down_bias
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            writes.append((aw, mw))
        parts = self.entry_parts(self.depth, e_norm, writes)
        x = parts[0]
        for p in parts[1:]:
            x = x + p
        x = F.rms_norm(x, (Dm,))
        return 30 * torch.tanh((x @ self.wte.weight.t()) / 30)


class V14bMini(QW.WindowMini):
    """N=6 window, embedding routed ONLY into the attention read at every block
    (never into entry sums / the mlp path / the readout)."""

    def __init__(self, depth=DEPTH, window=WINDOW):
        super().__init__(depth, window)
        self.variant = 'V14b'
        self.as_n6 = False                     # debug: reproduce plain N6

    def entry_parts(self, l, e_norm, writes):
        if self.as_n6:
            return super().entry_parts(l, e_norm, writes)
        parts = []                             # NO embedding, ever
        for j in range(max(0, l - self.window), l):
            parts.append(writes[j][0])
            parts.append(writes[j][1])
        return parts

    def forward(self, idx, collect=None, mlp_sub=None, emb_att_sub=None):
        """emb_att_sub: {block: tensor} replaces e_norm in that block's
        ATTENTION LINE only (ablation hook)."""
        B, Tq = idx.shape
        Dm, NHm, HDm = self.D, self.NH, self.HD
        e_norm = F.rms_norm(self.wte(idx), (Dm,))
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        writes = []
        for l, blk in enumerate(self.h):
            parts = self.entry_parts(l, e_norm, writes)
            if parts:
                x = parts[0]
                for p in parts[1:]:
                    x = x + p
            else:
                x = torch.zeros_like(e_norm)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach().float().cpu())
            if self.as_n6:
                h = F.rms_norm(x, (Dm,))
            else:
                e_att = e_norm
                if emb_att_sub is not None and l in emb_att_sub:
                    e_att = emb_att_sub[l]
                h = F.rms_norm(x + e_att, (Dm,))

            def qkf(lin):
                z = lin(h).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qkf(blk.c_q), qkf(blk.c_k)
            q2, k2 = qkf(blk.c_q2), qkf(blk.c_k2)
            v = blk.c_v(h).view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                hn = F.rms_norm(x, (Dm,))
                mw = blk.Down(blk.Left(hn) * blk.Right(hn)) + blk.Down_bias
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            writes.append((aw, mw))
        parts = self.entry_parts(self.depth, e_norm, writes)
        x = parts[0]
        for p in parts[1:]:
            x = x + p
        x = F.rms_norm(x, (Dm,))
        return 30 * torch.tanh((x @ self.wte.weight.t()) / 30)


def make_v14(which):
    torch.manual_seed(Q.SEED)
    cls = V14aMini if which == 'a' else V14bMini
    return cls(DEPTH, WINDOW).to(DEV)


def load_v14(which):
    ck = torch.load(f'{QK}/qk_v14{which}.pt', map_location=DEV,
                    weights_only=False)
    m = make_v14(which)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


@torch.no_grad()
def v14_controls():
    idx = Q.HELD[:2, :T]
    ref = QW.make_window(DEPTH, WINDOW).eval().float()      # plain N6 at init
    out_ref = ref(idx)
    del ref
    torch.cuda.empty_cache()
    ma = make_v14('a').eval().float()
    out_a = ma(idx, lamb_zero=True)
    d = (out_a - out_ref).abs().max().item()
    print(f"control V14a(lamb=0)==N6(init): max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del ma, out_a
    torch.cuda.empty_cache()
    mb = make_v14('b').eval().float()
    mb.as_n6 = True
    out_b = mb(idx)
    d = (out_b - out_ref).abs().max().item()
    print(f"control V14b(as_n6)==N6(init): max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    mb.as_n6 = False
    del mb, out_b, out_ref
    torch.cuda.empty_cache()
