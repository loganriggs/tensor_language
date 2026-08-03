"""Shared machinery for the SMALL-MODEL (width 264) V10/V11/V12 set (Logan's
2026-08-03 directives, updates two/three/four). Runs ALONGSIDE the width-768
window job on a shared GPU, so everything is sized to stay under ~6 GiB.

Config: depth 12, width 264 (24 slots x 11 dims = 264 exactly), NH = 6 heads of
head-dim 44 (the HD=64 convention cannot divide 264; noted as a deviation),
FineWeb cooc corpus train [0:5500] held [5500:6000], 6-epoch budget, batch from
pre-flight (8, fallback 4), SINGLE lr 0.002 for every arm (Logan authorized
skipping the sweep for speed), group-lasso 1e-4 on reads, nonzero write init for
all slotted arms, matched vanilla control at the same width.

Arms (all trained by qk_v10v11_train.py):
  VC     matched vanilla control: V8Route with sum kernel, full visibility, no
         slot projections, zero-init writes -- provably equal to variant-A
         MiniBilin at width 264 (checked).
  V10    READALL: slots + group-lasso + N=6 window for blocks, but the READOUT
         reads ALL 24 module writes (no embedding -- the readout consumes module
         writes only, per Logan's proposal a).
  V11    V10 + a learned affine decoder per module into the readout: readout
         input = sum_i (A_i w_i + b_i), A_i full d x d, initialized as the
         identity ON THE MODULE'S SLOT COLUMNS (off-slot columns zero, so the
         init is function-identical to V10 and carries no dead identity mass);
         group-lasso 1e-4 on each ||A_i||_F (unused decoders die visibly).
  V11nl  V11 WITHOUT the decoder lasso (read-weight lasso kept) -- isolates
         whether the decoder lasso matters. (Note: weight decay 0.1 still acts
         on A_i in both, as it does on every matrix.)
  V11lr  LOW-RANK decoders: A_i = U_i @ V_i^T with U_i, V_i of shape (d, r),
         r = 32, applied WITHOUT materializing A_i ((w @ V_i) @ U_i^T). Init:
         first 11 channels = the slot identity (U and V both carry the slot
         basis), channels 11..31 get V ~ noise / U = 0 (exact V10 function at
         init, no dead-channel fixed point since grad U_c != 0). Lasso on the
         product Frobenius norm computed from factors:
         ||U V^T||_F = sqrt(tr((U^T U)(V^T V))).
  (V12 -- constant read scales -- was CANCELLED by Logan update five before
  training: RMSNorm can be handled inside the tensor network by gauge/dummy-bond
  treatment, so removing it architecturally is unnecessary.)
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_deeproute_train_2 as R2
import qk_v9_common as C
from qk_deeproute_train import DEPTH

QK = Q.QK
DEV = 'cuda'
WIDTH = 264
NHW, HDW = 6, 44                 # 6 x 44 = 264 (HD=64 convention impossible here)
SUB = WIDTH // (2 * DEPTH)       # 11
RANK = 32                        # V11lr decoder rank
LR = 0.002                       # single lr for every arm (Logan: skip the sweep)
EPOCHS = 6
GC = C.GROUP_COEFF               # 1e-4
V, T = Q.V, Q.T


def patch_width():
    Q.D, Q.NH, Q.HD = WIDTH, NHW, HDW
    for mod in (R, V8T):
        mod.D, mod.NH, mod.HD, mod.SUBDIM = WIDTH, NHW, HDW, SUB
    R2.D, R2.SUBDIM = WIDTH, SUB
    assert SUB * 2 * DEPTH == WIDTH and NHW * HDW == WIDTH
    print(f"width patched: D {WIDTH}, NH {NHW}, HD {HDW}, slot dim {SUB}",
          flush=True)


def patch_batch(b):
    Q.BATCH = b
    Q.STEPS_PER_EPOCH = Q.NTR // b
    V8T.STEPS = EPOCHS * Q.STEPS_PER_EPOCH


# GPU is SHARED with the width-768 window job: never demand more than 5200 MiB
# free (the 264-width runs peak well under that), and wait patiently (60 s polls).
_orig_guard = Q.gpu_guard
def _shared_guard(min_free=4500, tries=45, sleep=20):
    _orig_guard(min_free=min(min_free, 5200), tries=360, sleep=60)
Q.gpu_guard = _shared_guard


# ---------------- visibility ----------------
def vis_v10(li):
    """Blocks: N=6 window (embedding visible only l<6). Readout: ALL 24 writes."""
    if li == DEPTH:
        return list(range(1, 2 * DEPTH + 1))
    return C.window_vis(li, N=6)


FULL_VIS = [list(range(1 + 2 * min(li, DEPTH))) for li in range(DEPTH + 1)]


# ---------------- factories ----------------
def make_control():
    """Vanilla control: sum kernel, full visibility, no projections, zero-init
    writes == variant-A MiniBilin at width 264 (control checks it)."""
    R.KERNEL['VC'] = 'sum'
    torch.manual_seed(Q.SEED)
    return V8T.V8Route('VC', DEPTH).to(DEV)


def make_v10():
    return C.make_variant('V10', vis_v10)


class V11Route(V8T.V8Route):
    """V10 + per-module affine decoders feeding the readout. Blocks use the
    inherited sum assembly; only the readout (li == depth) differs."""

    def __init__(self, variant, depth, dec_lasso=True):
        super().__init__(variant, depth)
        Dm = self.wte.weight.shape[1]
        self.dec = nn.ModuleList([nn.Linear(Dm, Dm) for _ in range(2 * depth)])
        with torch.no_grad():
            for k, lin in enumerate(self.dec):
                lin.weight.zero_()
                lin.bias.zero_()
                for c in range(SUB):                       # identity on slot cols
                    lin.weight[SUB * k + c, SUB * k + c] = 1.0
        self.dec_lasso = dec_lasso

    def dec_apply(self, k, s):
        return self.dec[k](s)

    def dec_penalty(self):
        tot = None
        for lin in self.dec:
            g = (lin.weight.pow(2).sum() + 1e-12).sqrt()
            tot = g if tot is None else tot + g
        return tot

    def assemble(self, li, streams, sub=None, coef_out=None):
        if li < self.depth:
            return super().assemble(li, streams, sub, coef_out)
        idxs = self.vis[li]
        get = lambda i: (sub[i] if (sub is not None and i in sub) else streams[i])
        h, cs = None, []
        for i in idxs:
            s = get(i)
            t = s if i == 0 else self.dec_apply(i - 1, s)   # emb (control only): raw
            h = t if h is None else h + t
            if coef_out is not None:
                cs.append(t.detach().float().norm(dim=-1)
                          / (s.detach().float().norm(dim=-1) + 1e-12))
        if coef_out is not None:
            coef_out[li] = (idxs, torch.stack(cs, -1))
        return h


class V11LRRoute(V11Route):
    """V11 with rank-32 factored decoders A_i = U_i V_i^T, never materialized in
    the forward: t = (s @ V_i) @ U_i^T + b_i."""

    def __init__(self, variant, depth, dec_lasso=True):
        V8T.V8Route.__init__(self, variant, depth)         # skip V11Route init
        Dm = self.wte.weight.shape[1]
        gen = torch.Generator().manual_seed(999)
        U, Vf, B = [], [], []
        for k in range(2 * depth):
            u = torch.zeros(Dm, RANK)
            v = torch.zeros(Dm, RANK)
            for c in range(SUB):                           # slot identity, rank 11
                u[SUB * k + c, c] = 1.0
                v[SUB * k + c, c] = 1.0
            # extra channels: V random / U zero -> exact V10 function at init,
            # no dead-channel fixed point (grad U_c propto (w.V_c) != 0)
            v[:, SUB:] = torch.randn(Dm, RANK - SUB, generator=gen) * 0.02
            U.append(nn.Parameter(u))
            Vf.append(nn.Parameter(v))
            B.append(nn.Parameter(torch.zeros(Dm)))
        self.dec_U = nn.ParameterList(U)
        self.dec_V = nn.ParameterList(Vf)
        self.dec_b = nn.ParameterList(B)
        self.dec_lasso = dec_lasso

    def dec_apply(self, k, s):
        return (s @ self.dec_V[k].to(s.dtype)) @ self.dec_U[k].to(s.dtype).t() \
            + self.dec_b[k].to(s.dtype)

    def dec_penalty(self):
        tot = None
        for k in range(len(self.dec_U)):
            Uk, Vk = self.dec_U[k], self.dec_V[k]
            g = ((Uk.t() @ Uk) * (Vk.t() @ Vk)).sum().clamp_min(1e-12).sqrt()
            tot = g if tot is None else tot + g            # = ||U V^T||_F
        return tot

    def dec_matrix(self, k):
        """Materialized A_k for ANALYSIS only (probe), fp32."""
        return (self.dec_U[k].float() @ self.dec_V[k].float().t()).detach()


def make_v11(variant='V11', dec_lasso=True, cls=V11Route):
    C.register(variant)
    torch.manual_seed(Q.SEED)
    m = cls(variant, DEPTH, dec_lasso=dec_lasso).to(DEV)
    m.vis = [vis_v10(li) for li in range(DEPTH + 1)]
    return m


# ---------------- penalty patch (read lasso + optional decoder lasso) ----------------
if not hasattr(V8T, '_orig_group_penalty'):
    V8T._orig_group_penalty = V8T.group_penalty

    def _patched_penalty(model):
        tot = V8T._orig_group_penalty(model)
        if getattr(model, 'dec_lasso', False):
            tot = tot + model.dec_penalty()
        return tot
    V8T.group_penalty = _patched_penalty


# ---------------- loaders ----------------
def load_arm(stem, variant, kind):
    ck = torch.load(f'{QK}/{stem}.pt', map_location=DEV, weights_only=False)
    if kind == 'control':
        m = make_control()
    elif kind == 'v10':
        m = make_v10()
    elif kind == 'v11':
        m = make_v11(variant, dec_lasso=False, cls=V11Route)
    elif kind == 'v11lr':
        m = make_v11(variant, dec_lasso=False, cls=V11LRRoute)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


# ---------------- positive controls ----------------
@torch.no_grad()
def ref_vanilla264():
    Q.NL = DEPTH
    return Q.make_model('A').eval().float()


@torch.no_grad()
def run_controls():
    idx = Q.HELD[:2, :T]
    ma = ref_vanilla264()
    ref = ma(idx)
    del ma
    torch.cuda.empty_cache()

    # (1) the matched control IS vanilla-A at width 264
    m = make_control().eval().float()
    d = (m(idx) - ref).abs().max().item()
    print(f"control VC==A264 at init: max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del m
    torch.cuda.empty_cache()

    # (2) penalty vectorization at slot 11
    m = make_v10().eval().float()
    p_fast = float(V8T._orig_group_penalty(m))
    p_naive = 0.0
    for blk in m.h:
        for M in V8T.READ_MATS(blk):
            for k in range(2 * DEPTH):
                p_naive += float(M[:, SUB * k:SUB * (k + 1)].pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel:.2e}", flush=True)
    assert rel < 1e-6

    # (3) V10 vis table spot checks
    v = [vis_v10(li) for li in range(DEPTH + 1)]
    assert v[0] == [0] and v[3] == [0, 1, 2, 3, 4, 5, 6]
    assert v[6] == list(range(1, 13)) and v[11] == list(range(11, 23))
    assert v[12] == list(range(1, 25))
    print("V10 visibility table matches spec (blocks window-6, readout all 24 "
          "writes, no embedding)", flush=True)

    # (4) V10 identity control (full vis + identity proj + zero writes == A264)
    mm = make_v10().eval().float()
    mm.vis = [list(fv) for fv in FULL_VIS]
    mm.wmask.fill_(1.0)
    for blk in mm.h:
        blk.c_proj.weight.zero_()
        blk.Down.weight.zero_()
    d = (mm(idx) - ref).abs().max().item()
    print(f"control V10(full vis, identity proj, zero writes)==A264: "
          f"max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del mm
    torch.cuda.empty_cache()

    # (5) V11 / V11lr at init == V10 at init (decoders are slot identities)
    m10 = make_v10().eval().float()
    out10 = m10(idx)
    for variant, cls in (('V11', V11Route), ('V11lr', V11LRRoute)):
        mv = make_v11(variant, cls=cls).eval().float()
        d = (mv(idx) - out10).abs().max().item()
        print(f"control {variant}(init)==V10(init): max |logit diff| {d:.2e}",
              flush=True)
        assert d < 1e-3
        del mv
        torch.cuda.empty_cache()
    del m10, out10

    # (6) V11 identity control against vanilla (full vis incl. embedding
    #     passthrough at the readout, identity proj, zero writes)
    mv = make_v11('V11', cls=V11Route).eval().float()
    mv.vis = [list(fv) for fv in FULL_VIS]
    mv.wmask.fill_(1.0)
    for blk in mv.h:
        blk.c_proj.weight.zero_()
        blk.Down.weight.zero_()
    d = (mv(idx) - ref).abs().max().item()
    print(f"control V11(full vis, identity everything, zero writes)==A264: "
          f"max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del mv

    del ref
    torch.cuda.empty_cache()


# ---------------- shared bookkeeping ----------------
def param_counts(model):
    tot = sum(p.numel() for p in model.parameters())
    emb = model.wte.weight.numel()
    dec = 0
    for nm, p in model.named_parameters():
        if nm.startswith('dec') or nm.startswith('rs_'):
            dec += p.numel()
    return {'total': tot, 'embedding': emb, 'body': tot - emb,
            'added_vs_base': dec}
