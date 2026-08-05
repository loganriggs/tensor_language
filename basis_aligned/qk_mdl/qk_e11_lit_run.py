"""E11 LITERATURE-DERIVED ARMS (Logan approval 2026-08-05; fresh single-epoch
batch-16, all paired vs E9a (recipe reference: per-slot norm + Muon lr 0.02 +
in-loss lasso 3e-5), E0a and E0b).

E11a -- SVFORMER-STYLE SHARED VALUES on the E9a recipe: per-layer value
  projections are DELETED (blocks 1..11); every attention layer applies its
  own pattern (s1*s2 score projections retained per block) to the SAME value
  vectors, computed once by block 0's value projection from block 0's entry
  (v(j) = W_v^0(per-slot-norm(block-0 input at j)), per-head reshape as
  usual). Writes stay slotted through each block's own c_proj. JSON note:
  bilin18's a.lamb value-lerp was the soft version of this; E11a is the full
  commitment. Params saved: 11 x 264 x 264 = 766,656. Positive control: with
  keep_local values and the shared path disabled the forward equals the E9a
  architecture exactly at init. Probes: wiring + token-determined (prediction:
  attention writes go ~fully token-determined; the mid-stack MLP relay may
  shrink -- both reported).

E11b -- SINKHORN-CONSTRAINED SOURCE ROUTING on the E9a recipe: hard write
  slots kept; a learned routing table over sources is added for the 24
  module-consumers (block l's attention read = consumer 2l, its MLP read =
  consumer 2l+1; the readout stays unrouted). theta in R^{24x25} (sources =
  24 writes + embedding as the 25th column), P = exp(theta) pushed through 5
  Sinkhorn-Knopp row/column steps. RECTANGULAR CONVENTION (documented): row
  targets 1 (each consumer's read budget), column targets 24/25 (each
  source's influence budget; totals match), so uniform P is a fixed point.
  The applied scale is S(theta)/S(0) with S(0) a frozen buffer computed by
  the identical code path at init -- elementwise y/y makes the init scale
  EXACTLY 1.0 (bit-for-bit identity control vs E9a), and S(theta)/S(0)
  equals n_sources * G up to the fixed uniform normalization the spec's
  n_src*G form intended. The in-loss lasso 3e-5 stays on (routing composes
  with it); theta trains under AdamW-no-decay (excluded from Muon). Probes:
  wiring group-norm Spearman AND the routing matrix vs causal consumption
  (two readability channels), token-determined; the routing rows' entropies
  are logged every 500 training steps (V4-style pathology watch -- the
  nonnegative parameterization structurally forbids cancellation, verified
  by reporting).

E11c -- WEIGHTS-ONLY DETOKENIZATION PROBE (probe-only, last; mirrors arXiv
  2501.15754's weights-only readout) on the E9a checkpoint: from weights
  alone, each attention module's token affinity is predicted as
  ||W_v(slot_norm(e_t))|| * sqrt(||W_k(.)|| * ||W_k2(.)||) over the vocab;
  checked by rank-correlation against activation-harvested per-token source
  contributions (column-sum of |pattern| times value norm, 32 held
  sequences, tokens with >= 3 occurrences). Per-module Spearman + top-10
  weight-predicted tokens reported.

Standard conventions: identity controls, curves in JSON, idempotent,
non-blocking guards. Results -> qk_e11.json."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math

import qk_e_common as E
from qk_e_common import Q, V8T, C, R2, DEPTH, F, nn, torch
import qk_e1_slotnorm_run as E1R
import qk_e7_evenout_run as E7R

JP = E.jpath('qk_e11.json')
GC11 = 3e-5
N_CONS = 2 * DEPTH
N_SRC = 2 * DEPTH + 1                # 24 writes + embedding (last column)

# dispatch: models may carry a custom_group_penalty (E11a's c_v-less blocks)
if not hasattr(V8T, '_e11_penalty_patch'):
    V8T._e11_penalty_patch = True
    _prev_pen11 = V8T.group_penalty

    def _e11_dispatch(model):
        if hasattr(model, 'custom_group_penalty'):
            return model.custom_group_penalty()
        return _prev_pen11(model)
    V8T.group_penalty = _e11_dispatch


# ---------------- E11a: shared values ----------------
class E11aRoute(E1R.E1Route):
    """Per-slot-norm base with block-0-shared attention values. keep_local
    retains the per-layer c_v modules; use_shared=False + keep_local=True is
    the identity-control mode (exact E9a-architecture forward)."""
    use_shared = True

    def __init__(self, variant, depth, keep_local=False):
        super().__init__(variant, depth)
        self.keep_local = keep_local
        if not keep_local:
            for l in range(1, depth):
                self.h[l].c_v = None

    def custom_group_penalty(self):
        S = self.wte.weight.shape[1] // E.NGROUP
        tot = None
        for blk in self.h:
            for nm in E.READ_NAMES:
                lin = getattr(blk, nm, None)
                if lin is None:
                    continue
                M = lin.weight
                g = (M.pow(2).view(M.shape[0], E.NGROUP, S).sum(dim=(0, 2))
                     + 1e-12).sqrt().sum()
                tot = g if tot is None else tot + g
        return tot

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        e = F.rms_norm(self.wte(idx), (Dm,))
        streams = [e]
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        v_shared = None
        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x)

            def qk(lin):
                z = lin(hn).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            if l == 0:
                v_shared = blk.c_v(hn).view(B, Tq, NHm, HDm)
            if self.use_shared or l == 0:
                v = v_shared
            else:
                v = blk.c_v(hn).view(B, Tq, NHm, HDm)   # control mode only
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = self.slot_norm(x)
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e11a(keep_local=False):
    C.register('E11a')
    torch.manual_seed(Q.SEED)
    m = E11aRoute('E11a', DEPTH, keep_local=keep_local).to(E.DEV)
    m.norm_groups = E.NGROUP
    return m


# ---------------- E11b: Sinkhorn source routing ----------------
def sinkhorn(theta, iters=5):
    P = torch.exp(theta)
    c_t = theta.shape[0] / theta.shape[1]          # 24/25 column target
    for _ in range(iters):
        P = P / P.sum(1, keepdim=True)             # rows -> 1
        P = P / P.sum(0, keepdim=True) * c_t       # cols -> 24/25
    return P


class E11bRoute(E1R.E1Route):
    """Per-slot-norm base + per-module-consumer Sinkhorn source routing."""

    def __init__(self, variant, depth):
        super().__init__(variant, depth)
        self.route_theta = nn.Parameter(torch.zeros(N_CONS, N_SRC))
        self.muon_exclude = ('route_theta',)
        with torch.no_grad():
            self.register_buffer('route_S0',
                                 sinkhorn(self.route_theta.detach().float()))

    def route_scale(self):
        return sinkhorn(self.route_theta.float()) / self.route_S0

    @torch.no_grad()
    def route_entropy(self):
        Ssc = sinkhorn(self.route_theta.detach().float())
        p = Ssc / Ssc.sum(1, keepdim=True)
        return (-(p * (p + 1e-12).log()).sum(1)).tolist()

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        e = F.rms_norm(self.wte(idx), (Dm,))
        streams = [e]
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        scale = self.route_scale()

        def routed(li, cons):
            """Consumer `cons`'s routed read over block li's visible sources
            (sub_entry mean-ablation supported on every stream)."""
            sub = sub_entry.get(li) if sub_entry is not None else None
            get = lambda i: (sub[i] if (sub is not None and i in sub)
                             else streams[i])
            tot = None
            for i in self.vis[li]:
                col = N_SRC - 1 if i == 0 else i - 1
                s = get(i)
                t = scale[cons, col].to(s.dtype) * s
                tot = t if tot is None else tot + t
            return tot

        for l, blk in enumerate(self.h):
            if entry_override is not None and l in entry_override:
                x = entry_override[l]
            else:
                x = routed(l, 2 * l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x)

            def qk(lin):
                z = lin(hn).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xm = routed(l, 2 * l + 1) + aw       # the MLP consumer's read
                xn = self.slot_norm(xm)
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        # readout: UNROUTED standard sum
        sub = sub_entry.get(self.depth) if sub_entry is not None else None
        get = lambda i: (sub[i] if (sub is not None and i in sub)
                         else streams[i])
        x = None
        for i in self.vis[self.depth]:
            x = get(i) if x is None else x + get(i)
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e11b():
    C.register('E11b')
    torch.manual_seed(Q.SEED)
    m = E11bRoute('E11b', DEPTH).to(E.DEV)
    m.norm_groups = E.NGROUP
    return m


# ---------------- controls ----------------
@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    ref = E7R.make_e7m1().eval().float()
    out_ref = ref(idx)
    # E11a control: local values + shared path off == E9a arch exactly
    ma = make_e11a(keep_local=True).eval().float()
    ma.use_shared = False
    d = (ma(idx) - out_ref).abs().max().item()
    print(f"control E11a(local values)==E9a-arch at init: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert (d == 0.0) if E.SMOKE else (d < 1e-6)
    ma.use_shared = True
    d2 = (ma(idx) - out_ref).abs().max().item()
    print(f"sanity E11a shared values differ: max diff {d2:.2e}", flush=True)
    assert d2 > 1e-6
    del ma
    # E11a penalty naive check (c_v-less blocks)
    m = make_e11a().eval().float()
    p_fast = float(m.custom_group_penalty())
    S = Q.D // E.NGROUP
    p_naive = 0.0
    for blk in m.h:
        for nm in E.READ_NAMES:
            lin = getattr(blk, nm, None)
            if lin is None:
                continue
            for k in range(E.NGROUP):
                p_naive += float(
                    lin.weight[:, S * k:S * (k + 1)].pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control E11a penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel:.2e}", flush=True)
    assert rel < 1e-6
    del m
    torch.cuda.empty_cache()
    # E11b control: Sinkhorn scale exactly 1 at init -> bit-for-bit vs E9a
    mb = make_e11b().eval().float()
    sc = mb.route_scale()
    ds = float((sc - 1.0).abs().max())
    d3 = (mb(idx) - out_ref).abs().max().item()
    print(f"control E11b init: max |scale-1| {ds:.2e}; forward vs E9a-arch "
          f"max |logit diff| {d3:.2e}", flush=True)
    assert ds == 0.0                             # y/y is exactly 1.0
    assert (d3 == 0.0) if E.SMOKE else (d3 < 1e-6)
    del mb, ref, out_ref
    torch.cuda.empty_cache()


# ---------------- E11c: weights-only detokenization probe ----------------
def e11c_detok():
    if E.SMOKE:
        return
    if 'E11c_detok' in E.loadj(JP) or not os.path.exists(E.ckpath('qk_e9_a')):
        print("E11c: done or qk_e9_a missing -- skip", flush=True)
        return
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    m, _ = E.load_arm('qk_e9_a', E7R.make_e7m1)
    Dm = Q.D
    with torch.no_grad():
        ehat = m.slot_norm(F.rms_norm(m.wte.weight.float(), (Dm,)))
        wpred = []
        for blk in m.h:
            av = (ehat @ blk.c_v.weight.float().t()).norm(dim=1)
            ak = (ehat @ blk.c_k.weight.float().t()).norm(dim=1)
            ak2 = (ehat @ blk.c_k2.weight.float().t()).norm(dim=1)
            wpred.append(av * (ak * ak2).sqrt())     # (V,) per module
        # activation harvest: colsum(|pattern|) * ||v(pos)|| per token
        Vn = Q.V
        hsum = torch.zeros(DEPTH, Vn, dtype=torch.float64)
        hcnt = torch.zeros(Vn, dtype=torch.float64)
        NHm, HDm = Q.NH, Q.HD
        for i0 in range(0, 32, 8):
            b = Q.HELD[i0:i0 + 8, :Q.T]
            col = {'entry': [], 'entry_norm': [], 'attn_write': [],
                   'mlp_write': []}
            m(b, collect=col)
            B, Tq = b.shape
            cos = m.cos[None, :Tq, None, :]
            sin = m.sin[None, :Tq, None, :]
            msk = m.mask[:Tq, :Tq]
            ids = b.reshape(-1).cpu()
            for l, blk in enumerate(m.h):
                hn = m.slot_norm(col['entry'][l].float())

                def qk(lin):
                    z = lin(hn).view(B, Tq, NHm, HDm)
                    return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

                q, k = qk(blk.c_q), qk(blk.c_k)
                q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
                v = blk.c_v(hn).view(B, Tq, NHm, HDm)
                s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
                s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
                pat = (s1 * s2).masked_fill(~msk, 0.0)
                contrib = (pat.abs().sum(dim=(1, 2))
                           * v.norm(dim=-1).mean(-1)).reshape(-1)
                hsum[l].index_add_(0, ids, contrib.double().cpu())
                if l == 0:
                    hcnt.index_add_(0, ids, torch.ones(len(ids),
                                                       dtype=torch.float64))
            del col
            torch.cuda.empty_cache()
        covered = (hcnt >= 3).nonzero().squeeze(1)
        res = {'n_tokens_covered': int(len(covered)),
               'harvest': '32 held seqs; contribution = colsum|pattern| * '
                          '||v||; tokens with >= 3 occurrences',
               'weight_score': '||W_v slot_norm(e_t)|| * sqrt(||W_k .|| * '
                               '||W_k2 .||)',
               'per_module': []}
        for l in range(DEPTH):
            harv = (hsum[l][covered] / hcnt[covered]).numpy()
            pred = wpred[l][covered].cpu().numpy()
            rho = R2.spearman(pred, harv)
            top = torch.topk(wpred[l], 10).indices.tolist()
            res['per_module'].append({
                'block': l, 'spearman_weights_vs_harvest': round(rho, 4),
                'top10_weight_predicted': [tok.decode([t]) for t in top]})
            print(f"  E11c block{l}: spearman {rho:.4f}", flush=True)
        E.merge(JP, 'E11c_detok', res)
    del m
    torch.cuda.empty_cache()


def pair_extra(stem, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


if __name__ == '__main__':
    E.setup()
    controls()
    mlr = E7R.muon_lr()

    # ---- E11a ----
    m = make_e11a()
    import qk_v10v11_common as W
    E.merge(JP, 'param_counts_E11a', dict(
        W.param_counts(m),
        params_saved_vs_e9a=11 * Q.D * Q.D,
        note='bilin18 a.lamb value-lerp was the soft version; this is the '
             'full commitment (blocks 1..11 have no value projection)'))
    del m
    torch.cuda.empty_cache()
    E.train_arm('qk_e11_a', JP, 'E11a', make_e11a, GC11, lr=mlr,
                trainer=lambda lr, gc, steps, **kw: E.train_muon(
                    lr, gc, steps, lr_adamw=E.get_lr(), **kw),
                extra={'optimizer': 'muon', 'design': 'SVFormer-style shared '
                       'values from block 0'})
    E.oldheld_record('qk_e11_a', make_e11a, JP, 'E11a_oldheld')
    E.paired_fresh('qk_e11_a', JP, 'E11a')
    pair_extra('qk_e11_a', 'E11a', (('qk_e9_a', 'e9a'),))
    E.probe_arm('qk_e11_a', make_e11a, JP, 'light_probe_E11a',
                tok_key='tok_probe_E11a')

    # ---- E11b ----
    ent_log = []

    def cb(step, model):
        with torch.no_grad():
            ent_log.append([step, [round(x, 4)
                                   for x in model.route_entropy()]])
    E.train_arm('qk_e11_b', JP, 'E11b', make_e11b, GC11, lr=mlr,
                trainer=lambda lr, gc, steps, **kw: E.train_muon(
                    lr, gc, steps, lr_adamw=E.get_lr(), step_cb=cb, **kw),
                extra={'optimizer': 'muon (routing theta on AdamW-no-decay)',
                       'design': 'Sinkhorn-constrained source routing, 24 '
                                 'consumers x 25 sources',
                       'sinkhorn': 'exp(theta), 5 row/col steps; rows -> 1, '
                                   'cols -> 24/25 (uniform is a fixed '
                                   'point); applied scale = S(theta)/S(0), '
                                   'exactly 1 at init'})
    if ent_log:
        E.merge(JP, 'E11b_routing_entropy_per_500', ent_log)
    E.oldheld_record('qk_e11_b', make_e11b, JP, 'E11b_oldheld')
    E.paired_fresh('qk_e11_b', JP, 'E11b')
    pair_extra('qk_e11_b', 'E11b', (('qk_e9_a', 'e9a'),))
    E.probe_arm('qk_e11_b', make_e11b, JP, 'light_probe_E11b',
                tok_key='tok_probe_E11b')
    if not E.SMOKE and os.path.exists(E.ckpath('qk_e11_b')):
        out = E.loadj(JP)
        if 'E11b_routing_vs_causal' not in out \
                and 'light_probe_E11b' in out:
            m, _ = E.load_arm('qk_e11_b', make_e11b)
            with torch.no_grad():
                sc = m.route_scale().cpu()
            dce = out['light_probe_E11b']['consumption_matrix']
            pairs, route_v, cau_v = [], [], []
            for li_s, row in dce.items():
                li = int(li_s)
                if li >= DEPTH:
                    continue
                for si_s, v in row.items():
                    si = int(si_s)
                    colj = N_SRC - 1 if si == 0 else si - 1
                    mean_scale = float((sc[2 * li, colj]
                                        + sc[2 * li + 1, colj]) / 2)
                    route_v.append(mean_scale)
                    cau_v.append(v)
                    pairs.append([li, si])
            E.merge(JP, 'E11b_routing_vs_causal', {
                'spearman': round(R2.spearman(route_v, cau_v), 4),
                'n_pairs': len(pairs),
                'routing_matrix': [[round(float(x), 4) for x in r]
                                   for r in sc.tolist()],
                'final_row_entropies': [round(x, 4)
                                        for x in m.route_entropy()]})
            del m
            torch.cuda.empty_cache()

    # ---- E11c ----
    e11c_detok()
    print('e11 lit run done', flush=True)
