"""E10 EMBEDDING SPLIT -- two-channel reads (Logan 2026-08-05 approval; fresh
single-epoch batch-16 protocol).

Today every consumer input is x = e_normed + sum(slot writes), so a read
column group in slot k reads BOTH module k's write AND the embedding's shadow
in those dims -- entangled by construction. E10 splits the channels: each
consumer consumes the PAIR (writes_sum, e_normed), and every one of the seven
read matrices gains a second 264-column half reading the embedding channel.
Implementation note: rather than materially concatenating [W_slots | W_emb]
into one 528-column matrix, each read matrix keeps its ORIGINAL module (the
slots half, so the existing lasso/wiring machinery stays valid verbatim) and
gains a parallel `<name>_emb` Linear (the embedding half); the forward adds
the two halves' outputs, which is algebraically identical to the concatenated
matrix acting on the concatenated channels, and lets the identity control
reuse the exact same GEMM as the unsplit model (bit-for-bit assertable).

INIT / POSITIVE CONTROL: W_emb := W_slots := W_init (same draw). In control
mode the writes channel carries the unsplit normed input and the embedding
channel carries exact zeros, so slots-half(x) + emb-half(0) reproduces the
unsplit forward EXACTLY (asserted == 0 in CPU smoke; printed and asserted
< 1e-6 on GPU where kernel-path selection could differ in the last bit).

LASSO (weighting documented here and in the JSON): the 24 slot groups per
matrix stay UNWEIGHTED (the harness convention); the new 25th group per
matrix -- the 264-column embedding group -- is weighted by sqrt(264/11)
= sqrt(24) ~ 4.899 so it is not unfairly cheap per-parameter vs the 11-column
slot groups. Verified against a naive loop.

NORMS: per-slot RMSNorm on the writes channel (E10a; E10b uses the global
norm via norm_groups=1); ONE global RMSNorm on the embedding channel (it
arrives normed already -- rms_norm is idempotent on it, kept for safety).

ARMS: E10a (primary) = the current best recipe + split (per-slot norm + Muon
+ in-loss lasso 3e-5; the E9a twin). Wiring probe (slot groups now read PURE
write content -- compare Spearman vs E9a's 0.77/0.75), token-determined
probe, and the NEW readable quantity: the 84-entry "token appetite" table
(per block x per read matrix embedding-group norm, straight from weights).
E10b = the split on the AdamW slots+lasso 1e-4 reference (E0b + split;
readability reference 0.778/0.578). Params recorded exactly: the seven
embedding halves cost out_dim x 264 each -- 5 x (264x264) + 2 x (1056x264)
= 906,048 per block, 10,872,576 total (~+71 percent body; the spec's ~+5.8M
assumed all seven matrices were square -- Left/Right are 1056x264; noted).

Predictions checked in the JSON: CE at or better than E9a (no bottleneck,
more params); wiring Spearman up (uncontaminated groups); embedding-group
norms concentrated early-block and on the attention reads (the V14b
token-for-attention story). All paired vs E9a/E0a/E0b. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import math

import qk_e_common as E
from qk_e_common import Q, V8T, C, DEPTH, F, nn, torch
import qk_e1_slotnorm_run as E1R
import qk_e7_evenout_run as E7R

JP = E.jpath('qk_e10.json')
EMB_GROUP_WEIGHT_NOTE = ('slot groups unweighted (harness convention); the '
                         '264-col embedding group weighted by sqrt(264/11) '
                         '= sqrt(24) ~ 4.899')


class E10Route(E1R.E1Route):
    """Two-channel reads: original read matrices consume the per-slot-normed
    writes channel; parallel *_emb Linears consume the globally-normed
    embedding channel. norm_groups=1 gives the global-norm (E0b-style) base.
    split_ctl=True reduces to the unsplit forward exactly (control)."""
    split_ctl = False

    def __init__(self, variant, depth):
        super().__init__(variant, depth)
        Dm = self.wte.weight.shape[1]
        with torch.no_grad():
            for blk in self.h:
                for nm in E.READ_NAMES:
                    lin = getattr(blk, nm)
                    new = nn.Linear(Dm, lin.weight.shape[0], bias=False)
                    new.weight.copy_(lin.weight)     # W_emb := W_init
                    setattr(blk, nm + '_emb', new)

    def e10_penalty(self):
        tot = V8T._orig_group_penalty(self)          # 24 unweighted slot groups
        w = math.sqrt(float(E.NGROUP))               # sqrt(264/11)
        for blk in self.h:
            for nm in E.READ_NAMES:
                M = getattr(blk, nm + '_emb').weight
                tot = tot + w * (M.pow(2).sum() + 1e-12).sqrt()
        return tot

    def token_appetite(self):
        """84-entry table: per (block, read matrix) embedding-half Frobenius
        norm -- which modules read the token, straight from weights."""
        rows = []
        for li, blk in enumerate(self.h):
            for nm in E.READ_NAMES:
                rows.append({'block': li, 'matrix': nm, 'emb_group_norm': round(
                    float(getattr(blk, nm + '_emb').weight.detach()
                          .float().norm()), 4)})
        return rows

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

        def entry_pair(li):
            """(writes-channel sum, per-consumer embedding channel), with
            sub_entry mean-ablation support on every stream incl. 0."""
            sub = sub_entry.get(li) if sub_entry is not None else None
            get = lambda i: (sub[i] if (sub is not None and i in sub)
                             else streams[i])
            xw = None
            for i in self.vis[li]:
                if i == 0:
                    continue
                xw = get(i) if xw is None else xw + get(i)
            if xw is None:
                xw = torch.zeros(B, Tq, Dm, device=e.device, dtype=e.dtype)
            e_li = get(0) if 0 in self.vis[li] else \
                torch.zeros(B, Tq, Dm, device=e.device, dtype=e.dtype)
            if entry_override is not None and li in entry_override:
                xw = entry_override[li]              # best effort (terms only)
            return xw, e_li

        zero_e = None
        for l, blk in enumerate(self.h):
            xw, e_li = entry_pair(l)
            if collect is not None:
                collect['entry_norm'].append(
                    (xw + e_li).detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append((xw + e_li).detach())
            if self.split_ctl:
                hn_w = self.slot_norm(xw + e_li)     # unsplit normed input
                if zero_e is None:
                    zero_e = torch.zeros_like(e_li)
                hn_e = zero_e                        # emb half reads exact 0
            else:
                hn_w = self.slot_norm(xw)
                hn_e = F.rms_norm(e_li, (Dm,))

            def read(nm):
                lin = getattr(blk, nm)
                lin_e = getattr(blk, nm + '_emb')
                return lin(hn_w) + lin_e(hn_e)

            def qk(nm):
                z = read(nm).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qk('c_q'), qk('c_k')
            q2, k2 = qk('c_q2'), qk('c_k2')
            v = read('c_v').view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            xw2 = xw + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                if self.split_ctl:
                    xn_w = self.slot_norm(xw2 + e_li)
                else:
                    xn_w = self.slot_norm(xw2)
                left = blk.Left(xn_w) + blk.Left_emb(hn_e)
                right = blk.Right(xn_w) + blk.Right_emb(hn_e)
                mw = blk.Down(left * right) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        xw, e_li = entry_pair(self.depth)            # readout: unsplit (tied)
        x = xw + e_li
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e10(variant, groups):
    C.register(variant)
    torch.manual_seed(Q.SEED)
    m = E10Route(variant, DEPTH).to(E.DEV)
    m.norm_groups = groups
    return m


def make_e10a():
    return make_e10('E10a', E.NGROUP)


def make_e10b():
    return make_e10('E10b', 1)


# penalty dispatch: split models carry their own 25-group weighted penalty
if not hasattr(V8T, '_e10_penalty_patch'):
    V8T._e10_penalty_patch = True
    _prev_pen = V8T.group_penalty

    def _e10_dispatch(model):
        if hasattr(model, 'e10_penalty'):
            return model.e10_penalty()
        return _prev_pen(model)
    V8T.group_penalty = _e10_dispatch


def emb_read_params(m):
    return sum(getattr(blk, nm + '_emb').weight.numel()
               for blk in m.h for nm in E.READ_NAMES)


@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    # (1) split model in control mode == unsplit twin, both norm settings
    for mk, ref_mk, name in ((make_e10a, E7R.make_e7m1, 'E10a==E9a-arch'),
                             (make_e10b, E.make_e0b, 'E10b==E0b-arch')):
        ref = ref_mk().eval().float()
        m = mk().eval().float()
        m.split_ctl = True
        d = (m(idx) - ref(idx)).abs().max().item()
        print(f"control {name} (ctl mode, emb half fed exact zeros): "
              f"max |logit diff| {d:.2e}", flush=True)
        if E.SMOKE:
            assert d == 0.0, "split control must be bit-for-bit on CPU"
        else:
            assert d < 1e-6
        m.split_ctl = False
        d2 = (m(idx) - ref(idx)).abs().max().item()
        assert d2 > 1e-6                     # sanity: real split differs
        del m, ref
        torch.cuda.empty_cache()
    # (2) weighted 25-group penalty vs naive loop
    m = make_e10a().eval().float()
    p_fast = float(m.e10_penalty())
    w = math.sqrt(float(E.NGROUP))
    p_naive = 0.0
    for blk in m.h:
        for nm in E.READ_NAMES:
            M = getattr(blk, nm).weight
            for k in range(E.NGROUP):
                p_naive += float(
                    M[:, E.SUB * k:E.SUB * (k + 1)].pow(2).sum()) ** 0.5
            p_naive += w * float(
                getattr(blk, nm + '_emb').weight.pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control E10 penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel:.2e}", flush=True)
    assert rel < 1e-6
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


def muon_lasso_trainer(lr, gc, steps, **kw):
    return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(),
                        prox_coeff=None, **kw)


if __name__ == '__main__':
    E.setup()
    controls()
    mlr = E7R.muon_lr()

    m = make_e10a()
    ep = emb_read_params(m)
    import qk_v10v11_common as W
    E.merge(JP, 'param_counts', dict(
        W.param_counts(m), emb_read_params=ep,
        emb_read_params_note='7 emb halves per block: 5x(DxD) + 2x(4DxD); '
                             'the spec estimate ~+5.8M assumed all square',
        lasso_weighting=EMB_GROUP_WEIGHT_NOTE))
    del m
    torch.cuda.empty_cache()

    arms = (
        ('qk_e10_a', 'E10a', make_e10a, muon_lasso_trainer, 3e-5, mlr,
         dict(optimizer='muon', lasso='in-loss 3e-5, 25 groups/matrix',
              lasso_weighting=EMB_GROUP_WEIGHT_NOTE,
              design='embedding split on the best recipe (E9a twin)'),
         (('qk_e9_a', 'e9a'),)),
        ('qk_e10_b', 'E10b', make_e10b, None, E.GC, None,
         dict(optimizer='adamw', lasso='in-loss 1e-4, 25 groups/matrix',
              lasso_weighting=EMB_GROUP_WEIGHT_NOTE,
              design='embedding split on the AdamW reference (E0b twin)'),
         (('qk_e10_a', 'e10a'),)),
    )
    for stem, key, factory, trainer, gc, lr, extra, extras in arms:
        E.train_arm(stem, JP, key, factory, gc, lr=lr, trainer=trainer,
                    extra=extra)
        E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
        E.paired_fresh(stem, JP, key)
        pair_extra(stem, key, extras)
        E.probe_arm(stem, factory, JP, f'light_probe_{key}',
                    tok_key=f'tok_probe_{key}')
        if not E.SMOKE and os.path.exists(E.ckpath(stem)):
            out = E.loadj(JP)
            if f'token_appetite_{key}' not in out:
                m, _ = E.load_arm(stem, factory)
                rows = m.token_appetite()
                attn_tot = sum(r['emb_group_norm'] for r in rows
                               if r['matrix'] in ('c_q', 'c_k', 'c_q2',
                                                  'c_k2', 'c_v'))
                mlp_tot = sum(r['emb_group_norm'] for r in rows
                              if r['matrix'] in ('Left', 'Right'))
                E.merge(JP, f'token_appetite_{key}', {
                    'table': rows,
                    'attn_total': round(attn_tot, 3),
                    'mlp_total': round(mlp_tot, 3),
                    'per_block_total': [
                        round(sum(r['emb_group_norm'] for r in rows
                                  if r['block'] == li), 3)
                        for li in range(DEPTH)]})
                del m
                torch.cuda.empty_cache()
    print('e10 embsplit run done', flush=True)
