"""E9 COMPOSITION ARMS (Logan 2026-08-05; fresh single-epoch batch-16).

Context: the fork verdict fixed the readable recipe as per-slot RMSNorm + Muon
+ IN-LOSS group-lasso 3e-5 (not proximal -- the prox never binds), now running
at width 1152 in the scale session; E8 separately showed the V14b
attention-only token line (embedding excluded from every entry sum, added only
into each block's attention read) is about -0.098 vs base by an orthogonal
mechanism. E9 asks whether they compose:

  E9a  per-slot RMSNorm + Muon + in-loss lasso 3e-5 at width 264 -- the LOCAL
       TWIN of the scale candidate. Wiring + token-determined probes
       (frontier prediction: Spearman ~0.6-0.75).
  E9b  E9a + the attention token line: hn_attention = per-slot-norm(entry + e)
       at every block, embedding in NO entry sum (block 0's entry is zero),
       MLP reads per-slot-norm(entry + attention write) with no embedding,
       readout reads all 24 writes. Wiring + token-determined probes; paired
       vs E9a / E0a / E0b / E8v14b -- do the gains stack or do both supply
       the same token demand?
  E9c  E9b + the N=6 lookback window (the readability-record ingredient):
       the maximal-readability stack. Wiring probe; run last (idle GPU, cost
       expected to stack; the record Spearman number is the deliverable).

Identity positive controls: E9b with the token line disabled (as_base) must
equal E9a exactly; E9c with full visibility installed must equal E9b exactly.
Curves into qk_e9.json; idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_e_common as E
from qk_e_common import Q, C, DEPTH, F, torch
import qk_e1_slotnorm_run as E1R
import qk_e7_evenout_run as E7R

JP = E.jpath('qk_e9.json')
GC9 = 3e-5
FULL_VIS = [list(range(1 + 2 * min(li, DEPTH))) for li in range(DEPTH + 1)]


class E9bRoute(E1R.E1Route):
    """Per-slot-norm base + attention-only token line. as_base=True (control)
    restores the plain E1Route path (token line off; embedding back in the
    entry sums via the externally installed full visibility)."""
    as_base = False

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
            if not self.vis[li]:
                return torch.zeros(B, Tq, Dm, device=e.device, dtype=e.dtype)
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x if self.as_base else x + e)

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
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = self.slot_norm(x)           # mlp: no embedding line
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


def make_e9a():
    """Local twin of the scale candidate (per-slot norm arch; the recipe's
    Muon + in-loss lasso 3e-5 live in the trainer)."""
    return E7R.make_e7m1()


def make_e9b(as_base=False, windowed=False):
    C.register('E9b')
    torch.manual_seed(Q.SEED)
    m = E9bRoute('E9b', DEPTH).to(E.DEV)
    m.norm_groups = E.NGROUP
    m.as_base = as_base
    if as_base:
        m.vis = [list(v) for v in FULL_VIS]
    elif windowed:
        m.vis = [[i for i in C.window_vis(li, N=6) if i != 0]
                 for li in range(DEPTH + 1)]
    else:
        m.vis = [[i for i in FULL_VIS[li] if i != 0]
                 for li in range(DEPTH + 1)]
    return m


def make_e9c():
    return make_e9b(windowed=True)


def muon_lasso_trainer(lr, gc, steps, **kw):
    """The fork-verdict recipe: Muon with the lasso IN THE LOSS (no prox)."""
    return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(),
                        prox_coeff=None, **kw)


@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    # E9b with the token line disabled == E9a exactly
    ma = make_e9a().eval().float()
    mb0 = make_e9b(as_base=True).eval().float()
    d = (mb0(idx) - ma(idx)).abs().max().item()
    print(f"control E9b(token line off)==E9a at init: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert d < 1e-4
    # sanity: the real E9b differs
    mb = make_e9b().eval().float()
    d2 = (mb(idx) - ma(idx)).abs().max().item()
    print(f"sanity E9b(token line on) differs from E9a: max diff {d2:.2e}",
          flush=True)
    assert d2 > 1e-6
    assert mb.vis[0] == [] and all(0 not in v for v in mb.vis)
    del ma, mb0
    # E9c with full visibility installed == E9b exactly
    mc = make_e9c().eval().float()
    assert all(0 not in v for v in mc.vis) and mc.vis[12] == list(range(13, 25))
    mc.vis = [list(v) for v in mb.vis]
    d3 = (mc(idx) - mb(idx)).abs().max().item()
    print(f"control E9c(full visibility)==E9b at init: max |logit diff| "
          f"{d3:.2e}", flush=True)
    assert d3 < 1e-4
    del mb, mc
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
    RECIPE = {'optimizer': 'muon', 'lasso': 'in-loss 3e-5 (fork verdict: '
              'proximal never binds)', 'lr_adamw_for_embedding': E.get_lr()}

    arms = (
        ('qk_e9_a', 'E9a', make_e9a, True,
         dict(RECIPE, design='per-slot RMSNorm; local twin of the width-1152 '
                             'scale candidate'), ()),
        ('qk_e9_b', 'E9b', make_e9b, True,
         dict(RECIPE, design='E9a + attention-only token line (embedding in '
                             'no entry sum)'),
         (('qk_e9_a', 'e9a'), ('qk_e8_v14b', 'e8v14b'))),
        ('qk_e9_c', 'E9c', make_e9c, False,
         dict(RECIPE, design='E9b + N=6 lookback window (maximal-readability '
                             'stack)'),
         (('qk_e9_b', 'e9b'),)),
    )
    for stem, key, factory, want_tok, extra, extra_pairs in arms:
        E.train_arm(stem, JP, key, factory, GC9, lr=mlr,
                    trainer=muon_lasso_trainer, extra=extra)
        E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
        E.paired_fresh(stem, JP, key)
        pair_extra(stem, key, extra_pairs)
        E.probe_arm(stem, factory, JP, f'light_probe_{key}',
                    tok_key=(f'tok_probe_{key}' if want_tok else None))
    print('e9 compose run done', flush=True)
