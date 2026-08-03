"""E1 PER-SLOT RMSNORM (fresh single-epoch protocol, see qk_e_common).

The V8-base architecture normalizes every module input with a GLOBAL RMSNorm
over all 264 dims, which couples every slot's scale to every other slot's (the
norm's denominator is the one global nonlinearity threading the slots
together). E1 replaces the two module-input norms (attention input hn, MLP
input xn) with an INDEPENDENT RMSNorm per 11-dim write slot (24 slots
normalized separately). The embedding input norm and the final pre-readout
norm stay GLOBAL (per spec; a per-slot readout norm variant was judged not
worth an arm). Questions: (a) CE cost of decoupling the global scale coupling,
paired vs E0a/E0b on the fresh held set; (b) does wiring readability improve
(weight-vs-causal wiring Spearman, same light probe as the V13 runs)?

Positive controls: (i) E1 with ONE norm group (group = all dims) reproduces
the plain V8Route forward bit-for-bit at init (< 1e-4, fp32) -- exercises the
overridden forward against the template; (ii) the vectorized per-slot norm
equals a naive per-slot loop (< 1e-6). Results -> qk_e1.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_e_common as E
from qk_e_common import Q, V8T, C, W, DEPTH, F, torch

JP = E.jpath('qk_e1.json')
STEM = 'qk_e1_slotnorm'


class E1Route(V8T.V8Route):
    """V8Route with per-slot RMSNorm at the two module inputs. norm_groups=1
    reduces exactly to the global norm (identity-control mode)."""
    norm_groups = 1

    def slot_norm(self, x):
        G = self.norm_groups
        Dm = x.shape[-1]
        if G <= 1:
            return F.rms_norm(x, (Dm,))
        S = Dm // G
        sh = x.shape
        return F.rms_norm(x.view(*sh[:-1], G, S), (S,)).view(sh)

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        e = F.rms_norm(self.wte(idx), (Dm,))          # embedding norm: GLOBAL
        streams = [e]
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x)                    # PER-SLOT (was global)

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
                xn = self.slot_norm(x)                # PER-SLOT (was global)
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
        x = F.rms_norm(x, (Dm,))                      # pre-readout norm: GLOBAL
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e1(norm_groups=None):
    C.register('E1')
    torch.manual_seed(Q.SEED)
    m = E1Route('E1', DEPTH).to(E.DEV)
    m.norm_groups = E.NGROUP if norm_groups is None else norm_groups
    return m


@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    # (i) one norm group == plain V8Route (same seed -> identical params)
    base = C.make_variant('E1ctl').eval().float()
    m1 = make_e1(norm_groups=1).eval().float()
    d = (m1(idx) - base(idx)).abs().max().item()
    print(f"control E1(1 group)==V8Route at init: max |logit diff| {d:.2e}",
          flush=True)
    assert d < 1e-4
    # sanity: the real per-slot norm actually changes the function
    m24 = make_e1().eval().float()
    d2 = (m24(idx) - base(idx)).abs().max().item()
    print(f"sanity E1({E.NGROUP} groups) differs from base: max diff {d2:.2e}",
          flush=True)
    assert d2 > 1e-6
    # (ii) vectorized slot norm == naive per-slot loop
    x = torch.randn(3, 5, Q.D)
    fast = m24.slot_norm(x)
    naive = torch.empty_like(x)
    for k in range(E.NGROUP):
        sl = slice(E.SUB * k, E.SUB * (k + 1))
        naive[..., sl] = F.rms_norm(x[..., sl], (E.SUB,))
    rel = (fast - naive).abs().max().item()
    print(f"control slot-norm fast vs naive: max abs diff {rel:.2e}", flush=True)
    assert rel < 1e-6
    del base, m1, m24
    torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()
    m = make_e1()
    E.merge(JP, 'param_counts', W.param_counts(m))
    del m
    torch.cuda.empty_cache()
    E.train_arm(STEM, JP, 'E1', make_e1, E.GC)
    E.oldheld_record(STEM, make_e1, JP, 'E1_oldheld')
    E.paired_fresh(STEM, JP, 'E1')
    E.probe_arm(STEM, make_e1, JP, 'light_probe_E1')
    if not E.SMOKE:
        e0 = E.loadj(E.jpath('qk_e0.json'))
        if 'light_probe_E0b' in e0:
            E.merge(JP, 'wiring_spearman_E0b_reference', {
                k: e0['light_probe_E0b'][k]
                for k in ('wiring_spearman_all', 'wiring_spearman_effectual',
                          'wiring_top10_precision')})
    print('e1 slotnorm run done', flush=True)
