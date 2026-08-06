"""Shared values at the w1152 scale recipe (the E12 transfer question).

The E12 funnel family's strong signal was SHARED VALUES: one value tensor
computed in block 0 and re-projected for every later block won twice
independently (E12Lv -0.086 at matched bandwidth, E12b -0.084 under
narrowing) while preserving or improving wiring readability. This arm asks
whether that transfers to the constant-width w1152 recipe (combo3e5loss =
per-slot RMSNorm + Muon + in-loss group-lasso 3e-5, held CE 4.10596).

E1SVRoute = E1Route with: block 0 computes v0 = c_v(hn0) as usual; blocks
1..11 all use the SAME value tensor P_sv(v0) (one shared projection, funnel
semantics) instead of their own c_v. The unused c_v weights of blocks 1..11
are zeroed at init: they receive no gradients (unused in forward), stay
zero, contribute zero to the group-lasso penalty, and read as exact zeros
in wiring probes. Active-param accounting: body loses 11 x 1152^2 active
c_v dims and gains one 1152^2 P_sv.

Seed parity: make_e1sv seeds identically to make_e1 before construction, so
all Block parameters match combo3e5loss at init; P_sv consumes RNG after.

Run via: python qk_s_muon_run.py combo3e5sv  (arm added to its CFG).
Standalone: python qk_s_e1sv_run.py  runs the identity/shape controls only.
"""
import qk_s_gate_run as G           # neuters Q.gpu_guard before probe chain
import qk_e_common as E
from qk_e_common import Q, F, torch
import qk_s_e1_run as E1R

nn = torch.nn


class E1SVRoute(E1R.E1Route):
    shared_values = True

    def __init__(self, variant, depth):
        super().__init__(variant, depth)
        Dm = self.wte.weight.shape[1]
        self.P_sv = nn.Linear(Dm, Dm, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(self.P_sv.weight)
            if self.shared_values:
                for blk in list(self.h)[1:]:
                    blk.c_v.weight.zero_()

    def _values(self, l, blk, hn):
        """Value tensor for block l (hook: the per-block-projection variant
        overrides this). Sequential-forward state via self._v0/_vsh."""
        if not self.shared_values:
            return blk.c_v(hn)
        if l == 0:
            self._v0 = blk.c_v(hn)
            self._vsh = self.P_sv(self._v0)
            return self._v0
        return self._vsh

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
            v = self._values(l, blk, hn).view(B, Tq, NHm, HDm)
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


def make_e1sv():
    from qk_e_common import C, DEPTH
    C.register('E1SV')
    torch.manual_seed(Q.SEED)
    m = E1SVRoute('E1SV', DEPTH).to(E.DEV)
    m.norm_groups = E.NGROUP
    return m


class E1SVPBRoute(E1SVRoute):
    """PARAMETER-MATCHED shared-source values (Logan's apples-to-apples
    directive): block l >= 1 uses its own projection P_sv_pb[l-1] of block
    0's value tensor v0. The 11 projections replace the 11 zeroed c_v
    matrices one-for-one (both Dm x Dm), so ACTIVE body params equal
    combo3e5loss exactly -- isolating the shared-SOURCE effect from the
    combo3e5sv arm's -4.6%% capacity confound."""

    def __init__(self, variant, depth):
        super().__init__(variant, depth)
        Dm = self.wte.weight.shape[1]
        del self.P_sv                    # single projection unused here
        self.P_sv_pb = nn.ModuleList(
            [nn.Linear(Dm, Dm, bias=False) for _ in range(depth - 1)])
        with torch.no_grad():
            for lin in self.P_sv_pb:
                nn.init.orthogonal_(lin.weight)

    def _values(self, l, blk, hn):
        if l == 0:
            self._v0 = blk.c_v(hn)
            return self._v0
        return self.P_sv_pb[l - 1](self._v0)


def make_e1svpb():
    from qk_e_common import C, DEPTH
    C.register('E1SVPB')
    torch.manual_seed(Q.SEED)
    m = E1SVPBRoute('E1SVPB', DEPTH).to(E.DEV)
    m.norm_groups = E.NGROUP
    return m


@torch.no_grad()
def controls():
    """(i) shared_values=False + P_sv ignored reduces exactly to E1Route;
    (ii) shared path is finite and differs from base; (iii) active-param
    accounting."""
    idx = Q.HELD[:2, :Q.T]
    base = E1R.make_e1().eval().float()
    E1SVRoute.shared_values = False
    try:
        m0 = make_e1sv().eval().float()
        m0.shared_values = False          # instance attr: survives the class
    finally:                              # toggle being restored below
        E1SVRoute.shared_values = True
    d = (m0(idx) - base(idx)).abs().max().item()
    print(f"control E1SV(shared off)==E1Route at init: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert d < 1e-4
    msv = make_e1sv().eval().float()
    out = msv(idx)
    assert torch.isfinite(out).all()
    d2 = (out - base(idx)).abs().max().item()
    print(f"sanity E1SV(shared on) differs from base: max diff {d2:.2e}",
          flush=True)
    assert d2 > 1e-6
    Dm = msv.wte.weight.shape[1]
    dead = sum(int(blk.c_v.weight.abs().sum() == 0) for blk in msv.h)
    body = sum(p.numel() for n, p in msv.named_parameters() if 'wte' not in n)
    print(f"accounting: {dead}/12 c_v zeroed; body params {body} "
          f"(active {body - dead * Dm * Dm})", flush=True)
    assert dead == 11
    del base, m0, msv
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import qk_w1152_train as W2
    W2.patch_width(G.WIDTH)
    G.setup_data()
    controls()
    print('e1sv controls done', flush=True)
