"""FIVE SHARING DESIGNS at w264 (Logan 2026-08-06: "run smaller versions of
1-5 now") -- alternative ways to allow slot mixing without giving up
interpretability, each priced on the fresh single-epoch protocol against
E9a (uniform slots, exclusive) and the E14c commons result (-0.156 of the
partition cost recovered; ledger showed the commons is mostly an OUTPUT
ACCUMULATOR: readout read norm 557 vs 62-85 for blocks).

Arms (chained in effort order, ~50 min each; JSON qk_s_share.json):
  S1 rocommons   readout-only commons: E14c layout (24x9 + 48) but blocks
                 CANNOT read the commons (block inputs see commons zeroed);
                 only the readout consumes it. If CE holds vs E14c, the
                 shared space is purely a logit-staging bus.
  S3 typedcommons four typed commons of 12 dims each; commons quarter q is
                 writable only by modules 6q..6q+5 (depth-banded writer
                 sets) -- attribution ambiguity bounded to 6 writers.
  S2 writelasso  NO commons; write masks OPEN (any module may write any
                 slot) with a group-lasso on off-slot write rows -- sharing
                 must be earned and shows up as an enumerable permission
                 list. Same coeff as the read lasso.
  S5 copyedges   exclusive slots + one depth-shared copy operator K applied
                 to every consumer entry (x + K(slotnorm(x)), K's 24x24
                 slot-pair blocks group-penalized, diagonal zeroed,
                 zero-init) -- sharing as visible, priced edges.
  S4 faccommons  commons whose writes factor through a shared 16-direction
                 basis B: module writes to the commons only via B @ A_m
                 (A_m zero-init) -- the shared space carries at most 16
                 enumerable features.

All exactly param-comparable in spirit (S1/S3/S4 same body as E14c up to
tiny A/B/K additions; S2/S5 same as E9a + penalty/K). Paired vs qk_e9_a
(heldloss array now pushed by local). Idempotent; safe to rerun."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_tokenline_train as _Q0
_Q0.gpu_guard = lambda *a, **k: None    # box guard bug (see qk_s_gate_run)

import qk_e_common as E
from qk_e_common import Q, C, DEPTH, F, nn, torch
import qk_e7_evenout_run as E7R
from qk_e14_slotcap_run import VarSlotRoute, var_light_probe

JP = E.jpath('qk_s_share.json')
GC = 3e-5
NG = 24


def _pad_last(t, Dm, r):
    out = torch.zeros(*t.shape[:-1], Dm, device=t.device, dtype=t.dtype)
    out[..., Dm - r:] = t
    return out


class ROCommons(VarSlotRoute):
    """S1: blocks cannot read the commons (their normed inputs see zeros
    there); the readout path (global rms_norm) reads everything."""

    def __init__(self, variant, depth, sizes, commons=0):
        super().__init__(variant, depth, sizes, commons=commons)
        Dm = self.wte.weight.shape[1]
        m = torch.ones(Dm)
        if commons > 0:
            m[Dm - commons:] = 0.0
        self.register_buffer('block_read_mask', m)

    def slot_norm(self, x):
        return super().slot_norm(x) * self.block_read_mask.to(x.dtype)


class TypedCommons(VarSlotRoute):
    """S3: the 48 commons dims split into 4 typed segments of 12; commons
    quarter q writable only by modules 6q..6q+5. Norm/penalty segments are
    rebuilt as 24 slots + 4 commons groups (28 groups)."""

    def __init__(self, variant, depth, sizes, commons=48):
        super().__init__(variant, depth, sizes, commons=commons)
        Dm = self.wte.weight.shape[1]
        q = commons // 4
        self.seg_sizes = list(sizes) + [q] * 4
        bounds, off = [], 0
        for s in self.seg_sizes:
            bounds.append((off, off + s))
            off += s
        self.seg_bounds = bounds
        ind = torch.zeros(Dm, len(self.seg_sizes))
        for g, (a, b) in enumerate(bounds):
            ind[a:b, g] = 1.0
        self.seg_ind = ind.to(self.seg_ind.device)
        wm = torch.zeros(2 * depth, Dm)
        for k in range(2 * depth):
            a, b = bounds[k]
            wm[k, a:b] = 1.0
            qa, qb = bounds[24 + k // 6]
            wm[k, qa:qb] = 1.0
        with torch.no_grad():
            self.wmask.copy_(wm)


class WriteLasso(VarSlotRoute):
    """S2: exclusive slots abolished -- write masks all-ones; off-slot
    write ROWS group-penalized (c_proj and Down per module) at the read
    coeff. Sharing becomes an earned, enumerable permission list."""

    def __init__(self, variant, depth, sizes):
        super().__init__(variant, depth, sizes, commons=0)
        with torch.no_grad():
            self.wmask.fill_(1.0)

    def custom_group_penalty(self):
        tot = super().custom_group_penalty()
        for k in range(2 * self.depth):
            blk = self.h[k // 2]
            W = blk.c_proj.weight if k % 2 == 0 else blk.Down.weight
            rowsq = W.pow(2).sum(1)
            g = (rowsq @ self.seg_ind + 1e-12).sqrt()
            tot = tot + g.sum() - g[k]
        return tot


class CopyEdges(VarSlotRoute):
    """S5: exclusive slots + one depth-shared copy operator applied to every
    consumer entry: x + K(slot_norm(x)); K's (in-slot, out-slot) blocks are
    group-penalized, diagonal blocks masked out, zero-init (exact E9a-arch
    reduction at init)."""

    def __init__(self, variant, depth, sizes):
        super().__init__(variant, depth, sizes, commons=0)
        Dm = self.wte.weight.shape[1]
        self.K = nn.Linear(Dm, Dm, bias=False)
        with torch.no_grad():
            self.K.weight.zero_()
        km = torch.ones(Dm, Dm)
        for (a, b) in self.seg_bounds:
            km[a:b, a:b] = 0.0
        self.register_buffer('kmask', km)

    def assemble(self, li, streams, sub, coef_out):
        x = super().assemble(li, streams, sub, coef_out)
        Ke = self.K.weight * self.kmask.to(self.K.weight.dtype)
        return x + F.linear(super().slot_norm(x), Ke)

    def custom_group_penalty(self):
        tot = super().custom_group_penalty()
        K2 = (self.K.weight * self.kmask).pow(2)
        gm = self.seg_ind.t() @ K2 @ self.seg_ind
        tot = tot + (gm + 1e-12).sqrt().sum()
        return tot


class FacCommons(VarSlotRoute):
    """S4: commons writable only through a shared 16-direction basis B --
    module k's commons write = B @ A_k(module mix), A_k zero-init. Base
    write masks exclude the commons entirely."""
    KDIM = 16

    def __init__(self, variant, depth, sizes, commons=48):
        super().__init__(variant, depth, sizes, commons=commons)
        Dm = self.wte.weight.shape[1]
        with torch.no_grad():
            self.wmask[:, Dm - commons:] = 0.0
        self.B = nn.Linear(self.KDIM, commons, bias=False)
        torch.manual_seed(Q.SEED + 1)
        nn.init.orthogonal_(self.B.weight)
        self.Aa = nn.ModuleList(
            [nn.Linear(Dm, self.KDIM, bias=False) for _ in range(depth)])
        self.Am = nn.ModuleList(
            [nn.Linear(Q.EXP * Dm, self.KDIM, bias=False)
             for _ in range(depth)])
        with torch.no_grad():
            for lin in list(self.Aa) + list(self.Am):
                lin.weight.zero_()

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B_, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        r = self.commons
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

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = self.slot_norm(x)

            def qk(lin):
                z = lin(hn).view(B_, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B_, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B_, Tq, Dm)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            aw = aw + _pad_last(self.B(self.Aa[l](y)), Dm, r)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = self.slot_norm(x)
                h = blk.Left(xn) * blk.Right(xn)
                mw = blk.Down(h) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
                mw = mw + _pad_last(self.B(self.Am[l](h)), Dm, r)
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


def _layout(with_commons):
    """(sizes, commons) for the real width, or the E14-style smoke layout."""
    if E.SMOKE:
        return ([1] * 20 + [0] * 4, 4) if with_commons else ([1] * 24, 0)
    return ([9] * 24, 48) if with_commons else ([11] * 24, 0)


def _mk(cls, variant, with_commons):
    C.register(variant)
    torch.manual_seed(Q.SEED)
    sizes, commons = _layout(with_commons)
    kw = {'sizes': sizes}
    if with_commons:
        kw['commons'] = commons
    return cls(variant, DEPTH, **kw).to(E.DEV)


def make_s1():
    return _mk(ROCommons, 'S1ro', True)


def make_s3():
    return _mk(TypedCommons, 'S3ty', True)


def make_s2():
    return _mk(WriteLasso, 'S2wl', False)


def make_s5():
    return _mk(CopyEdges, 'S5ce', False)


def make_s4():
    return _mk(FacCommons, 'S4fc', True)


ARMS = [
    ('qk_s_share_s1', 'S1rocommons', make_s1,
     'readout-only commons: E14c layout, blocks read commons as zeros'),
    ('qk_s_share_s3', 'S3typedcommons', make_s3,
     'four 12-dim typed commons, writer sets = depth-banded 6 modules'),
    ('qk_s_share_s2', 'S2writelasso', make_s2,
     'no commons; open write masks + off-slot write-row group lasso'),
    ('qk_s_share_s5', 'S5copyedges', make_s5,
     'exclusive slots + zero-init depth-shared copy operator K, slot-pair '
     'blocks penalized, diagonal masked'),
    ('qk_s_share_s4', 'S4faccommons', make_s4,
     'commons writes factored through a shared 16-direction basis B '
     '(A_m zero-init)'),
]


@torch.no_grad()
def controls():
    import qk_e1_slotnorm_run as E1R
    idx = Q.HELD[:2, :Q.T]
    base = E1R.make_e1().eval().float()
    # S5 zero-init K == uniform VarSlot == E1Route exactly
    m5 = make_s5().eval().float()
    d = (m5(idx) - base(idx)).abs().max().item()
    print(f"control S5(K=0)==E1Route at init: {d:.2e}", flush=True)
    assert d < 1e-4
    # S4 zero-init A: commons never written; finite + penalty runs
    m4 = make_s4().eval().float()
    assert torch.isfinite(m4(idx)).all()
    assert float(m4.wmask[:, -m4.commons:].abs().sum()) == 0.0
    # S1/S3/S2 structural + finite
    m1 = make_s1().eval().float()
    assert torch.isfinite(m1(idx)).all()
    assert float(m1.block_read_mask[-m1.commons:].sum()) == 0.0
    m3 = make_s3().eval().float()
    assert torch.isfinite(m3(idx)).all()
    assert len(m3.seg_sizes) == 28 and m3.seg_ind.shape[1] == 28
    if not E.SMOKE:
        for k in range(24):
            row = m3.wmask[k]
            assert float(row[216 + 12 * (k // 6):
                             216 + 12 * (k // 6 + 1)].sum()) == 12.0
            assert float(row[216:].sum()) == 12.0  # only its own quarter
    m2 = make_s2().eval().float()
    assert torch.isfinite(m2(idx)).all()
    assert float(m2.wmask.sum()) == m2.wmask.numel()   # all ones
    p2 = float(m2.custom_group_penalty())
    print(f"controls S1/S2/S3/S4 finite; S2 penalty {p2:.1f}", flush=True)
    del base, m1, m2, m3, m4, m5
    torch.cuda.empty_cache()


def run_arm(stem, key, factory, design):
    E.train_arm(stem, JP, key, factory, GC, lr=E7R.muon_lr(),
                trainer=lambda lr, gc, steps, **kw: E.train_muon(
                    lr, gc, steps, lr_adamw=E.get_lr(), **kw),
                extra={'optimizer': 'muon', 'design': design})
    E.paired_fresh(stem, JP, key)
    if not E.SMOKE and os.path.exists(f'{E.QK}/qk_e9_a_heldloss.npy'):
        E.merge(JP, f'{key}_minus_e9a_fresh',
                E.paired(f'{stem}_heldloss.npy', 'qk_e9_a_heldloss.npy',
                         len(Q.HELD), 'e9a'))
    if not E.SMOKE and os.path.exists(E.ckpath(stem)) \
            and f'light_probe_{key}' not in E.loadj(JP):
        try:
            m, _ = E.load_arm(stem, factory)
            E.merge(JP, f'light_probe_{key}', var_light_probe(m))
            del m
            torch.cuda.empty_cache()
        except Exception as ex:                      # noqa: BLE001
            E.merge(JP, f'light_probe_{key}',
                    {'error': f'{type(ex).__name__}: {ex}'})


if __name__ == '__main__':
    E.setup()
    controls()
    for stem, key, factory, design in ARMS:
        run_arm(stem, key, factory, design)
    print('share designs run done', flush=True)
