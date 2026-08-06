"""E16b (shrinking embedding channel, floor variant) at w1152 -- the scale
transfer of the local session's best readable arm (E16b beat E9a by -0.0315
at w264; suggested by local for the w1152 integration queue).

Reuses the local E16Route class verbatim (qk_e16_shrinkemb_run; it is
width-generic: slot dim S = Dm/(2*depth) = 48, remnant schedule
max(1152 - 96i, 192) for i=1..11 plus a 192-dim readout remnant -- the
4-slot floor scales to 192 dims, proportionally identical to w264 E16b).
Recipe = combo3e5loss conventions: per-slot RMSNorm + Muon + in-loss lasso
3e-5, identical data order. Run via: python qk_s_muon_run.py shrink3e5.

Identity control at scale: mode='control' (full-width normed embedding at
every entry incl. readout, W_rem unused) must equal the E1Route recipe
model (qk_s_e1_run.make_e1) bit-for-bit at init -- W_rem's identity-slice
init consumes no RNG, so all shared params are seed-identical.
"""
import qk_s_gate_run as G           # neuters Q.gpu_guard first
import qk_e_common as E
from qk_e_common import Q, C, DEPTH, torch
import qk_s_e1_run as E1R
from qk_e16_shrinkemb_run import E16Route, FLOOR_SLOTS


def make_shrink1152(mode='floor', variant='E16B1152'):
    C.register(variant)
    torch.manual_seed(Q.SEED)
    m = E16Route(variant, DEPTH, mode=mode).to(E.DEV)
    m.norm_groups = E.NGROUP
    return m


@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    base = E1R.make_e1().eval().float()
    mc = make_shrink1152(mode='control', variant='E16ctl1152').eval().float()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out_ref = base(idx)
        d = (mc(idx) - out_ref).abs().max().item()
        mb = make_shrink1152().eval().float()
        d2 = (mb(idx) - out_ref).abs().max().item()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f"control E16(no-shrink)==E1Route recipe at init: max |logit "
          f"diff| {d:.2e}", flush=True)
    assert d < 1e-4
    print(f"sanity E16b-1152 (floor schedule) differs: {d2:.2e}", flush=True)
    assert d2 > 1e-6
    Dm = mb.wte.weight.shape[1]
    S = Dm // (2 * DEPTH)
    fl = FLOOR_SLOTS * S
    assert S == 48 and fl == 192, (S, fl)
    assert mb.rem_dims[0] == Dm
    assert all(mb.rem_dims[li] == max(Dm - 2 * S * li, fl)
               for li in range(1, DEPTH + 1)), mb.rem_dims
    extra = sum(p.numel() for p in mb.W_rem.parameters())
    expect = Dm * sum(mb.rem_dims[1:])
    assert extra == expect, (extra, expect)
    assert torch.isfinite(mb(idx)).all()
    print(f"control E16b-1152 schedule {mb.rem_dims}; extra remnant params "
          f"{extra}", flush=True)
    del base, mc, mb, out_ref
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import qk_w1152_train as W2
    W2.patch_width(G.WIDTH)
    G.setup_data()
    controls()
    print('e16 w1152 controls done', flush=True)
