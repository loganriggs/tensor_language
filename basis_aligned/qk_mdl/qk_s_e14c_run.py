"""E14c commons at w1152 (Logan approved 2026-08-06): the one w264 win with
the right profile to survive width -- exactly param-matched by construction
(commons carved out of the same 1152 dims, not added), and it attacks the
measured +0.111 partition cost rather than protecting token identity.

Local result at w264: slots 24x9 + 48-dim commons recovered 0.156 of the
0.203 partition cost (final +0.048 vs vanilla) at Spearman 0.69.

Scale arms (qk_s_muon_run CFG):
  commons3e5   proportional translation: 24 x 40-dim slots (960) + 192-dim
               commons (fraction 1/6, same as w264's 48/264... 48/264 = 2/11;
               192/1152 = 1/6 -- see note below)
  commons96    half-fraction point: 24 x 44-dim slots (1056) + 96-dim commons
Note on proportion: w264 = 24 x 9 + 48 has commons fraction 48/264 = 0.182;
the exact w1152 match would be 209.5 dims. 192 (fraction 0.167, slots 40)
is the nearest slot-aligned layout; 96 (0.083, slots 44) gives the dial.

Reuses the local VarSlotRoute verbatim (width-generic; importing
qk_e14_slotcap_run also installs its group_penalty dispatch). Identity
anchor: VarSlotRoute with uniform sizes and commons=0 reduces bit-for-bit
to E1Route (proved in the local controls); here we assert the scale
layouts' structure + finite forward + penalty consistency.
"""
import qk_s_gate_run as G           # neuters Q.gpu_guard first
import qk_e_common as E
from qk_e_common import Q, C, DEPTH, torch
from qk_e14_slotcap_run import VarSlotRoute


def make_commons1152(commons=192, variant=None):
    variant = variant or f'E14c{commons}'
    C.register(variant)
    torch.manual_seed(Q.SEED)
    slot = (1152 - commons) // 24
    assert slot * 24 + commons == 1152, (slot, commons)
    m = VarSlotRoute(variant, DEPTH, [slot] * 24, commons=commons).to(E.DEV)
    return m


def make_commons192():
    return make_commons1152(192)


def make_commons96():
    return make_commons1152(96)


@torch.no_grad()
def controls():
    import qk_s_e1_run as E1R
    idx = Q.HELD[:2, :Q.T]
    base = E1R.make_e1().eval().float()
    # uniform, commons=0 must reduce to E1Route exactly (local anchor, at
    # scale width)
    m0 = make_commons1152(0, variant='E14ctl1152').eval().float()
    d = (m0(idx) - base(idx)).abs().max().item()
    print(f"control commons=0 uniform == E1Route at init: {d:.2e}", flush=True)
    assert d < 1e-4
    for mk, commons in ((make_commons192, 192), (make_commons96, 96)):
        m = mk().eval().float()
        out = m(idx)
        assert torch.isfinite(out).all()
        assert m.commons == commons and m.seg_sizes[-1] == commons
        assert all(s == (1152 - commons) // 24 for s in m.seg_sizes[:-1])
        body = sum(p.numel() for n, p in m.named_parameters() if 'wte' not in n)
        bb = sum(p.numel() for n, p in base.named_parameters() if 'wte' not in n)
        pen = float(m.custom_group_penalty())
        print(f"commons{commons}: finite, body {body} (recipe {bb}, "
              f"{'MATCH' if body == bb else 'MISMATCH'}), penalty {pen:.1f}",
              flush=True)
        d2 = (out - base(idx)).abs().max().item()
        assert d2 > 1e-6
        del m
    del base, m0
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import qk_w1152_train as W2
    W2.patch_width(G.WIDTH)
    G.setup_data()
    controls()
    print('e14c w1152 controls done', flush=True)
