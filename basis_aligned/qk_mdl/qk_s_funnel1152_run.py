"""Scale funnel, PARAMETER-MATCHED to the w1152 recipe (Logan's overnight
directive: larger versions, apples-to-apples).

Config: wide 1536 = 24 slot groups x 64 (heads 24x64) detokenization block,
narrow 1092 = 26 groups x 42 (heads 21x52, HDn even), depth 12. Body params
18*1536^2 + 11*18*1092^2 + 4*1536*1092 (P_a, P_m, P_sv, W_up) = 285.4M vs
combo3e5loss's 286.7M active body (-0.5%) -- the closest integer-slot match.
Two arms via qk_s_muon_run CFG:
  funnelsv  shared values (the w264 double-winner) -- frontier candidate
  funnel    no shared values -- decomposition control
Both on the scale protocol (G.setup_data order, batch 32 via micro accum,
9328 steps, scale held 1500) so they pair directly against combo3e5loss /
vanilla heldloss arrays. Penalty: importing qk_e12_funnel_run patches
V8T.group_penalty to dispatch to FunnelRoute.custom_group_penalty (wide
reads grouped by Gw=24, narrow by Gn=26) -- coeff 3e-5 as the recipe.
"""
import importlib

import qk_s_gate_run as G           # neuters Q.gpu_guard first
import qk_e_common as E
from qk_e_common import Q, torch

M12 = importlib.import_module('qk_e12_funnel_run')   # applies penalty patch


def cfg_plain():
    # body 283.6M (-1.1% vs recipe 286.7M)
    return dict(Dw=1536, NHw=24, HDw=64, Gw=24, Dn=1092, NHn=21, HDn=52,
                Gn=26, sub_n=42, control=False)


def cfg_sv():
    # shared values drop the 11 narrow c_v modules (-13.1M), so the sv arm
    # gets its params back as narrow width: Dn 1118 = 26 x 43 (heads 13x86)
    # -> body 283.1M, within 0.2% of the plain arm and -1.25% of the recipe
    return dict(Dw=1536, NHw=24, HDw=64, Gw=24, Dn=1118, NHn=13, HDn=86,
                Gn=26, sub_n=43, control=False)


def make_funnel_sv():
    torch.manual_seed(Q.SEED)
    return M12.FunnelRoute('E12sLv', cfg_sv(), shared_values=True).to(E.DEV)


def make_funnel_plain():
    torch.manual_seed(Q.SEED)
    return M12.FunnelRoute('E12sL', cfg_plain(),
                           shared_values=False).to(E.DEV)


if __name__ == '__main__':
    import qk_w1152_train as W2
    W2.patch_width(G.WIDTH)
    G.setup_data()
    with torch.no_grad():
        for mk, nm in ((make_funnel_sv, 'E12sLv'),
                       (make_funnel_plain, 'E12sL')):
            m = mk().eval().float()
            out = m(Q.HELD[:2, :Q.T])
            assert torch.isfinite(out).all()
            pen = float(m.custom_group_penalty())
            body = sum(p.numel() for n, p in m.named_parameters()
                       if 'wte' not in n)
            print(f"{nm}: finite forward, penalty {pen:.1f}, body {body} "
                  f"(recipe active body 286,668,288)", flush=True)
            del m
            torch.cuda.empty_cache()
    print('funnel1152 controls done', flush=True)
