"""S3 typed commons at w1152: does typing the shared reads survive scale?

Motivation (2026-08-06): at w264 the typed commons was the readability
frontier -- Spearman 0.823 (best of ANY arm, above the un-shared base's
0.62) while recovering a third of the full commons' gain. At scale the
un-typed commons192 wins on CE (-0.051 vs recipe) but COSTS readability
(Spearman 0.518 vs recipe 0.601, top10 0.10). If typing transfers, this
arm should land between commons192 and the recipe on CE while beating both
on wiring readability.

Layout (exact w264 analog, param-matched by masks): 24 x 40-dim slots +
192-dim commons split into 4 typed quarters of 48; module k writes its slot
+ quarter k//6; norm/penalty segments rebuilt as 28 groups. Reuses
TypedCommons from qk_s_share_run (width-generic) and the qk_s_muon_run
training path via the same module-global patching as qk_s_w1344_run.
Outputs qk_s_w1152_typed192.{json,pt} + heldloss/f34kloss npys.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_tokenline_train as Q
Q.gpu_guard = lambda *a, **k: None

import qk_s_gate_run as G                 # WIDTH stays 1152
import qk_s_muon_run as M
import qk_s_share_run as SR
import qk_e_common as E
from qk_e_common import C, DEPTH, torch

COMMONS = 192


def make_typed192():
    C.register('S3T1152')
    torch.manual_seed(Q.SEED)
    slot = (G.WIDTH - COMMONS) // 24
    assert slot * 24 + COMMONS == G.WIDTH
    return SR.TypedCommons('S3T1152', DEPTH, [slot] * 24,
                           commons=COMMONS).to(E.DEV)


M.ARM = 'typed192'
M.CFG = dict(stem='qk_s_w1152_typed192', coeff=3e-5, prox=None, sweep=False)
M.COEFF = 3e-5
M.PROX = None
M.STEM = 'qk_s_w1152_typed192'
M.JP = os.path.join(M.OUT_DIR, f'{M.STEM}.json')
M.factory = make_typed192


def controls():
    import qk_w1152_train as W2
    W2.patch_width(G.WIDTH)
    m = make_typed192().eval().float()
    assert len(m.seg_sizes) == 28 and m.seg_sizes[-4:] == [48] * 4
    # each module writes its 40-dim slot + one 48-dim quarter
    assert (m.wmask.sum(1) == 40 + 48).all()
    # quarter q written by exactly its 6 modules (over the commons dims)
    cm = m.wmask[:, G.WIDTH - COMMONS:]
    assert float(cm.sum()) == 24 * 48
    idx = Q.HELD[:2, :Q.T]
    out = m(idx)
    assert torch.isfinite(out).all()
    body = sum(p.numel() for n, p in m.named_parameters() if 'wte' not in n)
    print(f"typed192: finite, 28 seg groups, body {body} "
          f"(recipe 286668288 {'MATCH' if body == 286668288 else 'MISMATCH'}), "
          f"penalty {float(m.custom_group_penalty()):.1f}", flush=True)
    del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
    controls()
    M.main()
