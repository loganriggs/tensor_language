"""Effective-param-matched recipe: slots at width 1344 vs vanilla at 1152
(Logan 2026-08-06: "slots with a reduced decoder at layers and compensate
with a larger residual stream").

The hard write partition makes each block's c_proj/Down use only its slot's
rows, so the w1152 recipe's EFFECTIVE body is 210.35M vs vanilla's 286.65M
-- 27% of the apparent +0.143 tax is dead decoder rows. Dense reduced
decoders + wider residual fixes the accounting for real: at Dm=1344
(21 heads x 64, 24 slots x 56) the recipe's effective body is
12 x (13*Dm^2 + 5*Dm^2/24) = 286.3M ~= vanilla-1152's 286.65M.

(The masked full-width decoder IS functionally identical to a dense
56-row decoder -- the mask just wastes memory, not effective params --
so this arm is simply the combo3e5loss recipe run at width 1344.)

Prediction if the effective-param story is the whole story: final f34k CE
~= vanilla 4.0116 + residual tax 0.06-0.08 -> 4.07-4.09, vs the w1152
recipe's 4.155. Param-matched, NOT compute-matched (reads at 1344^2,
~1.36x recipe FLOPs) -- recorded in the arm metadata.

Reuses qk_s_muon_run's entire training path by patching its module globals
(ARM binds to 'base' on import, so CFG/COEFF/STEM/factory are re-pointed
explicitly). Outputs qk_s_w1344_recipe.{json,pt} + heldloss/f34kloss npys.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_tokenline_train as Q
Q.gpu_guard = lambda *a, **k: None

import qk_s_gate_run as G
G.WIDTH = 1344

import qk_s_muon_run as M
import qk_s_e1_run as E1R

M.ARM = 'combo3e5loss_w1344'
M.CFG = dict(stem='qk_s_w1344_recipe', coeff=3e-5, prox=None, sweep=False)
M.COEFF = 3e-5
M.PROX = None
M.STEM = 'qk_s_w1344_recipe'
M.JP = os.path.join(M.OUT_DIR, f'{M.STEM}.json')
M.factory = E1R.make_e1


def controls():
    import torch
    import qk_w1152_train as W2
    W2.patch_width(G.WIDTH)
    m = M.factory().eval().float()
    idx = Q.HELD[:2, :Q.T]
    out = m(idx)
    assert torch.isfinite(out).all()
    body = sum(p.numel() for n, p in m.named_parameters() if 'wte' not in n)
    # effective body: decoders (c_proj, Down) use only their 56-dim slot rows
    slot, Dm = G.WIDTH // 24, G.WIDTH
    dead = sum((Dm - slot) * W.shape[1] for blk in m.h
               for W in (blk.c_proj.weight, blk.Down.weight))
    eff = body - dead
    print(f"w1344 recipe: body {body} ({body/1e6:.1f}M), effective "
          f"{eff} ({eff/1e6:.1f}M) vs vanilla-1152 286.65M", flush=True)
    assert abs(eff - 286.65e6) / 286.65e6 < 0.02, eff
    del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
    controls()
    M.main()
