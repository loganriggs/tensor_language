"""Corrected §988 metric: is the front bilinear MLP near-linear because one factor is a near-CONSTANT GATE? For
each factor F in {Left(x), Right(x)} at a layer, compute the CONSTANT FRACTION = ||mean_over_positions(F)||^2 /
(||mean(F)||^2 + sum_units var_over_positions(F)) — the fraction of the factor's energy that is a fixed
(position-independent) constant. A near-constant GATE has HIGH constant-fraction; a content-carrying factor that
varies with the token has LOW constant-fraction. If the front (L1) has ONE factor with much higher constant-
fraction than the other (asymmetric: gate x content), that explains §941's near-linear front; if the middle
(L8/L11) has BOTH factors low (both vary), that is genuine multiplication.

REGISTERED PREDICTIONS:
  (0) SANITY: constant-fraction in [0,1] for both factors at each layer.
  (a) FRONT GATED / MIDDLE BOTH-VARY: at the FRONT (L1) the two factors' constant-fractions are ASYMMETRIC (one
      high >~0.5 = a gate), explaining §941's near-linearity; at the MIDDLE (L8/L11) both are LOWER / more
      symmetric (both vary) -> genuine multiplication;
  (b) report per-layer constant-fraction for Left and Right + the max (dominant gate) and asymmetry."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilinear_mlp_factors_v2_results.json'
NEVAL = 120; SEQ = 256; LAYERS = [0, 1, 4, 8, 11, 15]
CAP = {}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = i_[0] if isinstance(i_, tuple) else i_
                CAP[L] = (mlp.Left(x).detach().float().reshape(-1, mlp.Left.weight.shape[0]),
                          mlp.Right(x).detach().float().reshape(-1, mlp.Right.weight.shape[0]))
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    readout(x)
    for h in hs: h.remove()


@torch.no_grad()
def const_fraction(F_):  # F_: (N, HID)
    mean = F_.mean(0)  # (HID,)
    mean_energy = float((mean**2).sum())
    var_energy = float(F_.var(0).sum())
    return round(mean_energy / (mean_energy + var_energy + 1e-9), 4)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    Lx_all = {L: [] for L in LAYERS}; Rx_all = {L: [] for L in LAYERS}
    for i in range(0, nb, 4):
        forward_capture(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), LAYERS)
        for L in LAYERS:
            lx, rx = CAP[L]; n = lx.shape[0]; idx = torch.randperm(n, device=DEV)[:512]
            Lx_all[L].append(lx[idx].cpu()); Rx_all[L].append(rx[idx].cpu())
    out = {'layers': {}}
    for L in LAYERS:
        Lx = torch.cat(Lx_all[L], 0).to(DEV); Rx = torch.cat(Rx_all[L], 0).to(DEV)
        cfL = const_fraction(Lx); cfR = const_fraction(Rx)
        out['layers'][str(L)] = {'const_frac_Left': cfL, 'const_frac_Right': cfR,
                                 'max_const_frac': max(cfL, cfR), 'min_const_frac': min(cfL, cfR),
                                 'gate': 'Left' if cfL > cfR else 'Right'}
        print(f"L{L:>2}: const-frac Left {cfL} Right {cfR} | max(gate) {max(cfL,cfR)} min {min(cfL,cfR)}", flush=True)
        del Lx, Rx
    front_max = out['layers']['1']['max_const_frac']; mid_max = np.mean([out['layers'][str(L)]['max_const_frac'] for L in [8, 11]])
    out['front_L1_max_const_frac'] = front_max; out['mid_L8_11_max_const_frac'] = round(float(mid_max), 3)
    out['pred_a_front_gated'] = bool(front_max > 0.5 and front_max > mid_max + 0.1)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front(L1) max-const-frac {front_max} vs mid(L8,11) {mid_max:.3f}", flush=True)
    print(f"(a) front has a near-constant gate factor: {out['pred_a_front_gated']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
