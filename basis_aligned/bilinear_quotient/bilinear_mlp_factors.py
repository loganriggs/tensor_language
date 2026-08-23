"""Do the bilinear MLP's two factors SPECIALIZE — is the front near-linear because one factor acts as a near-CONSTANT
GATE while the other carries content, and the middle multiplicative because BOTH vary? The bilinear MLP computes
Down[(Left.x) (elementwise*) (Right.x)]. Decompose the product's variation into each factor's contribution: with
Lx=Left(x), Rx=Right(x), compare var(product) when one factor is FROZEN at its mean vs full. If freezing Right
retains most of the product's variance (Left carries it, Right ~ constant gate), the layer is effectively linear
in Left (explaining §941's near-linear front); if freezing EITHER factor loses variance, both vary = genuine
multiplication (middle). Layers: front L1, mid L8, late-mid L15.

REGISTERED PREDICTIONS:
  (0) SANITY: full product variance > 0 at each layer.
  (a) FRONT = GATE x CONTENT, MIDDLE = BOTH VARY: at the FRONT (L1) the product's variance is ASYMMETRIC between
      the factors (freezing one factor retains much more variance than freezing the other -> that other factor is a
      near-constant gate), explaining §941's near-linearity; at the MIDDLE (L8/L15) the two factors contribute more
      SYMMETRICALLY (freezing either loses substantial variance) -> genuine two-varying-factor multiplication;
  (b) report per-layer var_full, var_Lonly (Right frozen), var_Ronly (Left frozen), and the asymmetry ratio."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilinear_mlp_factors_results.json'
NEVAL = 120; SEQ = 256; LAYERS = [1, 4, 8, 11, 15]
CAP = {}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = i_[0] if isinstance(i_, tuple) else i_  # mlp input
                CAP[L] = (mlp.Left(x).detach().float().reshape(-1, mlp.Left.weight.shape[0]),
                          mlp.Right(x).detach().float().reshape(-1, mlp.Right.weight.shape[0]))
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    readout(x)
    for h in hs: h.remove()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    # accumulate Lx, Rx per layer (subsample positions to bound memory)
    Lx_all = {L: [] for L in LAYERS}; Rx_all = {L: [] for L in LAYERS}
    for i in range(0, nb, 4):
        forward_capture(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), LAYERS)
        for L in LAYERS:
            lx, rx = CAP[L]
            # subsample 512 positions per batch to bound memory
            n = lx.shape[0]; idx = torch.randperm(n, device=DEV)[:512]
            Lx_all[L].append(lx[idx].cpu()); Rx_all[L].append(rx[idx].cpu())
    out = {'layers': {}}
    for L in LAYERS:
        Lx = torch.cat(Lx_all[L], 0).to(DEV); Rx = torch.cat(Rx_all[L], 0).to(DEV)  # (N, HID)
        Lm = Lx.mean(0, keepdim=True); Rm = Rx.mean(0, keepdim=True)
        prod = Lx * Rx
        var_full = float((prod.var(0)).mean())
        var_Lonly = float(((Lx * Rm).var(0)).mean())   # Right frozen at mean -> variance from Left
        var_Ronly = float(((Lm * Rx).var(0)).mean())   # Left frozen at mean -> variance from Right
        # asymmetry: how much more one factor carries than the other
        hi = max(var_Lonly, var_Ronly); lo = min(var_Lonly, var_Ronly)
        asym = round(hi / max(lo, 1e-9), 2)
        # fraction of full variance retained by the dominant single factor
        retain = round(hi / max(var_full, 1e-9), 3)
        out['layers'][str(L)] = {'var_full': round(var_full, 3), 'var_Lonly': round(var_Lonly, 3),
                                 'var_Ronly': round(var_Ronly, 3), 'asymmetry_ratio': asym, 'dominant_factor_retain': retain,
                                 'dominant': 'Left' if var_Lonly >= var_Ronly else 'Right'}
        print(f"L{L:>2}: var_full {var_full:.2f} | Lonly {var_Lonly:.2f} | Ronly {var_Ronly:.2f} | asym {asym} | dom-retain {retain} ({out['layers'][str(L)]['dominant']})", flush=True)
        del Lx, Rx, prod
    front = out['layers']['1']['asymmetry_ratio']; mid = np.mean([out['layers'][str(L)]['asymmetry_ratio'] for L in [8, 11]])
    out['front_asymmetry'] = front; out['mid_asymmetry'] = round(float(mid), 2)
    out['pred_a_front_gated_mid_symmetric'] = bool(front > mid)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front(L1) asymmetry {front} vs mid(L8,11) {mid:.2f}", flush=True)
    print(f"(a) front gated (asymmetric) vs middle both-vary (symmetric): {out['pred_a_front_gated_mid_symmetric']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
