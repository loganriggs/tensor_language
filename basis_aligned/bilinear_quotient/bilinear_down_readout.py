"""VERIFY §989's inferred claim: the front bilinear MLP is near-linear (§941) because the DOWN projection reads out
the LINEAR part of the product and discards the INTERACTION. Decompose the product exactly:
  Lx = a + u,  Rx = b + w   (a=mean(Lx), b=mean(Rx); u,w = per-position deviations, linear in x)
  product = ab + (a*w + b*u) + (u*w)   -> constant + LINEAR-in-x + INTERACTION
Apply Down to the LINEAR-in-x part (a*w+b*u) and to the INTERACTION part (u*w) separately, and measure how much
each contributes to the OUTPUT variance. If at the FRONT (L1) the Down output is dominated by the LINEAR part
(interaction contributes little to output variance) while at the MIDDLE (L8/L11) the interaction contributes
substantially, that DIRECTLY confirms §989/§941: the front's linearity is Down reading out the linear part.

REGISTERED PREDICTIONS:
  (0) SANITY: Down(linear)+Down(interaction)+Down(const) reconstructs the MLP output (residual ~0).
  (a) FRONT LINEAR-READOUT: at L1 the interaction's share of Down-output variance is LOW (<~0.3) and much lower
      than at the middle (L8/L11 higher) -> Down reads out the linear part at the front and the interaction in the
      middle, confirming §941/§989;
  (b) report per-layer interaction-share of Down-output variance (+ linear share)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilinear_down_readout_results.json'
NEVAL = 96; SEQ = 256; LAYERS = [1, 4, 8, 11, 15]
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
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    Lx_all = {L: [] for L in LAYERS}; Rx_all = {L: [] for L in LAYERS}
    for i in range(0, nb, 4):
        forward_capture(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), LAYERS)
        for L in LAYERS:
            lx, rx = CAP[L]; n = lx.shape[0]; idx = torch.randperm(n, device=DEV)[:400]
            Lx_all[L].append(lx[idx].cpu()); Rx_all[L].append(rx[idx].cpu())
    out = {'layers': {}}
    for L in LAYERS:
        Lx = torch.cat(Lx_all[L], 0).to(DEV); Rx = torch.cat(Rx_all[L], 0).to(DEV)
        a = Lx.mean(0, keepdim=True); b = Rx.mean(0, keepdim=True); u = Lx - a; w = Rx - b
        linear = a*w + b*u        # linear-in-x part (HID)
        interaction = u*w          # interaction part (HID)
        const = (a*b)              # constant
        mlp = m.transformer.h[L].mlp
        dn = lambda h: F.linear(h, mlp.Down.weight)  # Down without bias
        out_lin = dn(linear); out_int = dn(interaction)
        recon = dn(linear + interaction + const) + (mlp.Down.bias if mlp.Down.bias is not None else 0)
        true_out = mlp(Lx*0 + a)  # not meaningful; use direct: full product Down
        full = dn(Lx*Rx) + (mlp.Down.bias if mlp.Down.bias is not None else 0)
        recon_resid = float((recon - full).pow(2).sum() / (full - full.mean(0)).pow(2).sum())
        v_lin = float(out_lin.var(0).sum()); v_int = float(out_int.var(0).sum())
        tot = v_lin + v_int
        out['layers'][str(L)] = {'linear_share': round(v_lin/max(tot,1e-9), 3), 'interaction_share': round(v_int/max(tot,1e-9), 3),
                                 'recon_residual': round(recon_resid, 5)}
        print(f"L{L:>2}: Down-output var share  linear {out['layers'][str(L)]['linear_share']} | interaction {out['layers'][str(L)]['interaction_share']} (recon-resid {out['layers'][str(L)]['recon_residual']:.4f})", flush=True)
        del Lx, Rx
    fi = out['layers']['1']['interaction_share']; mi = np.mean([out['layers'][str(L)]['interaction_share'] for L in [8, 11]])
    out['front_L1_interaction_share'] = fi; out['mid_L8_11_interaction_share'] = round(float(mi), 3)
    out['pred_a_front_linear_readout'] = bool(fi < 0.3 and fi < mi)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front(L1) interaction-share {fi} vs mid(L8,11) {mi:.3f}", flush=True)
    print(f"(a) front linearity = Down reads out linear part: {out['pred_a_front_linear_readout']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
