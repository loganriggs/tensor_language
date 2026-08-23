"""PRONG 1 (done right, after §1036's optimization failure): is the middle a LOW-RANK BILINEAR computation we can
capture? Don't fit a generic bilinear map -- use the model's OWN bilinear neurons. Each MLP is a sum of HID rank-1
bilinear neurons: y = a @ Down.T + bias, where a = (Wl x) ⊙ (Wr x) is the (n, HID) neuron-activation matrix. Rank
neurons by importance (||Down column|| × std of activation), keep the top-R, and measure held-out reconstruction R²
of the output vs R. This is EXACT at full rank (R = HID) by construction, and the effective rank (smallest R with
R² ≥ 0.9) says how compactly the middle's bilinear computation can be expressed -- the principled test of the
"why not multilinear?" challenge.

REGISTERED PREDICTIONS:
  (0) SANITY: R = HID reconstructs the output exactly (R² ≈ 1.0); linear (§1005) only reached ~0.15-0.55.
  (a) LOW EFFECTIVE BILINEAR RANK: modest R (target ≤ ~1024 of HID≈4608, ideally ≤256) reaches R² ≥ 0.9 at the
      MIDDLE layers where the LINEAR stand-in was ~0.15-0.25 -> the middle is a bounded-rank bilinear computation,
      NOT irreducible; the "1.6-nat ceiling" was a LINEAR-stand-in artifact;
  (b) report held-out R² vs R and the effective rank per layer, and the linear-vs-bilinear gap."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilinear_neuron_rank_results.json'
NEVAL = 200; SEQ = 256; LAYERS = [1, 6, 8, 11, 15]
NTRAIN = 8000; NTEST = 4000
CAP = {}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = i_[0] if isinstance(i_, tuple) else i_
                CAP[L] = x.detach().float().reshape(-1, D)
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    readout(x)
    for h in hs: h.remove()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); need = NTRAIN + NTEST
    Xs = {L: [] for L in LAYERS}
    for i in range(0, blocks.shape[0], 4):
        forward_capture(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), LAYERS)
        for L in LAYERS:
            x = CAP[L]; k = x.shape[0]; idx = torch.randperm(k, device=DEV)[:200]
            Xs[L].append(x[idx].cpu())
        if sum(t.shape[0] for t in Xs[LAYERS[0]]) >= need + 512: break
    out = {'layers': {}}
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        Wl = mlp.Left.weight.float(); Wr = mlp.Right.weight.float(); Dn = mlp.Down.weight.float()  # (HID,D),(HID,D),(D,HID)
        bias = (mlp.Down.bias.float() if mlp.Down.bias is not None else torch.zeros(D, device=DEV))
        HID = Wl.shape[0]
        X = torch.cat(Xs[L], 0)[:need].to(DEV)
        Xtr = X[:NTRAIN]; Xte = X[NTRAIN:]
        a_tr = (Xtr @ Wl.T) * (Xtr @ Wr.T)   # (ntr, HID)
        a_te = (Xte @ Wl.T) * (Xte @ Wr.T)   # (nte, HID)
        Yte = a_te @ Dn.T + bias             # true output on test (exact)
        Yvar = ((Yte - Yte.mean(0))**2).sum()
        imp = a_tr.std(0) * Dn.norm(dim=0)   # (HID,) importance per neuron
        order = torch.argsort(imp, descending=True)
        RANKS = [16, 64, 256, 1024, HID]
        r2s = {}
        for R in RANKS:
            sel = order[:R]
            yR = a_te[:, sel] @ Dn[:, sel].T + bias
            r2s[str(R)] = round(float(1.0 - ((Yte - yR)**2).sum() / Yvar), 4)
        eff = next((R for R in RANKS if r2s[str(R)] >= 0.9), None)
        out['layers'][str(L)] = {'HID': HID, 'r2_by_rank': r2s, 'eff_rank_90': eff}
        print(f"L{L:>2} (HID {HID}): R2 by rank {r2s} | eff_rank_90 {eff}", flush=True)
        del X, a_tr, a_te, Yte
    mids = [8, 11, 15]
    out['mid_r2_at_256'] = round(float(np.mean([out['layers'][str(L)]['r2_by_rank']['256'] for L in mids])), 3)
    out['mid_r2_at_1024'] = round(float(np.mean([out['layers'][str(L)]['r2_by_rank']['1024'] for L in mids])), 3)
    out['mid_eff_ranks'] = [out['layers'][str(L)]['eff_rank_90'] for L in mids]
    out['pred_a_low_rank_bilinear'] = bool(out['mid_r2_at_1024'] > 0.9 or out['mid_r2_at_256'] > 0.9)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"MIDDLE (L8/11/15): R2@256 {out['mid_r2_at_256']} | R2@1024 {out['mid_r2_at_1024']} | eff-ranks {out['mid_eff_ranks']}", flush=True)
    print(f"pred_a middle is low-rank bilinear: {out['pred_a_low_rank_bilinear']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
