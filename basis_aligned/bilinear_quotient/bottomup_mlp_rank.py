"""PRONG 1, loss-relevant (follows §1037). §1037 showed the middle output is high-rank in the neuron basis BY OUTPUT
VARIANCE -- but output variance includes loss-irrelevant directions (massive activations / norm channels, §737/§748).
The right question for UNDERSTANDING is: how many bilinear neurons does the LOSS need? Replace a middle MLP's output
with its top-R-neuron reconstruction (using the model's real neurons), and measure CE LOSS-RECOVERY vs R:
  recovery(R) = (CE[top-R reconstruction] mapped as: (CE_meanablate_L - CE_R)/(CE_meanablate_L - CE_full).
If loss-recovery reaches ~0.9 at a MUCH lower R than the output-R² did (§1037: R²@256=0.16), then the loss-relevant
content is LOW-RANK bilinear even though the raw output isn't -> the middle is understandable as a low-rank bilinear
map for the part that matters.

REGISTERED PREDICTIONS:
  (0) SANITY: R=0 (bias only) ~ mean-ablate (recovery ~0); R=HID (full) = exact (recovery ~1).
  (a) LOSS IS LOWER-RANK THAN OUTPUT VARIANCE: per middle layer, loss-recovery reaches >=0.9 at an R where the
      output-R² (§1037) was well below 0.9 (e.g. recovery(256) >> R²(256)=0.16) -> the loss-relevant bilinear
      content is low-rank; most of the high output-rank is loss-irrelevant;
  (b) report loss-recovery vs R per middle layer + the effective loss-rank (R for recovery>=0.9)."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bottomup_mlp_rank_results.json'
NCAL = 48; NEVAL = 160; SEQ = 256; LAYERS = list(range(18)); RANKS = [0, 16, 64, 256, 1024, 4608]
ORD = {}; SUB = {'L': None, 'R': None}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook_factory(L):
    mlp = m.transformer.h[L].mlp
    Wl = mlp.Left.weight.float(); Wr = mlp.Right.weight.float(); Dn = mlp.Down.weight.float()
    bias = (mlp.Down.bias.float() if mlp.Down.bias is not None else torch.zeros(D, device=DEV))
    def h(mo, i_, o_):
        if SUB['L'] != L: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        R = SUB['R']
        if R == 0: return bias.expand_as(o_[0] if isinstance(o_, tuple) else o_).to((o_[0] if isinstance(o_,tuple) else o_).dtype)
        sel = ORD[L][:R]
        a = (x.reshape(-1, D) @ Wl[sel].T) * (x.reshape(-1, D) @ Wr[sel].T)   # (n, R)
        y = a @ Dn[:, sel].T + bias
        o = o_[0] if isinstance(o_, tuple) else o_
        return y.reshape(o.shape).to(o.dtype)
    return h


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL)
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    # compute neuron importance order per layer on calib (std of activation x Down-col-norm; §1037 ranking)
    caps = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = (i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D)
                caps[L].append(((x @ mlp.Left.weight.float().T) * (x @ mlp.Right.weight.float().T)).detach().cpu())
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    for i in range(0, calib.shape[0], 8): forward_logits(calib[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    for L in LAYERS:
        a = torch.cat(caps[L], 0).to(DEV); imp = a.std(0) * m.transformer.h[L].mlp.Down.weight.float().norm(dim=0)
        ORD[L] = torch.argsort(imp, descending=True); del a
    hooks = [m.transformer.h[L].mlp.register_forward_hook(sub_hook_factory(L)) for L in LAYERS]
    SUB['L'] = None; ce_full = ce(blocks); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        SUB['L'] = L; SUB['R'] = 0; ce_ma = ce(blocks); denom = max(ce_ma - ce_full, 1e-6)
        rec = {}
        for R in RANKS:
            SUB['R'] = R; c = ce(blocks); rec[str(R)] = round(float((ce_ma - c)/denom), 4)
        SUB['L'] = None
        eff = next((R for R in RANKS if rec[str(R)] >= 0.9), None)
        out['layers'][str(L)] = {'meanabl_cost': round(ce_ma - ce_full, 4), 'loss_recovery_by_rank': rec, 'eff_loss_rank_90': eff}
        print(f"L{L:>2} (meanabl {ce_ma-ce_full:.3f}): loss-recovery {rec} | eff-loss-rank {eff}", flush=True)
    for h in hooks: h.remove()
    effr = {L: out['layers'][str(L)]['eff_loss_rank_90'] for L in LAYERS}
    out['eff_loss_rank_by_layer'] = {str(L): effr[L] for L in LAYERS}
    frontok = [effr[L] for L in [0, 1, 2, 3] if effr[L] is not None]
    out['pred_a_front_low_rank'] = bool(len(frontok) >= 2 and all(r <= 256 for r in frontok))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print("\nBOTTOM-UP effective loss-rank (R for 90% recovery) per MLP:", flush=True)
    for L in LAYERS: print(f"  mlp{L:>2}: eff-rank {effr[L]}  {out['layers'][str(L)]['loss_recovery_by_rank']}", flush=True)
    print(f"pred_a front MLPs low-rank: {out['pred_a_front_low_rank']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
