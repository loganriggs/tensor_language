"""IS THE READOUT (L16-17) A NAMEABLE LINEAR MAP of the L15 residual? §944: the last two blocks close a 2.56-nat
gap (logit-lens L15 5.82 -> final 3.26), and §941/§942 say they are near-LINEAR. If so, a single fitted LINEAR
map from the L15 residual to the L17 residual, followed by the model's own readout head, should recover most of
that gap. Fit the ridge map on train rows, evaluate CE on held-out rows. Compare to: the L15 logit-lens (no
transform, ~5.82), the true final CE (3.26), and a shuffled-map null.

REGISTERED PREDICTIONS:
  (0) SANITY: L15 logit-lens CE reproduces §944 (~5.8); true final CE ~3.26; shuffled-map null ~ logit-lens.
  (a) READOUT IS ~LINEAR: a linear map L15-residual -> L17-residual (then the readout head) recovers MOST of the
      2.56-nat gap (held-out CE close to 3.26, well below the 5.82 logit-lens) -> the last two blocks are a
      nameable near-linear rotation of the residual into the token basis;
  (b) report held-out CE for: L15 logit-lens, linear-map, true final, shuffled null; recovered fraction of the gap."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'readout_linear_results.json'
NEVAL = 200; SEQ = 256; L_SRC = 15; L_DST = 17; RIDGE_MAP = 1e3


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def cap_src_dst(idx):
    res = {}
    def mk(L):
        def h(mo, i_, o_): res[L] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
        return h
    hs = [m.transformer.h[L_SRC].register_forward_hook(mk(L_SRC)), m.transformer.h[L_DST].register_forward_hook(mk(L_DST))]
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    for h in hs: h.remove()
    return res[L_SRC], res[L_DST]


@torch.no_grad()
def ce_from_resid(resid_flat, tgt_flat):
    lg = readout(resid_flat).float()
    return float(F.cross_entropy(lg, tgt_flat, reduction='mean'))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]; ntr = int(0.7*nb)
    TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True
    src = []; dst = []; tgt = []
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous()
        s, dd = cap_src_dst(idx); src.append(s.reshape(-1, D)); dst.append(dd.reshape(-1, D))
        tgt.append(bb[:, 1:].reshape(-1))
    Src = torch.cat(src, 0); Dst = torch.cat(dst, 0); Tgt = torch.cat(tgt, 0)
    trm = torch.tensor(np.repeat(TRAIN, SEQ-1), device=DEV)
    Xtr = Src[trm]; Ytr = Dst[trm]; Xte = Src[~trm]; Yte = Dst[~trm]; Tte = Tgt[~trm]
    # fit ridge linear map src->dst on train
    A = Xtr.T @ Xtr + RIDGE_MAP*torch.eye(D, device=DEV); M = torch.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ M
    # shuffled-map null
    pr = torch.randperm(Xtr.shape[0], generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
    Msh = torch.linalg.solve(A, Xtr.T @ Ytr[pr]); pred_sh = Xte @ Msh
    ce_lens = ce_from_resid(Xte, Tte)          # L15 logit-lens (no transform)
    ce_final = ce_from_resid(Yte, Tte)         # true final
    ce_linmap = ce_from_resid(pred, Tte)       # linear map L15->L17 then readout
    ce_shuf = ce_from_resid(pred_sh, Tte)      # shuffled-map null
    gap = ce_lens - ce_final
    recovered = (ce_lens - ce_linmap)/max(gap, 1e-6)
    out = {'ce_L15_logitlens': round(ce_lens, 4), 'ce_final': round(ce_final, 4),
           'ce_linear_map': round(ce_linmap, 4), 'ce_shuffled_null': round(ce_shuf, 4),
           'gap_L15_to_final': round(gap, 4), 'gap_recovered_frac': round(float(recovered), 4)}
    out['pred_a_readout_linear'] = bool(recovered > 0.6 and ce_shuf > ce_linmap + 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"L15 logit-lens {ce_lens:.4f} | linear-map {ce_linmap:.4f} | final {ce_final:.4f} | shuffled {ce_shuf:.4f}", flush=True)
    print(f"gap {gap:.4f}, linear map recovers {recovered:.3f} of it", flush=True)
    print(f"(a) readout is a nameable ~linear map of L15: {out['pred_a_readout_linear']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
