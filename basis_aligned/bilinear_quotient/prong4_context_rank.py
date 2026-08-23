"""PRONG 4 culmination: §1041 showed the deep-middle bilinear form is CONTEXT×CONTEXT (pooled content multiplied by
itself) -- structured, but the context is high-dimensional. Is that context×context term LOW-RANK in the content
subspace (i.e. effectively topic×topic)? Restrict the context x_ctx = x - x_loc (per-token mean) to its top-K
principal directions and reconstruct the middle MLP output as loc + cross + Down[(Wl x_ctxK)⊙(Wr x_ctxK)]; measure CE
loss-recovery vs K. If a MODEST K recovers ~90%, the deep-middle is a BOUNDED topic⊗topic bilinear map -- understood.

REGISTERED PREDICTIONS:
  (0) SANITY: K = D (full) -> identity (recovery ~1); K = 0 -> loc+cross only (~ the §1041 low floor).
  (a) LOW-RANK CONTEXT (90%-per-module win): loss-recovery reaches >=0.9 at a modest K (target K <= ~128 of 1152) ->
      the deep-middle's context×context is low-rank in the content subspace; the middle is a bounded topic⊗topic
      bilinear map, understood structurally;
  (b) report loss-recovery vs K per deep-middle layer + the effective content rank (K for 90%)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'prong4_context_rank_results.json'
NCAL = 64; NEVAL = 160; SEQ = 256; LAYERS = [8, 11]; KS = [0, 16, 64, 128, 256, 512, 1152]
XBAR = {}; UCTX = {}; SUB = {'L': None, 'K': None, 'tokids': None}


def forward_logits(idx):
    SUB['tokids'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook_factory(L):
    mlp = m.transformer.h[L].mlp
    Wl = mlp.Left.weight.float(); Wr = mlp.Right.weight.float(); Dn = mlp.Down.weight.float()
    bias = (mlp.Down.bias.float() if mlp.Down.bias is not None else torch.zeros(D, device=DEV))
    def h(mo, i_, o_):
        if SUB['L'] != L or SUB['K'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D)
        xloc = XBAR[L][SUB['tokids'].reshape(-1)]; xdev = x - xloc
        K = SUB['K']
        if K > 0:
            U = UCTX[L][:, :K]; xctxK = (xdev @ U) @ U.T
        else:
            xctxK = torch.zeros_like(xdev)
        aml = xloc @ Wl.T; bml = xloc @ Wr.T
        adl = xctxK @ Wl.T; bdl = xctxK @ Wr.T
        y = ((aml*bml) + (aml*bdl + adl*bml) + (adl*bdl)) @ Dn.T + bias   # loc + cross + ctxK×ctxK
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
    V = int(m.lm_head.weight.shape[0])
    caps = {L: [] for L in LAYERS}; toks = []; hs = []
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): caps[L].append((i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D).cpu())
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    SUB['L'] = None
    for i in range(0, calib.shape[0], 8):
        idx = calib[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu()); forward_logits(idx)
    for h in hs: h.remove()
    tok = torch.cat(toks, 0).to(DEV)
    for L in LAYERS:
        X = torch.cat(caps[L], 0).to(DEV); xbar = X.mean(0, keepdim=True).repeat(V, 1)
        sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        sums.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        seen = cnts >= 5; xbar[seen] = sums[seen]/cnts[seen].unsqueeze(1); XBAR[L] = xbar
        xdev = X - xbar[tok]
        # PCA of context deviation -> top directions
        U, S, Vt = torch.linalg.svd(xdev - xdev.mean(0), full_matrices=False)
        UCTX[L] = Vt.T.contiguous()   # (D, D) principal directions
        del X, xdev
    hooks = [m.transformer.h[L].mlp.register_forward_hook(sub_hook_factory(L)) for L in LAYERS]
    SUB['L'] = None; ce_full = ce(blocks); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        SUB['L'] = L; SUB['K'] = 0; ce0 = ce(blocks); denom = max(ce0 - ce_full, 1e-6)
        rec = {}
        for K in KS:
            SUB['K'] = K; c = ce(blocks); rec[str(K)] = round(float((ce0 - c)/denom), 4)
        SUB['L'] = None
        eff = next((K for K in KS if rec[str(K)] >= 0.9), None)
        out['layers'][str(L)] = {'loc_cross_floor_cost': round(ce0-ce_full,4), 'recovery_by_K': rec, 'eff_context_rank_90': eff}
        print(f"L{L}: floor(loc+cross) cost {ce0-ce_full:.3f} | context-rank recovery {rec} | eff-rank {eff}", flush=True)
    for h in hooks: h.remove()
    out['eff_ranks'] = [out['layers'][str(L)]['eff_context_rank_90'] for L in LAYERS]
    out['mid_recovery_128'] = round(float(np.mean([out['layers'][str(L)]['recovery_by_K']['128'] for L in LAYERS])), 3)
    out['pred_a_low_rank_context'] = bool(all(e is not None and e <= 128 for e in out['eff_ranks']))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"deep-middle context-rank recovery@128 {out['mid_recovery_128']} | eff-ranks {out['eff_ranks']}", flush=True)
    print(f"pred_a low-rank context (topic x topic, bounded): {out['pred_a_low_rank_context']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
