"""§1072 showed the deep-middle content is NOT a linear bag of raw context embeddings. Is it instead linearly built from
the PROCESSED residual stream, and from how early? Trace construction: predict the pooled-L8-12 content coordinate
(top-64 PCA of content deviation) from the residual (mlp input) at each EARLIER layer L=1..7, held-out R^2. Where R^2
rises locates where the content becomes linearly determined. Compare to the raw-embedding bag baseline (§1072, ~0).
Complements §1052 (subspace-overlap onset) with a functional predictability measure.

REGISTERED PREDICTIONS:
  (0) SANITY: raw-embedding bag baseline R^2 ~0 (§1072); predicting from L8-12's own residual would be ~1 (not tested,
      trivial).
  (a) LINEARLY BUILT FROM THE PROCESSED STREAM: R^2 predicting the L8-12 content from earlier residuals is FAR above the
      bag baseline and RISES with layer (L1 low -> L7 high), locating the content's construction in the transition/early-
      middle -> the content is progressively, ~linearly accumulated from the processed representation (not a bag, but
      traceable);
  (b) report R^2 by source layer + bag baseline."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_construction_results.json'
NEVAL = 240; SEQ = 256; SRC = [1, 2, 3, 4, 5, 6, 7]; TGT = [8, 10, 12]; K = 64; RIDGE = 1e2
CAP = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return x


def capture(idx, layers):
    hs = []
    for L in layers:
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    fwd(idx)
    for h in hs: h.remove()


def token_dev(X, tok, V):
    xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
    return X - xbar[tok]


def r2(Xtr, Ytr, Xte, Yte):
    X1 = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1)
    M = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(X1.shape[1], device=DEV), X1.T @ Ytr)
    Xt1 = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1)
    pred = Xt1 @ M; return float(1 - ((Yte-pred)**2).sum()/((Yte-Yte.mean(0))**2).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    alllayers = SRC + TGT
    for L in alllayers: CAP[L] = []
    idsL = []; embs = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1))
        E = F.rms_norm(m.transformer.wte(idx), (D,)).float(); ar = torch.arange(1, idx.shape[1]+1, device=DEV).view(1, idx.shape[1], 1)
        embs.append((torch.cumsum(E, 1)/ar).reshape(-1, D))     # causal-mean bag baseline
        capture(idx, alllayers)
    tok = torch.cat(idsL, 0); bag = torch.cat(embs, 0)
    # target: pooled L8-12 content coordinate
    devsum = None
    for L in TGT:
        dv = token_dev(torch.cat(CAP[L], 0), tok, V); devsum = dv if devsum is None else devsum + dv
    devc = devsum/len(TGT); devc = devc - devc.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False); U = Vt[:K].T.contiguous(); Y = devc @ U
    n = Y.shape[0]; ntr = int(0.7*n); Ytr, Yte = Y[:ntr], Y[ntr:]
    out = {'K': K, 'tgt_layers': TGT, 'n': n, 'r2_by_source_layer': {}}
    for L in SRC:
        Xdev = token_dev(torch.cat(CAP[L], 0), tok, V)          # source residual content deviation
        out['r2_by_source_layer'][str(L)] = round(r2(Xdev[:ntr], Ytr, Xdev[ntr:], Yte), 4)
        CAP[L] = []
        print(f"predict L8-12 content from L{L} residual-dev: R2 {out['r2_by_source_layer'][str(L)]}", flush=True)
    out['r2_bag_baseline'] = round(r2(bag[:ntr], Ytr, bag[ntr:], Yte), 4)
    lo = out['r2_by_source_layer']['1']; hi = out['r2_by_source_layer']['7']
    out['rises_L1_to_L7'] = bool(hi > lo + 0.1)
    out['pred_a_linearly_built'] = bool(hi > 0.5 and hi > 3*max(out['r2_bag_baseline'], 0.01) and out['rises_L1_to_L7'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"bag baseline R2 {out['r2_bag_baseline']} | rises L1->L7 {out['rises_L1_to_L7']} | pred_a {out['pred_a_linearly_built']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
