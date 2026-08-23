"""What CREATES the content seed? §1073 showed the content is ~linearly built from the processed stream and 38% present
after one layer (nonlinearity front-loaded). Decompose the early layers into their attention vs MLP steps and see which
drives the content's linear predictability. At each early layer L=0..3, capture the residual at three points: BEFORE
attn (block input), AFTER attn / BEFORE mlp (mlp input), and AFTER mlp (block output). Predict the pooled-L8-12 content
coordinate from each point (held-out R^2). The rise across the ATTN step (before->mid) vs the MLP step (mid->after)
tells whether the content seed is created by attention (context gathering) or the bilinear MLP (the content x content
multiply).

REGISTERED PREDICTIONS:
  (0) SANITY: R^2 rises monotonically through the early stream (consistent with §1073's cumulative 0.38@L1->...).
  (a) MLP (BILINEAR MULTIPLY) CREATES THE CONTENT: the per-layer R^2 gain across MLP steps (mid->after) exceeds the gain
      across attn steps (before->mid), summed over L0-3 -> the bilinear MLP builds the content (consistent with the
      content being a content x content product, §1041), attention mainly gathers/moves;
  (b) report R^2 at each point + attn-step vs mlp-step gains."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_seed_results.json'
NEVAL = 240; SEQ = 256; EARLY = [0, 1, 2, 3]; TGT = [8, 10, 12]; K = 64; RIDGE = 1e2
CAP = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return x


def capture(idx):
    hs = []
    for L in EARLY:
        blk = m.transformer.h[L]; mlp = m.transformer.h[L].mlp
        def mkb(L):
            def h(mo, i_, o_):
                CAP[('before', L)].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                CAP[('after', L)].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        def mkm(L):
            def h(mo, i_, o_): CAP[('mid', L)].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(blk.register_forward_hook(mkb(L))); hs.append(mlp.register_forward_hook(mkm(L)))
    for L in TGT:
        def mkt(L):
            def h(mo, i_, o_): CAP[('tgt', L)].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mkt(L)))
    fwd(idx)
    for h in hs: h.remove()


def dev(X, tok, V):
    xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    xbar = xbar / cnts.clamp_min(1).unsqueeze(1); return X - xbar[tok]


def r2(Xtr, Ytr, Xte, Yte):
    X1 = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1)
    M = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(X1.shape[1], device=DEV), X1.T @ Ytr)
    Xt1 = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1); pred = Xt1 @ M
    return float(1 - ((Yte-pred)**2).sum()/((Yte-Yte.mean(0))**2).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    for L in EARLY:
        for p in ('before', 'mid', 'after'): CAP[(p, L)] = []
    for L in TGT: CAP[('tgt', L)] = []
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); capture(idx)
    tok = torch.cat(idsL, 0)
    # target content coordinate
    devsum = None
    for L in TGT:
        dv = dev(torch.cat(CAP[('tgt', L)], 0), tok, V); devsum = dv if devsum is None else devsum + dv; CAP[('tgt', L)] = []
    devc = devsum/len(TGT); devc = devc - devc.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False); U = Vt[:K].T.contiguous(); Y = devc @ U
    n = Y.shape[0]; ntr = int(0.7*n); Ytr, Yte = Y[:ntr], Y[ntr:]
    pts = {}
    for L in EARLY:
        for p in ('before', 'mid', 'after'):
            Xd = dev(torch.cat(CAP[(p, L)], 0), tok, V); pts[(p, L)] = round(r2(Xd[:ntr], Ytr, Xd[ntr:], Yte), 4); CAP[(p, L)] = []
    out = {'K': K, 'tgt': TGT, 'n': n, 'r2_points': {f'{p}_L{L}': pts[(p, L)] for L in EARLY for p in ('before', 'mid', 'after')}}
    attn_gain = 0.0; mlp_gain = 0.0
    for L in EARLY:
        attn_gain += pts[('mid', L)] - pts[('before', L)]     # attn step
        mlp_gain += pts[('after', L)] - pts[('mid', L)]        # mlp step
    out['attn_step_total_gain'] = round(attn_gain, 4); out['mlp_step_total_gain'] = round(mlp_gain, 4)
    out['pred_a_mlp_creates_content'] = bool(mlp_gain > attn_gain)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    for L in EARLY:
        print(f"L{L}: before {pts[('before',L)]} -> [attn] mid {pts[('mid',L)]} -> [mlp] after {pts[('after',L)]}", flush=True)
    print(f"total attn-step gain {out['attn_step_total_gain']} | mlp-step gain {out['mlp_step_total_gain']} | pred_a mlp-creates {out['pred_a_mlp_creates_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
