"""Is the deep-middle content literally a RECENCY-WEIGHTED BAG-OF-WORDS of the context? The mechanism account (§894/930)
says the content is a running bag of recent word content; §1065 showed it is multi-scale (running local context). Test
constructively: can we predict the deep-middle content coordinate from an explicit bag of context TOKEN EMBEDDINGS
(bypassing the model's own computation)? Fit ridge maps from bag features -- current-token embedding, causal running
mean of embeddings, exponential-decay running means -- to the content coordinate (top-64 PCA of pooled L8-12 content
deviation), held-out, and report R^2. If a smoothed bag predicts the content far above a current-token-only baseline,
the content IS a bag-of-words topic representation, understood constructively; if not, it needs the model's nonlinear
processing (more than a bag).

REGISTERED PREDICTIONS:
  (0) SANITY: current-token-only R^2 is the floor; a shuffled-position bag predicts ~0.
  (a) CONTENT IS A BAG-OF-WORDS: the recency-weighted bag of context embeddings predicts the content coordinate with
      R^2 substantially above the current-token-only baseline (target: bag R^2 > 2x cur-only, and > ~0.4) -> the deep-
      middle content is constructively a smoothed bag of recent word content;
  (b) report R^2 for cur-only / cummean / exp-decay / full-bag + shuffled null."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_from_bag_results.json'
NEVAL = 240; SEQ = 256; REF = [8, 10, 12]; K = 64; RIDGE = 1e2
CAP = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return x


def capture(idx):
    hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float())
            return h
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mk(L)))
    fwd(idx)
    for h in hs: h.remove()


def bag_feats(idx):
    """(B,T,D) features from context token embeddings: cur, causal cummean, exp-decay means (lam .9,.99)."""
    E = F.rms_norm(m.transformer.wte(idx), (D,)).float(); B, T, _ = E.shape
    ar = torch.arange(1, T+1, device=DEV).view(1, T, 1)
    cummean = torch.cumsum(E, 1) / ar
    def expmean(lam):
        out = torch.zeros_like(E); acc = torch.zeros(B, D, device=DEV); norm = 0.0
        cols = []
        for t in range(T):
            acc = lam*acc + E[:, t]; norm = lam*norm + 1.0; cols.append(acc/norm)
        return torch.stack(cols, 1)
    e9 = expmean(0.9); e99 = expmean(0.99)
    return {'cur': E, 'cummean': cummean, 'exp9': e9, 'exp99': e99}


@torch.no_grad()
def content_coord(blocks):
    for L in REF: CAP[L] = []
    idsL = []; bags = {k: [] for k in ('cur', 'cummean', 'exp9', 'exp99')}
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1))
        bf = bag_feats(idx)
        for k in bags: bags[k].append(bf[k].reshape(-1, D))
        capture(idx)
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for L in REF:
        X = torch.cat(CAP[L], 0).reshape(-1, D); xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X; CAP[L] = []
    dev = devsum/len(REF); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False); U = Vt[:K].T.contiguous()
    coord = devc @ U
    return coord, {k: torch.cat(bags[k], 0) for k in bags}


def r2(Xtr, Ytr, Xte, Yte):
    X1 = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1)
    M = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(X1.shape[1], device=DEV), X1.T @ Ytr)
    Xt1 = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1)
    pred = Xt1 @ M; ss_res = ((Yte - pred)**2).sum(); ss_tot = ((Yte - Yte.mean(0))**2).sum()
    return float(1 - ss_res/ss_tot)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    coord, bags = content_coord(blocks)
    n = coord.shape[0]; ntr = int(0.7*n)
    Ytr, Yte = coord[:ntr], coord[ntr:]
    feats = {k: bags[k] for k in bags}
    combos = {'cur': ['cur'], 'cummean': ['cummean'], 'exp9': ['exp9'], 'exp99': ['exp99'],
              'full_bag': ['cur', 'cummean', 'exp9', 'exp99']}
    out = {'K': K, 'ref_layers': REF, 'n': n, 'r2': {}}
    for name, ks in combos.items():
        Xtr = torch.cat([feats[k][:ntr] for k in ks], 1); Xte = torch.cat([feats[k][ntr:] for k in ks], 1)
        out['r2'][name] = round(r2(Xtr, Ytr, Xte, Yte), 4)
    # shuffled-position null (full bag rows permuted)
    g = torch.Generator(device=DEV).manual_seed(0); perm = torch.randperm(ntr, generator=g, device=DEV)
    Xtr = torch.cat([feats[k][:ntr] for k in combos['full_bag']], 1)
    out['r2']['shuffled_null'] = round(r2(Xtr[perm], Ytr, torch.cat([feats[k][ntr:] for k in combos['full_bag']], 1), Yte), 4)
    out['bag_over_cur_ratio'] = round(out['r2']['full_bag']/max(out['r2']['cur'], 1e-6), 2)
    out['pred_a_content_is_bag'] = bool(out['r2']['full_bag'] > 2*out['r2']['cur'] and out['r2']['full_bag'] > 0.4)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content-from-bag R2: {out['r2']}", flush=True)
    print(f"full-bag/cur ratio {out['bag_over_cur_ratio']} | pred_a content-is-bag {out['pred_a_content_is_bag']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
