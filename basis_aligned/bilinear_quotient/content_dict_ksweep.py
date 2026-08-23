"""Registered in §1113: price HOW distributed the content manifold is in its own 64-dim coordinate space.
Top-k SAE (256 atoms) at k = 4, 8, 16, 32, 64: reconstruction R² and cross-seed stability (2 seeds per k) —
plus the PCA baseline (top-k' PCA of the coords, matched retained dimensionality per position is k) for
reference: does the learned dictionary beat a rank-k linear basis at matched per-position budget?

REGISTERED PREDICTIONS:
  (0) SANITY: R² monotone in k; k=64 R² ~1 (64-dim space, 64 active atoms).
  (a) DENSITY PRICED: R² reaches 0.9 only at k >= 24 (interpolating §1113's 0.71@8) -> the manifold needs
      ~1/3 of its coordinate dimensionality ACTIVE per position — "8-sparse features" was the wrong sparsity,
      and the honest description is 'moderately dense code' not 'sparse features';
  (b) DICTIONARY BEATS PCA at every k (learned overcomplete basis captures curvature a linear basis can't) —
      by >= 0.05 R² at k=8; if NOT, the manifold has no useful overcomplete structure at all and PCA axes are
      literally sufficient (strongest form of §1113's conclusion; report plainly);
  (c) stability stays >= 0.7 across k (reproducibility is not k-fragile)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_dict_ksweep_results.json'
NSEQ = 192; SEQ = 256; REF = [8, 10, 12]; K = 64; NATOM = 256; STEPS = 2500
H = m.transformer.h
KS = [4, 8, 16, 32, 64]


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return x


class TopKSAE(torch.nn.Module):
    def __init__(self, d, n, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.Parameter(torch.randn(n, d, generator=g)*0.1)
        self.Dm = torch.nn.Parameter(torch.randn(d, n, generator=g)*0.1)
        self.k = k
    def forward(self, x):
        a = x @ self.E.T
        top = a.topk(self.k, -1)
        code = torch.zeros_like(a).scatter_(-1, top.indices, top.values)
        return code @ self.Dm.T, code


def train_sae(Cc, k, seed):
    sae = TopKSAE(K, NATOM, k, seed).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    N = Cc.shape[0]
    with torch.enable_grad():
        for step in range(STEPS):
            idx = torch.randint(0, N, (4096,), device=DEV)
            x = Cc[idx]
            xh, _ = sae(x)
            loss = ((xh - x)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xh, _ = sae(Cc[:20000])
        r2 = 1 - float(((xh - Cc[:20000])**2).sum()/(Cc[:20000]**2).sum())
    return sae, r2


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); devsum = None
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    for L in REF:
        X = torch.cat(cap[L], 0); cap[L] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, S, Vt = torch.linalg.svd(dev, full_matrices=False)
    Uc = Vt[:K].T.contiguous()
    Cc = (dev @ Uc).contiguous(); del dev

    # PCA-of-coords baseline: rank-k projection R² inside the 64-dim coord space
    _, Sc, Vtc = torch.linalg.svd(Cc - Cc.mean(0), full_matrices=False)
    tot = float((Sc**2).sum())
    pca_r2 = {k: round(float((Sc[:k]**2).sum())/tot, 4) for k in KS}

    res = {}
    for k in KS:
        sae0, r2a = train_sae(Cc, k, 0)
        sae1, r2b = train_sae(Cc, k, 1)
        Da = F.normalize(sae0.Dm.T, dim=-1); Db = F.normalize(sae1.Dm.T, dim=-1)
        stab = float((Da @ Db.T).abs().max(1).values.mean())
        res[str(k)] = {'sae_r2': round((r2a+r2b)/2, 4), 'pca_r2': pca_r2[k], 'stability': round(stab, 3),
                       'sae_minus_pca': round((r2a+r2b)/2 - pca_r2[k], 4)}
        print(f"k={k}: SAE R2 {(r2a+r2b)/2:.4f} | PCA R2 {pca_r2[k]:.4f} | delta {(r2a+r2b)/2-pca_r2[k]:+.4f} | stability {stab:.3f}", flush=True)
    # k where SAE reaches 0.9 (linear interp)
    ks = KS; r2s = [res[str(k)]['sae_r2'] for k in ks]
    k90 = None
    for i in range(len(ks)-1):
        if r2s[i] < 0.9 <= r2s[i+1]:
            k90 = ks[i] + (0.9 - r2s[i])/(r2s[i+1]-r2s[i])*(ks[i+1]-ks[i]); break
    out = {'per_k': res, 'k_for_r2_090': round(k90, 1) if k90 else None}
    out['pred_a_density'] = bool(k90 is not None and k90 >= 24)
    out['pred_b_dict_beats_pca'] = bool(res['8']['sae_minus_pca'] >= 0.05)
    out['pred_c_stability_robust'] = bool(min(res[str(k)]['stability'] for k in KS) >= 0.7)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"k for R2=0.9: {out['k_for_r2_090']} | pred_a density {out['pred_a_density']} | pred_b dict>pca {out['pred_b_dict_beats_pca']} | pred_c stab {out['pred_c_stability_robust']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
