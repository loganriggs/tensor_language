"""CROSS-WIDTH extension of §1061. §1061 showed bilin18 and swiglu18 (both D=1152) encode the SAME content information
(CCA 0.95-0.97). Does this survive DIFFERENT WIDTH and DEPTH? Add bilin12 (D=768, 12 layers) and compute all three
pairwise CCAs between per-position content coordinates on the same corpus: bilin18<->swiglu18 (replicate §1061),
bilin18<->bilin12 (cross-width), swiglu18<->bilin12 (cross-width). CCA operates in the K=64 content-coordinate space
regardless of D, so width differences don't block it. High cross-width CCA => the content information is width/depth-
independent -- a genuinely architecture-independent representation.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-position CCA ~ 0 for every pair.
  (a) CROSS-WIDTH UNIVERSAL: the cross-width pairs (bilin12 vs the D=1152 models) also show HIGH top canonical
      correlations (several > 0.7), comparable to the same-width §1061 pair -> the content representation is
      width/depth-independent, not an artifact of matched dimensionality;
  (b) report the canonical-correlation spectrum for all three pairs + shuffled controls."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m as M18, DEV
from tier2_model import load_elriggs
import census_lib as cl

D18 = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'crossmodel_cca_width_results.json'
NEVAL = 200; SEQ = 256; REF = [8, 10, 12]; K = 64
CAP = {}


def fwd18(idx):
    x = F.rms_norm(M18.transformer.wte(idx), (D18,)); x0 = x; v1 = None
    for blk in M18.transformer.h: x, v1 = blk(x, v1, x0)
    return x


@torch.no_grad()
def content_coords(model, ref, blocks, custom_fwd=None, needs_target=False):
    """Return (N,K) per-position content coordinates on the shared corpus, aligned by position."""
    D = model.transformer.wte.weight.shape[1]; V = model.transformer.wte.weight.shape[0]
    for L in ref: CAP[L] = []
    hs = []
    for L in ref:
        mlp = model.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    toks = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1))
        if custom_fwd is not None: custom_fwd(idx)
        elif needs_target: model(idx, idx)
        else: model(idx)
    for h in hs: h.remove()
    tok = torch.cat(toks, 0)
    devsum = None
    for L in ref:
        X = torch.cat(CAP[L], 0); xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X, dv
        CAP[L] = []
    dev = devsum / len(ref); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    return (devc @ Vt[:K].T).contiguous()   # (N,K) content coordinates


def cca_corrs(A, B, ridge=1e-3):
    A = A - A.mean(0); B = B - B.mean(0); N = A.shape[0]
    Cxx = A.T @ A / N + ridge*torch.eye(A.shape[1], device=DEV)
    Cyy = B.T @ B / N + ridge*torch.eye(B.shape[1], device=DEV)
    Cxy = A.T @ B / N
    def invsqrt(C):
        w, Q = torch.linalg.eigh(C); return Q @ torch.diag(w.clamp_min(1e-8).rsqrt()) @ Q.T
    Mmat = invsqrt(Cxx) @ Cxy @ invsqrt(Cyy)
    return torch.linalg.svdvals(Mmat).clamp(0, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    # per-position content coordinates for each model (K=64), aligned by position
    P = {}
    P['bilin18'] = content_coords(M18, [8, 10, 12], blocks, custom_fwd=fwd18)
    sg, _ = load_elriggs('swiglu18', device=DEV, dtype=torch.float32); sg.eval()
    P['swiglu18'] = content_coords(sg, [8, 10, 12], blocks, needs_target=True); del sg; torch.cuda.empty_cache()
    b12, _ = load_elriggs('bilin12', device=DEV, dtype=torch.float32); b12.eval()
    P['bilin12'] = content_coords(b12, [5, 6, 7], blocks, needs_target=True); del b12; torch.cuda.empty_cache()
    n = min(v.shape[0] for v in P.values())
    for k in P: P[k] = P[k][:n]
    g = torch.Generator(device=DEV).manual_seed(0); perm = torch.randperm(n, generator=g, device=DEV)
    pairs = [('bilin18', 'swiglu18', 'same-width'), ('bilin18', 'bilin12', 'cross-width'), ('swiglu18', 'bilin12', 'cross-width')]
    out = {'K': K, 'n_positions': n, 'D': {'bilin18': 1152, 'swiglu18': 1152, 'bilin12': 768}, 'pairs': {}}
    for a, b, kind in pairs:
        c = cca_corrs(P[a], P[b]); csh = cca_corrs(P[a], P[b][perm]); cl_ = [round(float(x), 3) for x in c.tolist()]
        out['pairs'][f'{a}__{b}'] = {'kind': kind, 'top10': cl_[:10], 'mean': round(float(c.mean()), 3),
                                     'n_above_0.7': int((c > 0.7).sum()), 'shuffled_mean': round(float(csh.mean()), 3)}
        print(f"{a} vs {b} ({kind}): top5 {cl_[:5]} | mean {out['pairs'][f'{a}__{b}']['mean']} | >0.7 {int((c>0.7).sum())}/{K} | shuf {out['pairs'][f'{a}__{b}']['shuffled_mean']}", flush=True)
    cw = [v for k, v in out['pairs'].items() if v['kind'] == 'cross-width']
    out['pred_a_crosswidth_universal'] = bool(all(v['n_above_0.7'] >= 3 and v['mean'] > 3*v['shuffled_mean'] for v in cw))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a cross-width universal: {out['pred_a_crosswidth_universal']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
