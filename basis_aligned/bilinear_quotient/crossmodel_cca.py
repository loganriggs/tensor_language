"""Do two INDEPENDENTLY-TRAINED bilinear models encode the SAME content information (not just similarly-structured
subspaces)? §1057 showed the same extremal contexts recur across bilin18/swiglu18; the rigorous, basis-independent test
is CCA: run bilin18 and swiglu18 on the SAME corpus, take each model's per-position content coordinates (projection of
its middle content deviation onto its own top-K content PCA), and compute the canonical correlations between the two
coordinate sets over the aligned positions. High canonical correlations => the two models carry the same content
information, aligned up to a linear map (their content manifolds are the same object in different bases). Control: CCA
against position-shuffled coordinates (destroys alignment) should give ~0.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-position CCA ~ 0 (no spurious alignment).
  (a) SHARED CONTENT INFORMATION: the top canonical correlations between bilin18 and swiglu18 content coordinates are
      HIGH (several > ~0.7) and far above the shuffled control -> the content manifold is universal in INFORMATION, not
      just in structure -- the same topic/register signal, in two independently-learned bases;
  (b) report the canonical-correlation spectrum (top values, mean of top-K) + shuffled control."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m as M18, DEV
from tier2_model import load_elriggs
import census_lib as cl

D18 = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'crossmodel_cca_results.json'
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
    P18 = content_coords(M18, REF, blocks, custom_fwd=fwd18)
    sg, _ = load_elriggs('swiglu18', device=DEV, dtype=torch.float32); sg.eval()
    Psg = content_coords(sg, REF, blocks, needs_target=True)
    del sg; torch.cuda.empty_cache()
    n = min(P18.shape[0], Psg.shape[0]); P18 = P18[:n]; Psg = Psg[:n]
    corrs = cca_corrs(P18, Psg)
    g = torch.Generator(device=DEV).manual_seed(0)
    perm = torch.randperm(n, generator=g, device=DEV)
    corrs_sh = cca_corrs(P18, Psg[perm])
    corrs_l = [round(float(c), 3) for c in corrs.tolist()]
    out = {'K': K, 'ref_layers': REF, 'n_positions': n,
           'canonical_correlations_top10': corrs_l[:10],
           'mean_topK': round(float(corrs.mean()), 3),
           'n_above_0.7': int((corrs > 0.7).sum()), 'n_above_0.5': int((corrs > 0.5).sum()),
           'shuffled_control_top5': [round(float(c), 3) for c in corrs_sh[:5].tolist()],
           'shuffled_mean': round(float(corrs_sh.mean()), 3)}
    out['pred_a_shared_content_info'] = bool(out['n_above_0.7'] >= 3 and out['mean_topK'] > 3*out['shuffled_mean'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top-10 canonical corrs (bilin18 vs swiglu18): {corrs_l[:10]}", flush=True)
    print(f"mean {out['mean_topK']} | >0.7: {out['n_above_0.7']}/{K} | >0.5: {out['n_above_0.5']}/{K} | shuffled mean {out['shuffled_mean']}", flush=True)
    print(f"pred_a shared content info: {out['pred_a_shared_content_info']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
