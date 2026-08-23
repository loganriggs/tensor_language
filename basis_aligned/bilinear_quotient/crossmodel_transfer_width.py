"""Cross-WIDTH CAUSAL transfer (causal companion to §1063). §1062 showed same-width content transfer (swiglu18->bilin18,
96% of within-model); §1063 showed bilin12 (D=768) shares content INFO with bilin18 (CCA cross-width 0.96). Does bilin12's
content CAUSALLY drive bilin18 despite different width AND depth? Extract bilin12's pooled-middle content coordinate on a
source text, map it into bilin18's content space via a fitted linear W (64x64, coordinate space is width-independent),
reconstruct, and patch it into bilin18 running on a target text (all deep-middle layers). If bilin18's output moves
toward bilin18-on-source -- a good fraction of the within-model upper bound, far above a random-map control -- the
content is causally interchangeable ACROSS ARCHITECTURES.

NOTE vs §1062: uses a SINGLE pooled per-position content coordinate injected at all deep-middle layers (to bridge the
12- vs 18-layer depth mismatch), so absolute alignments are lower than §1062's per-layer 0.96; the within-model control
uses the SAME pooled method, so the cross/within RATIO is the fair quantity.

REGISTERED PREDICTIONS:
  (0) SANITY: random-map control ~ low; within-model is the upper bound.
  (a) CROSS-WIDTH CAUSAL TRANSFER: bilin12->bilin18 mapped-content patching moves bilin18-target toward bilin18-source
      at a meaningful fraction (>~0.5) of the within-model upper bound and clearly above the random-map control -> the
      content is causally interchangeable across width/depth;
  (b) report alignment for within-model / cross-width-mapped / random-map."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m as M18, DEV
from tier2_model import load_elriggs
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'crossmodel_transfer_width_results.json'
NEVAL = 240; SEQ = 256; ABL = list(range(6, 15)); K = 64
ST = {'patch': False, 'inj': None}
CAP = {}


def fwd18(idx, patch=False):
    x = F.rms_norm(M18.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(M18.transformer.h):
        x, v1 = blk(x, v1, x0)
        if patch and li in ABL:
            x = x - (x @ U18) @ U18.T + ST['inj']
    return 30.0*torch.tanh(M18.lm_head(F.rms_norm(x, (D,)))/30.0)


def _block_out(o_):
    return (o_[0] if isinstance(o_, tuple) else o_).detach().float()


@torch.no_grad()
def build_basis(model, ref, blocks, is18):
    """content basis (D,K) from pooled BLOCK-OUTPUT residual (matches the removal point in fwd18), mean-centered."""
    Dm = model.transformer.wte.weight.shape[1]
    for L in ref: CAP[L] = []
    hs = [model.transformer.h[L].register_forward_hook((lambda L: (lambda mo, i_, o_: CAP[L].append(_block_out(o_).reshape(-1, Dm))))(L)) for L in ref]
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous()
        if is18: fwd18(idx)
        else: model(idx, idx)
    for h in hs: h.remove()
    Xavg = sum(torch.cat(CAP[L], 0) for L in ref) / len(ref)   # pooled block-output residual
    for L in ref: CAP[L] = []
    devc = Xavg - Xavg.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    return Vt[:K].T.contiguous()


@torch.no_grad()
def coord(model, ref, U, idx, is18):
    """pooled BLOCK-OUTPUT residual projected onto U -> (B,T,K); raw (matches fwd18 removal scale)."""
    Dm = U.shape[0]
    for L in ref: CAP[L] = []
    hs = [model.transformer.h[L].register_forward_hook((lambda L: (lambda mo, i_, o_: CAP[L].append(_block_out(o_))))(L)) for L in ref]
    if is18: fwd18(idx)
    else: model(idx, idx)
    for h in hs: h.remove()
    B, T = idx.shape
    Xavg = sum(CAP[L][0] for L in ref) / len(ref)   # (B,T,Dm)
    for L in ref: CAP[L] = []
    return (Xavg.reshape(-1, Dm) @ U).reshape(B, T, K)


@torch.no_grad()
def main():
    global U18
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(M18.lm_head.weight.shape[0])
    b12, _ = load_elriggs('bilin12', device=DEV, dtype=torch.float32); b12.eval()
    U18 = build_basis(M18, [8, 10, 12], blocks, is18=True)
    U12 = build_basis(b12, [5, 6, 7], blocks, is18=False)
    # fit W (K,K): c18 ~= c12 @ W on fit half (pooled coords, flattened)
    nfit = blocks.shape[0] // 2; fitb = blocks[:nfit].contiguous()
    A = []; B = []
    for i in range(0, nfit, 8):
        idx = fitb[i:i+8].to(DEV)[:, :-1].contiguous()
        A.append(coord(b12, [5, 6, 7], U12, idx, False).reshape(-1, K))
        B.append(coord(M18, [8, 10, 12], U18, idx, True).reshape(-1, K))
    A = torch.cat(A, 0); B = torch.cat(B, 0)
    W = torch.linalg.solve(A.T @ A + 1e-2*torch.eye(K, device=DEV), A.T @ B)
    g = torch.Generator(device=DEV).manual_seed(0); Wr = W[torch.randperm(K, generator=g, device=DEV)]
    # eval on held-out half: source/target pairs
    ev = blocks[nfit:].contiguous(); n = ev.shape[0] // 2; S = ev[:n].contiguous(); T = ev[n:2*n].contiguous()
    al = {'within': 0.0, 'cross': 0.0, 'randmap': 0.0}; npos = 0
    for i in range(0, n, 8):
        si = S[i:i+8].to(DEV)[:, :-1].contiguous(); ti = T[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        lb = fwd18(ti, patch=False).float()
        ls = fwd18(si, patch=False).float()
        dsrc = (ls - lb).reshape(-1, V)
        c18_S = coord(M18, [8, 10, 12], U18, si, True)      # (B,T,K)
        c12_S = coord(b12, [5, 6, 7], U12, si, False)
        for tag, cc in (('within', c18_S), ('cross', c12_S @ W), ('randmap', c12_S @ Wr)):
            ST['inj'] = (cc @ U18.T)                                # (B,T,D)
            lp = fwd18(ti, patch=True).float()
            cos = F.cosine_similarity((lp - lb).reshape(-1, V), dsrc, dim=-1)
            al[tag] += float(cos.sum())
        npos += si.shape[0] * si.shape[1]
    out = {'K': K, 'abl_range': [ABL[0], ABL[-1]], 'n_positions': npos, 'method': 'pooled-coord single vector all deep-middle layers',
           'alignment_within_model': round(al['within']/npos, 4),
           'alignment_crosswidth_mapped': round(al['cross']/npos, 4),
           'alignment_random_map': round(al['randmap']/npos, 4)}
    up = max(out['alignment_within_model'], 1e-6)
    out['cross_frac_of_within'] = round(out['alignment_crosswidth_mapped']/up, 3)
    out['pred_a_crosswidth_transfer'] = bool(out['alignment_crosswidth_mapped'] > 0.5*up and
                                             out['alignment_crosswidth_mapped'] > 2*out['alignment_random_map'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"within {out['alignment_within_model']} | cross-width {out['alignment_crosswidth_mapped']} | random-map {out['alignment_random_map']}", flush=True)
    print(f"cross frac of within {out['cross_frac_of_within']} | pred_a {out['pred_a_crosswidth_transfer']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
