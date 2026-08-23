"""Is the content FUNCTIONALLY interchangeable across independently-trained models? §1061 (CCA) showed bilin18 and
swiglu18 encode the same content up to a linear map W. §1059/§1060 showed patching bilin18's OWN source content into a
bilin18 target transports topic. Combine them: extract content from SWIGLU18 running on a source text, map it into
bilin18's content space via W, and patch it into BILIN18 running on a target text. If bilin18's output then moves toward
what bilin18-on-source would predict -- a good fraction of the within-model (bilin18->bilin18) upper bound, and far
above a random-map control -- then the content is not just correlated across models but CAUSALLY interchangeable: one
model's context representation drives another's computation.

REGISTERED PREDICTIONS:
  (0) SANITY: random-map control ~ 0 alignment; within-model patch (bilin18 own source content) is the strong upper bound.
  (a) CROSS-MODEL TRANSFER WORKS: swiglu18->bilin18 mapped-content patching moves bilin18-target toward bilin18-source
      (alignment clearly POSITIVE, >> random-map control, a meaningful fraction of the within-model upper bound) -> the
      content representation is functionally interchangeable across independently-trained models;
  (b) report alignment for within-model / cross-model-mapped / random-map patches."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m as M18, DEV
from tier2_model import load_elriggs
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'crossmodel_transfer_results.json'
NEVAL = 200; SEQ = 256; REF = [8, 10, 12]; ABL = list(range(6, 15)); K = 64
ST = {'mode': None, 'inj': None}   # mode None|'cap'(store block residuals)|'patch'(inject content); inj: {L:(N?,D) per-batch}
STORE = {}


def fwd18(idx, patch=False):
    x = F.rms_norm(M18.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(M18.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            STORE[li] = x.detach()
        elif patch and li in ABL:
            inj = ST['inj'][li]                       # (B,T,D) content vector to inject
            x = x - (x @ U18) @ U18.T + inj
    return 30.0*torch.tanh(M18.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def content_basis(model, ref, blocks, is18):
    """top-K content PCA basis (D,K) from pooled deep-middle content deviation."""
    D_ = model.transformer.wte.weight.shape[1]; V = model.transformer.wte.weight.shape[0]
    cap = {L: [] for L in ref}; hs = []
    for L in ref:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D_))
            return h
        hs.append(model.transformer.h[L].mlp.register_forward_hook(mk(L)))
    toks = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1))
        if is18: ST['mode'] = None; fwd18(idx)
        else: model(idx, idx)
    for h in hs: h.remove()
    tok = torch.cat(toks, 0); devsum = None
    for L in ref:
        X = torch.cat(cap[L], 0); xbar = torch.zeros(V, D_, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(ref); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    return Vt[:K].T.contiguous()


@torch.no_grad()
def sg_block_res(sg, idx):
    """swiglu18 per-deep-middle-layer residual after block (B,T,D), via block-output hooks."""
    st = {}; hs = []
    for L in ABL:
        def mk(L):
            def h(mo, i_, o_): st[L] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
            return h
        hs.append(sg.transformer.h[L].register_forward_hook(mk(L)))
    sg(idx, idx)
    for h in hs: h.remove()
    return st


@torch.no_grad()
def fit_W(sg, blocks):
    """ridge map W (K,K): bilin18 layer-residual proj (via U18) ~= swiglu18 proj (via Usg) @ W, pooled over deep-middle."""
    A = []; B = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous()
        ST['mode'] = 'cap'; STORE.clear(); fwd18(idx); ST['mode'] = None
        res18 = {L: STORE[L] for L in ABL}
        ressg = sg_block_res(sg, idx)
        for L in ABL:
            A.append((ressg[L].reshape(-1, D) @ Usg))   # (N,K) swiglu proj
            B.append((res18[L].reshape(-1, D) @ U18))    # (N,K) bilin18 proj
    A = torch.cat(A, 0); B = torch.cat(B, 0)
    W = torch.linalg.solve(A.T @ A + 1e-2*torch.eye(K, device=DEV), A.T @ B)  # (K,K)
    return W


@torch.no_grad()
def main():
    global U18, Usg
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    sg, _ = load_elriggs('swiglu18', device=DEV, dtype=torch.float32); sg.eval()
    U18 = content_basis(M18, REF, blocks, is18=True)
    Usg = content_basis(sg, REF, blocks, is18=False)
    nfit = blocks.shape[0] // 2; fitb = blocks[:nfit].contiguous()
    W = fit_W(sg, fitb)
    g = torch.Generator(device=DEV).manual_seed(0); Wr = W[torch.randperm(K, generator=g, device=DEV)]  # random-map control (row-permuted)
    # eval: source/target pairs from the held-out half
    ev = blocks[nfit:].contiguous(); n = ev.shape[0] // 2; S = ev[:n].contiguous(); T = ev[n:2*n].contiguous()
    V = int(M18.lm_head.weight.shape[0])
    al = {'within': 0.0, 'cross': 0.0, 'randmap': 0.0}; npos = 0
    for i in range(0, n, 8):
        si = S[i:i+8].to(DEV)[:, :-1].contiguous(); ti = T[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        # bilin18 source residuals (for within-model) and logits; bilin18 target baseline
        ST['mode'] = 'cap'; STORE.clear(); ls = fwd18(si).float(); ST['mode'] = None
        res18_S = {L: STORE[L] for L in ABL}
        lb = fwd18(ti).float()
        dsrc = (ls - lb).reshape(-1, V)
        # swiglu18 source residuals -> content coords
        ressg_S = sg_block_res(sg, si)
        for tag, Wmat in (('within', None), ('cross', W), ('randmap', Wr)):
            inj = {}
            for L in ABL:
                if tag == 'within':
                    inj[L] = (res18_S[L].reshape(-1, D) @ U18 @ U18.T).reshape(res18_S[L].shape)
                else:
                    c = ressg_S[L].reshape(-1, D) @ Usg          # (N,K) swiglu content coords
                    inj[L] = ((c @ Wmat) @ U18.T).reshape(res18_S[L].shape)
            ST['inj'] = inj
            lp = fwd18(ti, patch=True).float()
            cos = F.cosine_similarity((lp - lb).reshape(-1, V), dsrc, dim=-1)
            al[tag] += float(cos.sum())
        npos += si.shape[0] * si.shape[1]
    out = {'K': K, 'abl_range': [ABL[0], ABL[-1]], 'n_positions': npos,
           'alignment_within_model': round(al['within']/npos, 4),
           'alignment_crossmodel_mapped': round(al['cross']/npos, 4),
           'alignment_random_map': round(al['randmap']/npos, 4)}
    upper = max(out['alignment_within_model'], 1e-6)
    out['cross_frac_of_within'] = round(out['alignment_crossmodel_mapped']/upper, 3)
    out['pred_a_transfer_works'] = bool(out['alignment_crossmodel_mapped'] > 0.05 and
                                        out['alignment_crossmodel_mapped'] > 2*out['alignment_random_map'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"alignment  within {out['alignment_within_model']} | cross-mapped {out['alignment_crossmodel_mapped']} | random-map {out['alignment_random_map']}", flush=True)
    print(f"cross as frac of within {out['cross_frac_of_within']} | pred_a {out['pred_a_transfer_works']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
