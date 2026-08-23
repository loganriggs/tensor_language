"""THREAD D (pooler criterion): the middle attention (attn3-14) 'broadly pools content' (§1047/§1054) -- but pools
BY WHAT? The squared attention multiplies two QK criteria; never decomposed for middle heads. Candidate criteria for
a key's weight: (1) CONTENT SIMILARITY between query and key residuals (projected on the content subspace U_c),
(2) key CONTENT NORM (how content-laden the key position is), (3) positional distance, (4) key residual norm
(massive activations), (5) key token frequency. Two tests at middle layers 6/8/10/12: (A) correlational -- sample
(query,key) pairs, correlate |pattern| with each feature + joint least-squares R^2; (B) CAUSAL -- mask (zero) the
top-16 content-similar keys per query (excluding the local window d<=4) vs 16 RANDOM eligible keys, per layer,
and compare CE cost -> if content-similar keys carry the pooling, masking them costs far more.

REGISTERED PREDICTIONS:
  (0) SANITY: random-key masking cost small; joint R^2 > any single feature.
  (a) CONTENT-DRIVEN POOLING: content-similarity (or key content norm) is the TOP correlate of middle |pattern|
      (|r| >= 2x the positional-distance |r|), and masking top-content-sim keys costs >= 3x masking random keys
      -> the middle gathers BY content relevance (a soft content-addressed lookup);
  (b) if position/norm dominate instead, the middle pool is content-blind aggregation (report plainly -- then the
      content specificity lives in the values, not the pattern)."""
import json, time, sys, types, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pooler_criterion_results.json'
NSEQ = 96; SEQ = 256; MID = [6, 8, 10, 12]; REF = [8, 10, 12]; K = 64
NMASK = 16; DMIN = 5
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
CAPX = {}; MASK = {'layer': -1, 'mask': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def capx_hook(L):
    def h(mo, args): CAPX[L] = args[0].detach()
    return h


def make_masked_sq(attn, L):
    orig = attn.squared_attention
    def patched(self, q, k, v, q2, k2):
        B, T, Hh, Dh = q.shape
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
        pat = (s1/Dh)*(s2/Dh)
        cm = torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool))
        pat.masked_fill_(cm.logical_not(), 0.0)
        if MASK['layer'] == L and MASK['mask'] is not None:
            pat = pat * MASK['mask'].unsqueeze(1).to(pat.dtype)   # B,1,T,T broadcast over heads
        return torch.einsum('bhqk,bkhd->bhqd', pat, v)   # [B,H,T,dh], same as original
    return orig, types.MethodType(patched, attn)


@torch.no_grad()
def pattern_for(attn, x):
    B, T, C = x.shape
    q = attn.c_q(x).view(B, T, NH, HD); k = attn.c_k(x).view(B, T, NH, HD)
    q2 = attn.c_q2(x).view(B, T, NH, HD); k2 = attn.c_k2(x).view(B, T, NH, HD)
    cos, sin = attn.rotary(q)
    q, k = F.rms_norm(q, (HD,)), F.rms_norm(k, (HD,))
    q, k = MOD.apply_rotary_emb(q, cos, sin), MOD.apply_rotary_emb(k, cos, sin)
    q2, k2 = F.rms_norm(q2, (HD,)), F.rms_norm(k2, (HD,))
    q2, k2 = MOD.apply_rotary_emb(q2, cos, sin), MOD.apply_rotary_emb(k2, cos, sin)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pat = (s1/HD)*(s2/HD)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return pat.masked_fill_(mask.logical_not(), 0.0)


@torch.no_grad()
def content_basis(blocks):
    """standard idiom: top-K PCA of pooled L8-12 mlp-input deviation."""
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for L in REF:
        X = torch.cat(cap[L], 0); xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; cap[L] = []; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    return Vt[:K].T.contiguous()


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    U_c = content_basis(blocks)
    tfreq = torch.zeros(int(m.lm_head.weight.shape[0]), device=DEV)
    ta = blocks.to(DEV).reshape(-1); tfreq.index_add_(0, ta, torch.ones_like(ta, dtype=torch.float))

    # ---- (A) correlational ----
    hcap = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in MID]
    feats = {L: [] for L in MID}; pats = {L: [] for L in MID}
    g = torch.Generator(device=DEV).manual_seed(0)
    for i in range(0, 32, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); fwd(idx); T = idx.shape[1]
        for L in MID:
            x = CAPX[L].float()                       # B,T,D (attn input, post-norm)
            pat = pattern_for(H[L].attn, CAPX[L]).abs().mean(1)   # B,T,T mean over heads
            c = x @ U_c                                # B,T,K content coords
            cn = c.norm(dim=-1)                        # content norm
            xn = x.norm(dim=-1)
            for b in range(x.shape[0]):
                qi = torch.randint(DMIN+1, T, (1500,), generator=g, device=DEV)
                kj = (torch.rand(1500, generator=g, device=DEV) * (qi - DMIN).float()).long()
                csim = F.cosine_similarity(c[b, qi], c[b, kj], dim=-1)
                fmat = torch.stack([csim, cn[b, kj], (qi-kj).float().log1p(), xn[b, kj],
                                    tfreq[idx[b, kj]].log1p()], 1)
                feats[L].append(fmat.cpu()); pats[L].append(pat[b, qi, kj].cpu())
    for h in hcap: h.remove()
    FEATNAMES = ['content_sim', 'key_content_norm', 'log_dist', 'key_norm', 'key_log_freq']
    corr = {}
    for L in MID:
        Fm = torch.cat(feats[L], 0); y = torch.cat(pats[L], 0)
        Fz = (Fm - Fm.mean(0)) / Fm.std(0).clamp_min(1e-6); yz = (y - y.mean()) / y.std().clamp_min(1e-6)
        r = (Fz * yz.unsqueeze(1)).mean(0)
        beta = torch.linalg.lstsq(Fz, yz.unsqueeze(1)).solution.squeeze(1)
        r2 = 1 - float(((Fz @ beta - yz)**2).mean())
        corr[str(L)] = {'r': {n: round(float(v), 3) for n, v in zip(FEATNAMES, r)}, 'joint_r2': round(r2, 3)}
        print(f"L{L} r: {corr[str(L)]['r']} | joint R2 {r2:.3f}", flush=True)

    # ---- (B) causal masking ----
    ce_blocks = blocks[:48]
    orig = {}; hcap2 = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in MID]
    for L in MID:
        o, p = make_masked_sq(H[L].attn, L); orig[L] = o; H[L].attn.squared_attention = p
    # pre-hook that BUILDS the mask each minibatch for the active layer/mode
    MODE = {'kind': None}
    def build_mask_hook(L):
        def h(mo, args):
            if MASK['layer'] != L or MODE['kind'] is None: return None
            x = args[0].float(); B, T, _ = x.shape
            c = x @ U_c
            cs = F.normalize(c, dim=-1) @ F.normalize(c, dim=-1).transpose(1, 2)  # B,T,T
            di = torch.arange(T, device=x.device).view(-1, 1) - torch.arange(T, device=x.device).view(1, -1)
            elig = (di >= DMIN)
            msk = torch.ones(B, T, T, device=x.device)
            if MODE['kind'] == 'content':
                cs2 = cs.masked_fill(~elig.unsqueeze(0), -2.0)
                topk = cs2.topk(NMASK, -1).indices                      # B,T,NMASK
                msk.scatter_(2, topk, 0.0)
                msk = msk * elig.unsqueeze(0) + (~elig).unsqueeze(0).float()  # only mask eligible
            else:  # random eligible keys
                rnd = torch.rand(B, T, T, device=x.device, generator=None).masked_fill(~elig.unsqueeze(0), -1.0)
                topk = rnd.topk(NMASK, -1).indices
                msk.scatter_(2, topk, 0.0)
                msk = msk * elig.unsqueeze(0) + (~elig).unsqueeze(0).float()
            MASK['mask'] = msk
            return None
        return h
    hmask = [H[L].attn.register_forward_pre_hook(build_mask_hook(L)) for L in MID]
    MASK['layer'] = -1; base = ce(ce_blocks)
    causal = {}
    for L in MID:
        row = {}
        for kind in ['content', 'random']:
            MODE['kind'] = kind; MASK['layer'] = L
            row[kind] = round(ce(ce_blocks) - base, 4)
            MASK['layer'] = -1; MASK['mask'] = None; MODE['kind'] = None
        row['ratio'] = round(row['content']/max(row['random'], 1e-4), 2)
        causal[str(L)] = row
        print(f"L{L} mask cost: content-sim {row['content']} | random {row['random']} | ratio {row['ratio']}", flush=True)
    for L in MID: H[L].attn.squared_attention = orig[L]
    for h in hcap2 + hmask: h.remove()

    top_feat = {L: max(corr[str(L)]['r'].items(), key=lambda kv: abs(kv[1])) for L in MID}
    content_top = sum(1 for L in MID if top_feat[L][0] in ('content_sim', 'key_content_norm')
                      and abs(top_feat[L][1]) >= 2*abs(corr[str(L)]['r']['log_dist']))
    causal_ok = sum(1 for L in MID if causal[str(L)]['ratio'] >= 3)
    out = {'base_ce': round(base, 4), 'corr': corr, 'causal_mask': causal,
           'top_feature': {str(L): top_feat[L] for L in MID},
           'pred_a_content_driven': bool(content_top >= 3 and causal_ok >= 3),
           'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content-top layers {content_top}/4 | causal>=3x layers {causal_ok}/4 | pred_a {out['pred_a_content_driven']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
