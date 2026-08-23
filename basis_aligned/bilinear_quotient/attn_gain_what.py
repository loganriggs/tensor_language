"""Registered in §1108: the GATHERER band's (L3-5) un-kerneled pooling structure is substantially per-query
MASS (oracle gain lifts 0.39->0.64). What does the gain TRACK? Per (layer, head, query-position): oracle
alpha_h(q) = real |pattern| row mass / kernel row mass. Regress alpha on query features: (1) log position
(mass grows with available context?), (2) query token log-frequency, (3) query CONTENT norm ||U_c^T x_q||
(content-hungry positions pool more?), (4) query class one-hots (function vs content word), (5) local
inductability (query token seen before in context). Correlational + joint R^2, per layer; plus the same for
the middle band as contrast. NSEQ=96 (pattern-heavy).

REGISTERED PREDICTIONS:
  (0) SANITY: alpha variance is substantial in the gatherer (else §1108's oracle gain would have been small).
  (a) CONTENT-DEMAND GAIN: query content-norm and/or content-word class are the top correlates of gatherer
      alpha (|r| >= 2x log-position's) -> the gatherer pools MORE for content-hungry queries — the mass channel
      is the content machine's demand signal (ties §1108 to §1030's induction-state mass amplification);
  (b) if log-position or frequency dominates instead, the gain is a generic ramp/frequency effect (report
      plainly — then the mass channel is not content-demand)."""
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_gain_what_results.json'
NSEQ = 96; SEQ = 256; REF = [8, 10, 12]; K = 64
BANDS = {'gatherer_L3_5': [3, 4, 5], 'middle_L6_14': [6, 8, 10, 12, 14]}
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
CAPX = {}
enc = tiktoken.get_encoding('gpt2')


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def capx_hook(L):
    def h(mo, args): CAPX[L] = args[0].detach()
    return h


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
    cap = {Lr: [] for Lr in REF}; hs = []
    for Lr in REF:
        def mk(Lr):
            def h(mo, i_, o_): cap[Lr].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[Lr].mlp.register_forward_hook(mk(Lr)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for Lr in REF:
        X = torch.cat(cap[Lr], 0); cap[Lr] = []
        xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    return Vt[:K].T.contiguous()


def is_function_word(tid):
    raw = enc.decode([tid]); s = raw.strip().lower()
    return s in {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'and', 'or', 'but', 'is',
                 'are', 'was', 'were', 'be', 'it', 'he', 'she', 'they', 'that', 'this', 'as', 'not', 'from'}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    Uc = content_basis(blocks)
    tfreq = torch.zeros(V, device=DEV)
    ta = blocks.to(DEV).reshape(-1); tfreq.index_add_(0, ta, torch.ones_like(ta, dtype=torch.float))
    func_mask = torch.zeros(V, dtype=torch.bool)
    for t in range(min(V, 50257)):
        try:
            if is_function_word(t): func_mask[t] = True
        except Exception:
            pass
    func_mask = func_mask.to(DEV)

    ALL_L = sorted({L for ls in BANDS.values() for L in ls})
    hcap = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in ALL_L]
    T = SEQ - 1
    di = torch.arange(T, device=DEV).view(-1, 1) - torch.arange(T, device=DEV).view(1, -1)

    # kernels from first 32 seqs
    ksum = {L: torch.zeros(NH, T, device=DEV) for L in ALL_L}; kcnt = torch.zeros(T, device=DEV)
    for i in range(0, 32, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx)
        for L in ALL_L:
            pat = pattern_for(H[L].attn, CAPX[L])
            for dd in range(T):
                mask = (di == dd); nel = int(mask.sum())
                if nel == 0: continue
                ksum[L][:, dd] += pat[:, :, mask].sum((0, 2))
                if L == ALL_L[0]: kcnt[dd] += nel*pat.shape[0]
            del pat
    kern = {L: ksum[L]/kcnt.clamp_min(1) for L in ALL_L}   # NH,T mean pattern by distance

    # alpha + features on next 64 seqs
    feats = {L: [] for L in ALL_L}; alphas = {L: [] for L in ALL_L}
    for i in range(32, 96, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); fwd(idx)
        seen_before = torch.zeros_like(idx, dtype=torch.float)
        for b in range(idx.shape[0]):
            _, inv, cnts = torch.unique(idx[b], return_inverse=True, return_counts=True)
            # seen-before flag: occurrence index > 0
            first = torch.zeros_like(idx[b], dtype=torch.bool)
            seen = {}
            arr = idx[b].tolist()
            fl = [0.0]*len(arr)
            s2 = set()
            for j2, t2 in enumerate(arr):
                fl[j2] = 1.0 if t2 in s2 else 0.0
                s2.add(t2)
            seen_before[b] = torch.tensor(fl, device=DEV)
        for L in ALL_L:
            x = CAPX[L].float()
            pat = pattern_for(H[L].attn, CAPX[L]).abs().sum(-1)     # B,NH,T real row mass
            # kernel row mass at position q: sum_d<=q kern(d)
            kmass = torch.cumsum(kern[L].abs(), dim=1)               # NH,T (approx row mass by position)
            alpha = pat / kmass.unsqueeze(0).clamp_min(1e-6)         # B,NH,T
            cnorm = (x @ Uc).norm(dim=-1)                            # B,T
            pos = torch.arange(T, device=DEV).float().log1p().expand(idx.shape[0], T)
            lfreq = tfreq[idx].log1p()
            fw = func_mask[idx].float()
            F5 = torch.stack([pos, lfreq, cnorm, fw, seen_before], -1)   # B,T,5
            feats[L].append(F5.reshape(-1, 5).cpu())
            alphas[L].append(alpha.mean(1).reshape(-1).cpu())        # head-mean alpha per position
            del pat
    for h in hcap: h.remove()

    NAMES = ['log_pos', 'log_freq', 'content_norm', 'function_word', 'seen_before']
    res = {}
    for bname, ls in BANDS.items():
        Fm = torch.cat([torch.cat(feats[L], 0) for L in ls], 0)
        y = torch.cat([torch.cat(alphas[L], 0) for L in ls], 0)
        ok = torch.isfinite(y)
        Fm, y = Fm[ok], y[ok]
        Fz = (Fm - Fm.mean(0))/Fm.std(0).clamp_min(1e-6); yz = (y - y.mean())/y.std().clamp_min(1e-6)
        r = (Fz*yz.unsqueeze(1)).mean(0)
        beta = torch.linalg.lstsq(Fz, yz.unsqueeze(1)).solution.squeeze(1)
        r2 = 1 - float(((Fz@beta - yz)**2).mean())
        res[bname] = {'r': {n: round(float(v), 3) for n, v in zip(NAMES, r)}, 'joint_r2': round(r2, 3),
                      'alpha_cv': round(float(y.std()/y.mean().clamp_min(1e-6)), 3)}
        print(f"{bname}: r {res[bname]['r']} | R2 {r2:.3f} | alpha CV {res[bname]['alpha_cv']}", flush=True)
    g = res['gatherer_L3_5']['r']
    top = max(g.items(), key=lambda kv: abs(kv[1]))
    out = {'bands': res, 'gatherer_top_feature': top,
           'pred_a_content_demand': bool(top[0] in ('content_norm', 'function_word')
                                         and abs(top[1]) >= 2*abs(g['log_pos'])),
           'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"gatherer top feature: {top} | pred_a content-demand {out['pred_a_content_demand']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
