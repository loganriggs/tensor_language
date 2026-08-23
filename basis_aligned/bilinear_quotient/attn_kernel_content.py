"""Registered next rung from §1099 (dossier attn-middle-pooling): the static distance kernel recovers 0.583 of
the middle band's collective dynamic value (gatherer 0.394). §1085 says the remaining pattern structure is a
content-similarity bias (second-order, depth-growing). TWO-TERM pattern model: pattern_hat(q,k) = k_h(d) *
(1 + beta_h * csim(q,k)), where csim = cosine of content coordinates (U_c) at the attention input and beta_h is
fit per head by least squares on real patterns (weights+activations, no CE fitting). Values stay dynamic.
Bands: middle L6-14, gatherer L3-5. Also a POSITION-ONLY refit control (beta=0, same pipeline) to confirm the
kernel-only baseline reproduces §1099.

REGISTERED PREDICTIONS:
  (0) SANITY: beta=0 reproduces §1099's recoveries (~0.58/~0.39); fitted beta_h > 0 for most middle heads
      (§1085's positive content-sim correlation).
  (a) CONTENT TERM LIFTS THE MIDDLE: kernel+content recovers >= 0.75 of the middle band's collective value
      (vs 0.583 kernel-only) -> the middle pool is UNDERSTOOD as 'recency kernel x content-similarity gate'
      — the band converts to a named two-term stand-in on the benchmark;
  (b) GATHERER GAP CLOSES LESS: the gatherer improves but stays < 0.6 (its routing uses structure beyond
      distance+content-sim — e.g. induction/syntax);
  (c) if the content term adds < 0.05, the §1085 correlational content bias is causally inert in the pattern
      (the collective value lives in residual pattern structure we haven't named — report plainly)."""
import json, time, sys, types, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_kernel_content_results.json'
NSEQ = 192; SEQ = 256; REF = [8, 10, 12]; K = 64
BANDS = {'gatherer_L3_5': [3, 4, 5], 'middle_L6_14': list(range(6, 15))}
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
CTL = {'mode': None, 'band': None}
MEANS = {}; KERNELS = {}; BETAS = {}; CSIM = {}
CAPX = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def const_hook(L):
    def h(mo, args):
        if CTL['mode'] != 'const' or L not in CTL['band']: return None
        y = args[0].clone()
        for hh in range(NH):
            y[..., hh*HD:(hh+1)*HD] = MEANS[L][hh].view(1, 1, HD).to(y.dtype)
        return (y,) + tuple(args[1:])
    return h


def capx_hook(L):
    def h(mo, args): CAPX[L] = args[0].detach()
    return h


def make_sq(attn, L):
    orig = attn.squared_attention
    def patched(self, q, k, v, q2, k2):
        B, T, Hh, Dh = q.shape
        if CTL['mode'] in ('kernel', 'kernel_content') and L in CTL['band']:
            pat = KERNELS[L][:, :T, :T].unsqueeze(0).expand(B, NH, T, T).clone().to(v.dtype)
            if CTL['mode'] == 'kernel_content':
                cs = CSIM[L]                              # B,T,T from pre-hook
                pat = pat * (1.0 + BETAS[L].view(1, NH, 1, 1).to(pat.dtype) * cs.unsqueeze(1).to(pat.dtype))
        else:
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
            pat = (s1/Dh)*(s2/Dh)
            cm = torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool))
            pat = pat.masked_fill(cm.logical_not(), 0.0)
        return torch.einsum('bhqk,bkhd->bhqd', pat, v)
    return orig, types.MethodType(patched, attn)


def csim_hook(L):
    def h(mo, args):
        if CTL['mode'] != 'kernel_content' or L not in CTL['band']: return None
        x = args[0].float()
        c = F.normalize(x @ CTL['Uc'], dim=-1)
        CSIM[L] = c @ c.transpose(1, 2)
        return None
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
    for i in range(0, 96, 8):
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
    CTL['Uc'] = content_basis(blocks)
    ALL_L = sorted({L for ls in BANDS.values() for L in ls})
    T = SEQ - 1

    # fit kernels + betas from data (weights+activations; least squares per head)
    hcapx = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in ALL_L]
    caps = {L: torch.zeros(NH, HD, device=DEV) for L in ALL_L}
    hmean = []
    for L in ALL_L:
        def mk(L):
            def h(mo, args): caps[L] += args[0].detach().float().reshape(-1, NH, HD).sum(0)
            return h
        hmean.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    ksum = {L: torch.zeros(NH, T, device=DEV) for L in ALL_L}
    kcnt = torch.zeros(T, device=DEV)
    num = {L: torch.zeros(NH, device=DEV) for L in ALL_L}     # for beta: sum k(d)*csim*resid terms
    den = {L: torch.zeros(NH, device=DEV) for L in ALL_L}
    di = torch.arange(T, device=DEV).view(-1, 1) - torch.arange(T, device=DEV).view(1, -1)
    npos = 0
    # pass A: kernels
    for i in range(0, 64, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx); npos += idx.numel()
        for L in ALL_L:
            pat = pattern_for(H[L].attn, CAPX[L])
            for dd in range(T):
                mask = (di == dd); nel = int(mask.sum())
                if nel == 0: continue
                ksum[L][:, dd] += pat[:, :, mask].sum((0, 2))
                if L == ALL_L[0]: kcnt[dd] += nel * pat.shape[0]
            del pat
    for L in ALL_L:
        kern = ksum[L] / kcnt.clamp_min(1)
        km = torch.zeros(NH, T, T, device=DEV)
        for dd in range(T):
            iidx = torch.arange(dd, T, device=DEV)
            km[:, iidx, iidx-dd] = kern[:, dd].unsqueeze(1)
        KERNELS[L] = km
    # pass B: betas — regress (pat - k(d)) on k(d)*csim per head
    for i in range(0, 64, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx)
        for L in ALL_L:
            x = CAPX[L].float()
            c = F.normalize(x @ CTL['Uc'], dim=-1)
            cs = c @ c.transpose(1, 2)                     # B,T,T
            pat = pattern_for(H[L].attn, CAPX[L])          # B,NH,T,T
            kd = KERNELS[L].unsqueeze(0)                   # 1,NH,T,T
            resid = pat - kd
            feat = kd * cs.unsqueeze(1)
            num[L] += (resid * feat).sum((0, 2, 3))
            den[L] += (feat * feat).sum((0, 2, 3))
            del pat, resid, feat, cs
    for L in ALL_L:
        BETAS[L] = (num[L] / den[L].clamp_min(1e-9))
        MEANS[L] = caps[L] / npos
    for h in hmean + hcapx: h.remove()
    beta_pos_frac = float(torch.cat([BETAS[L] for L in BANDS['middle_L6_14']]).gt(0).float().mean())
    print(f"middle beta>0 fraction: {beta_pos_frac:.2f} | example betas L8: {[round(float(b),3) for b in BETAS[8]]}", flush=True)

    origs = {}
    for L in ALL_L:
        o, p = make_sq(H[L].attn, L); origs[L] = o; H[L].attn.squared_attention = p
    hconst = [H[L].attn.c_proj.register_forward_pre_hook(const_hook(L)) for L in ALL_L]
    hcs = [H[L].attn.register_forward_pre_hook(csim_hook(L)) for L in ALL_L]
    CTL['mode'] = None; base = ce(blocks)
    out = {'base_ce': round(base, 4), 'middle_beta_pos_frac': round(beta_pos_frac, 3), 'bands': {}}
    for bname, ls in BANDS.items():
        row = {}
        for mode in ['const', 'kernel', 'kernel_content']:
            CTL['mode'] = mode; CTL['band'] = set(ls)
            row[mode] = round(ce(blocks) - base, 4)
            CTL['mode'] = None; CTL['band'] = None
        row['kernel_recovery'] = round(1 - row['kernel']/max(row['const'], 1e-6), 3)
        row['kernel_content_recovery'] = round(1 - row['kernel_content']/max(row['const'], 1e-6), 3)
        row['content_term_gain'] = round(row['kernel_content_recovery'] - row['kernel_recovery'], 3)
        out['bands'][bname] = row
        print(f"{bname}: const +{row['const']} | kernel +{row['kernel']} ({row['kernel_recovery']}) | +content +{row['kernel_content']} ({row['kernel_content_recovery']}) | gain {row['content_term_gain']}", flush=True)
    for L in ALL_L: H[L].attn.squared_attention = origs[L]
    for h in hconst + hcs: h.remove()

    mid = out['bands']['middle_L6_14']; gat = out['bands']['gatherer_L3_5']
    out['pred_a_two_term_works'] = bool(mid['kernel_content_recovery'] >= 0.75)
    out['pred_b_gatherer_below'] = bool(gat['kernel_content_recovery'] < 0.6)
    out['pred_c_content_inert'] = bool(mid['content_term_gain'] < 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a two-term {out['pred_a_two_term_works']} | pred_b gatherer<0.6 {out['pred_b_gatherer_below']} | pred_c inert {out['pred_c_content_inert']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
