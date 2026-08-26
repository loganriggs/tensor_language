# circuit_screen3: CONTEXTUAL CIRCUITS (the frontier past target-token classes:
# classes defined by CONTEXT patterns). Classes per position j:
#   copy      — target tg[j] occurred in the previous 64 context tokens.
#   induction — some k<j has idx[k]==idx[j] and idx[k+1]==tg[j] (AB...A -> B).
#   novel_cap — target is a capitalized word NOT present in context (anti-copy).
# Weights-only scores cannot see context, so both methods here are data-driven:
#   method T (target-attribution): head's mean contribution to the CORRECT target
#     logit at class positions (per-position direction u_{tg[j]}).
#   method P (pattern-heuristic): head's mean attention mass on context positions
#     holding the target token (copy/induction) or uniformly (novel_cap: excluded,
#     method T only).
# Top-5 per method, graded by optimal-constant removal selectivity (NR=480).
#
# Registered predictions:
#   pred_a copy and induction both yield >= 2x-selective ensembles.
#   pred_b the copy and induction ensembles overlap (Jaccard >= .3).
#   pred_c method T beats method P on class effect for >= 2 of the 3 gradeable
#          (class, method) comparisons.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_screen3_results.json'
NR = 480
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
HSET = {'set': []}


def mk_hook(L):
    def hook(mod, args):
        hs = [hh for (LL, hh) in HSET['set'] if LL == L]
        if not hs:
            return None
        x = args[0].clone()
        for hh in hs:
            x[:, :, hh * 128:(hh + 1) * 128] = \
                CONSTS[f'head{L}.{hh}'].to(DEV).float().to(x.dtype)
        return (x,)
    return hook


@torch.no_grad()
def fwd_plain(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def class_masks(idx, tg):
    """idx, tg: [B, T] on DEV. Returns dict of [B, T] bool masks."""
    B = idx.shape[0]
    copy = torch.zeros(B, T, dtype=torch.bool, device=DEV)
    induc = torch.zeros(B, T, dtype=torch.bool, device=DEV)
    for lag in range(1, 65):
        past = torch.roll(idx, lag, dims=1)
        past[:, :lag] = -1
        copy |= (past == tg)
        pastn = torch.roll(idx, lag - 1, dims=1)   # token AFTER the matched one
        pastn[:, :max(lag - 1, 1)] = -1
        induc |= (past == idx) & (pastn == tg) & (lag >= 2)
    CAPV = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        pass
    return {'copy': copy, 'induction': induc}


CAPV = None


@torch.no_grad()
def main():
    global CAPV
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    CAPV = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(r'^ [A-Z]', ENC.decode([t])):
            CAPV[t] = True
    CAPV = CAPV.to(DEV)
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in range(18)]

    def masks_for(idx, tg):
        mm = class_masks(idx, tg)
        mm['novel_cap'] = CAPV[tg] & ~mm['copy']
        for k in mm:
            mm[k][:, :64] = False
        return mm

    # ---- scoring pass (96 rows): method T and method P per head per class ----
    AR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    ST = {cn: torch.zeros(18, 9) for cn in ('copy', 'induction', 'novel_cap')}
    SP = {cn: torch.zeros(18, 9) for cn in ('copy', 'induction')}
    NTOT = {cn: 0 for cn in ('copy', 'induction', 'novel_cap')}
    for i in range(0, 96, 4):
        bb = AR[i:i + 4]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV).contiguous()
        mm = masks_for(idx, tg)
        B = idx.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        # source-position mask for method P: context token == target
        src_match = torch.zeros(B, T, T, dtype=torch.bool, device=DEV)
        for j_lag in range(1, 65):
            past = torch.roll(idx, j_lag, dims=1)
            past[:, :j_lag] = -1
            eq = past == tg                       # [B, T] at lag
            jj = torch.arange(T, device=DEV)
            kk = jj - j_lag
            ok = kk >= 0
            src_match[:, jj[ok], kk[ok]] |= eq[:, jj[ok]]
        for L, blk in enumerate(H):
            at = blk.attn
            xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = F.rms_norm(xm, (D,))
            cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
            q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
            q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
                * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            Wp = at.c_proj.weight.float().to(DEV)
            utg = WU[tg.clamp_max(50256)]           # [B, T, D]
            for cn in ST:
                pm = mm[cn]
                if int(pm.sum()) == 0:
                    continue
                yy = y.float()[pm]                   # [n, 9, 128]
                uu = utg[pm]                         # [n, D]
                for hh in range(9):
                    contr = ((yy[:, hh, :] @ Wp[:, hh * 128:(hh + 1) * 128].T)
                             * uu).sum(-1)
                    ST[cn][L, hh] += float(contr.sum())
            for cn in SP:
                pm = mm[cn]
                if int(pm.sum()) == 0:
                    continue
                w_src = (pat * src_match.unsqueeze(1).float()).sum(-1)  # [B,9,T]
                for hh in range(9):
                    SP[cn][L, hh] += float(w_src[:, hh, :].transpose(0, 1)
                                           .T[pm].sum())
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        for cn in ST:
            NTOT[cn] += int(mm[cn].sum())
    for cn in ST:
        ST[cn] /= max(NTOT[cn], 1)
    for cn in SP:
        SP[cn] /= max(NTOT[cn], 1)
    print("scoring done", {c: NTOT[c] for c in NTOT}, flush=True)

    def top5(S):
        return [(int(i) // 9, int(i) % 9)
                for i in S.flatten().argsort(descending=True)[:5]]

    ENS = {}
    for cn in ST:
        ENS[(cn, 'T')] = top5(ST[cn])
    for cn in SP:
        ENS[(cn, 'P')] = top5(SP[cn])

    def measure(hset):
        HSET['set'] = hset
        gs = 0.0; gn = 0
        cs = {cn: 0.0 for cn in ST}; cnn = {cn: 0 for cn in ST}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            mm = masks_for(idx, tg)
            lo = fwd_plain(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            anym = torch.zeros_like(mk)
            for cn in mm:
                anym |= mm[cn]
            gs += float(ce[mk & ~anym].sum()); gn += int((mk & ~anym).sum())
            for cn in mm:
                cs[cn] += float(ce[mm[cn]].sum()); cnn[cn] += int(mm[cn].sum())
        HSET['set'] = []
        return gs / max(gn, 1), {cn: cs[cn] / max(cnn[cn], 1) for cn in ST}, cnn

    g0, c0, nn = measure([])
    print("clean:", {c: round(c0[c], 3) for c in c0}, nn, flush=True)
    res = {}
    for (cn, meth), ens in ENS.items():
        g1, c1, _ = measure(ens)
        sel = (c1[cn] - c0[cn]) / max(g1 - g0, 1e-6)
        res[f'{cn}_{meth}'] = {'heads': [f'{L}.{h}' for L, h in ens],
                               'rise_class': round(c1[cn] - c0[cn], 4),
                               'rise_global': round(g1 - g0, 4),
                               'selectivity': round(sel, 2)}
        print(f'{cn}_{meth}', res[f'{cn}_{meth}'], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    def best_sel(cn):
        keys = [k for k in res if k.startswith(cn)]
        return max(res[k]['rise_class'] / max(res[k]['rise_global'], 1e-6)
                   for k in keys)
    pa = best_sel('copy') >= 2 and best_sel('induction') >= 2
    a_ = set(res['copy_T']['heads']); b_ = set(res['induction_T']['heads'])
    jac = len(a_ & b_) / len(a_ | b_)
    pb = jac >= 0.3
    t_wins = sum(1 for cn in ('copy', 'induction')
                 if res[f'{cn}_T']['rise_class'] > res[f'{cn}_P']['rise_class'])
    pc = t_wins >= 2
    out = {'res': res, 'copy_induction_jaccard': round(jac, 3), 't_wins': t_wins,
           'pred_a_both_selective': bool(pa), 'pred_b_overlap_3': bool(pb),
           'pred_c_T_beats_P': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
