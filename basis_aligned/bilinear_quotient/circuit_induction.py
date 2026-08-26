# circuit_induction: GREEDY REFINEMENT + LAG-SPLIT GENERALIZATION for the S1520
# induction circuit (pattern-method top-5 = 28.9x). Greedy over the top-12
# pattern-method candidates (selectivity-preserving rule); then the final ensemble
# graded separately on SHORT-lag (matched pair within 16 tokens) and LONG-lag
# (17-64) induction masks — does the same ensemble serve both regimes?
#
# Registered predictions:
#   pred_a greedy induction ensemble >= 40x selectivity.
#   pred_b BOTH lag-halves >= 2x selective under the same ensemble.
#   pred_c ensemble size <= 4 heads.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_induction_results.json'
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

    cand = [(int(i) // 9, int(i) % 9)
            for i in SP['induction'].flatten().argsort(descending=True)[:12]]
    print("candidates:", [f'{L}.{h}' for L, h in cand], flush=True)

    def masks_lag(idx, tg, lo_lag, hi_lag):
        B = idx.shape[0]
        mres = torch.zeros(B, T, dtype=torch.bool, device=DEV)
        for lag in range(max(lo_lag, 2), hi_lag + 1):
            past = torch.roll(idx, lag, dims=1)
            past[:, :lag] = -1
            pastn = torch.roll(idx, lag - 1, dims=1)
            pastn[:, :max(lag - 1, 1)] = -1
            mres |= (past == idx) & (pastn == tg)
        mres[:, :64] = False
        return mres

    def measure2(hset, lo_lag=2, hi_lag=64):
        HSET['set'] = hset
        gs = 0.0; gn = 0; cs = 0.0; cn = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            pm = masks_lag(idx, tg, lo_lag, hi_lag)
            lo = fwd_plain(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            gs += float(ce[mk & ~pm].sum()); gn += int((mk & ~pm).sum())
            cs += float(ce[pm].sum()); cn += int(pm.sum())
        HSET['set'] = []
        return gs / max(gn, 1), cs / max(cn, 1)

    SCRN = EVR[:160]

    def measure_scr(hset):
        HSET['set'] = hset
        gs = 0.0; gn = 0; cs = 0.0; cn = 0
        for i in range(0, 160, 8):
            bb = SCRN[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            pm = masks_lag(idx, tg, 2, 64)
            lo = fwd_plain(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            gs += float(ce[mk & ~pm].sum()); gn += int((mk & ~pm).sum())
            cs += float(ce[pm].sum()); cn += int(pm.sum())
        HSET['set'] = []
        return gs / max(gn, 1), cs / max(cn, 1)

    g0s, c0s = measure_scr([])
    cur = []; best = {'rise': 0.0, 'sel': 0.0}
    for hd in cand:
        trial = cur + [hd]
        g1, c1 = measure_scr(trial)
        rise = c1 - c0s; grise = g1 - g0s
        sel = rise / max(grise, 1e-6)
        ok_sel = (not cur) or sel >= 0.9 * best['sel']
        ok_gain = rise >= 1.15 * best['rise'] if cur else rise > 0.01
        if ok_sel and ok_gain and sel >= 2.0:
            cur = trial; best = {'rise': rise, 'sel': sel}
    print("greedy set:", [f'{L}.{h}' for L, h in cur], flush=True)

    g0, c0 = measure2([])
    g1, c1 = measure2(cur)
    selV = (c1 - c0) / max(g1 - g0, 1e-6)
    g0s_, cs0 = measure2([], 2, 16)
    g1s_, cs1 = measure2(cur, 2, 16)
    sel_short = (cs1 - cs0) / max(g1s_ - g0s_, 1e-6)
    g0l_, cl0 = measure2([], 17, 64)
    g1l_, cl1 = measure2(cur, 17, 64)
    sel_long = (cl1 - cl0) / max(g1l_ - g0l_, 1e-6)

    pa = selV >= 40
    pb = sel_short >= 2 and sel_long >= 2
    pc = len(cur) <= 4
    out = {'greedy_heads': [f'{L}.{h}' for L, h in cur],
           'verified': {'sel': round(selV, 2), 'rise_class': round(c1 - c0, 4),
                        'rise_global': round(g1 - g0, 4)},
           'lag_split': {'short_sel': round(sel_short, 2),
                         'short_rise': round(cs1 - cs0, 4),
                         'long_sel': round(sel_long, 2),
                         'long_rise': round(cl1 - cl0, 4)},
           'pred_a_sel_40': bool(pa), 'pred_b_both_lags_2x': bool(pb),
           'pred_c_size_le_4': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out['verified'], out['lag_split'])
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
