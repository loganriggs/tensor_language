# circuit_screen4: TWELVE FRESH CLASSES (registry at 20; keep the generator
# running). Same two-method protocol.
# Original header: MANY CIRCUITS AT ONCE, TWO DISCOVERY METHODS, GRADED (user
# directive 2026-08-26: find many circuits with the per-circuit suite and grade the
# compressions against a BASELINE method). For each of 10 target-token classes:
#   method W (compression-guided, WEIGHTS-ONLY): score each of the 162 heads by the
#     fraction of the class's mean unembedding direction inside that head's c_proj
#     image slice subspace; take the top-5 heads.
#   method B (baseline, DATA logit-attribution): one capture pass; score each head
#     by the mean projection of its actual output (at class-target positions) onto
#     the class unembedding direction; top-5 heads.
# Grade per class and method: replace the 5 heads with their optimal constants,
# measure class-CE rise vs global rise (selectivity), NR=480 screening.
# Classes: newline, capitalized-word, digits, open-quote, close-paren, comma,
# question-mark, ' the'(determiner target), ' of'(preposition), ' is'(copula).
#
# Registered predictions:
#   pred_a >= 7 of 12 yield a selective circuit (>= 2x).
#   pred_b weights-only wins >= 7 of 12.
#   pred_c >= 1 closed class yields a near-single-head circuit (top-1 head carries
#          >= half the top-5 class effect — checked via the W scores, mechanical).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_screen4_results.json'
NR = 480
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')


def class_masks():
    V = 50257
    def rx(pat):
        v = torch.zeros(V, dtype=torch.bool)
        for t in range(V):
            if re.match(pat, ENC.decode([t])):
                v[t] = True
        return v
    C = {}
    C['open_paren'] = rx(r'^ ?\($')
    C['ordinals'] = rx(r'^ (first|second|third|fourth|fifth|last|next)$')
    C['percent'] = rx(r'^ ?(%|percent)$')
    C['according'] = rx(r'^ (according|based|due)$')
    C['pronouns'] = rx(r'^ (he|she|they|He|She|They)$')
    C['days'] = rx(r'^ (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$')
    C['titles'] = rx(r'^ (Mr|Mrs|Dr|Ms|Prof)$')
    C['magnitudes'] = rx(r'^ (billion|million|thousand|hundred)$')
    C['decades'] = rx(r'^ ?(19|20)[0-9]0s$')
    C['however'] = rx(r'^ (However|Meanwhile|Moreover|Furthermore|Additionally)$|^(However|Meanwhile|Moreover)$')
    C['hyphen'] = rx(r'^-$|^ ?--$|^\u2013$|^\u2014$| ?\u2014$')
    C['about'] = rx(r'^ (about|approximately|around|roughly)$')
    return C


@torch.no_grad()
def fwd_heads_capture(idx, store, tgmask_all):
    """Clean fwd capturing per-head outputs y at every layer; accumulate per-class
    per-head logit-attribution."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    ys = {}
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
        ys[L] = y
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    # attribution: for each class c with unembed direction u_c, and each head:
    # mean over class-target positions of (c_proj_h y_h) . u_c
    for cname, (u_c, mask_v) in tgmask_all.items():
        posm = mask_v.to(DEV)[idx.new_zeros(1)]  # placeholder replaced below
    return ys, x


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    CLS = class_masks()
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    UDIR = {}
    for cn, v in CLS.items():
        rows = WU[v.to(DEV)]
        u = rows.mean(0)
        UDIR[cn] = u / u.norm()
    print({k: int(v.sum()) for k, v in CLS.items()}, flush=True)

    # method W: weights-only head scores
    SW = {}
    for cn, u in UDIR.items():
        s = torch.zeros(18, 9)
        for L in range(18):
            W = H[L].attn.c_proj.weight.float().to(DEV)
            for hh in range(9):
                s[L, hh] = float((u @ W[:, hh * 128:(hh + 1) * 128]).norm())
        SW[cn] = s
    print("method W scored", flush=True)

    # method B: data logit-attribution (one capture pass over 96 rows)
    AR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    SB = {cn: torch.zeros(18, 9) for cn in CLS}
    NB = {cn: 0 for cn in CLS}
    for i in range(0, 96, 8):
        bb = AR[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
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
            for cn, u in UDIR.items():
                posm = CLS[cn].to(DEV)[tg.to(DEV)]      # class-target positions
                posm[:, :64] = False
                if int(posm.sum()) == 0:
                    continue
                yy = y.float()[posm]                     # [n, 9, 128]
                for hh in range(9):
                    contr = (yy[:, hh, :] @ Wp[:, hh * 128:(hh + 1) * 128].T) @ u
                    SB[cn][L, hh] += float(contr.sum())
                NB[cn] += int(posm.sum())
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    for cn in SB:
        SB[cn] = SB[cn] / max(NB[cn], 1)
    print("method B scored", flush=True)

    def top5(S):
        flat = S.flatten()
        idxs = flat.argsort(descending=True)[:5]
        return [(int(i) // 9, int(i) % 9) for i in idxs]

    ENSW = {cn: top5(SW[cn]) for cn in CLS}
    ENSB = {cn: top5(SB[cn]) for cn in CLS}

    HSET = {'set': None}

    @torch.no_grad()
    def fwd_rm(idx):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        by_layer = {}
        for (L, hh) in HSET['set']:
            by_layer.setdefault(L, []).append(hh)
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
            if L in by_layer:
                y = y.clone()
                for hh in by_layer[L]:
                    y[:, :, hh, :] = CONSTS[f'head{L}.{hh}'].to(DEV).float() \
                        .to(y.dtype)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)

    def ce_run(hset, cname):
        HSET['set'] = hset if hset else []
        mask_v = CLS[cname].to(DEV)
        s_ = 0.0; n_ = 0; sc = 0.0; nc = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = (fwd_rm(idx) if hset else fwd_rm(idx)).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cls = mask_v[tg] & mk
            s_ += float(ce[mk & ~cls].sum()); n_ += int((mk & ~cls).sum())
            sc += float(ce[cls].sum()); nc += int(cls.sum())
        return s_ / max(n_, 1), sc / max(nc, 1), nc

    res = {}
    HSET['set'] = []
    base = {}
    for cn in CLS:
        g0, c0, nc = ce_run([], cn)
        base[cn] = (g0, c0, nc)
    print("clean measured", flush=True)
    for cn in CLS:
        res[cn] = {'n_class_positions': base[cn][2]}
        for mname, ens in (('W', ENSW[cn]), ('B', ENSB[cn])):
            g1, c1, _ = ce_run(ens, cn)
            g0, c0, _ = base[cn]
            sel = (c1 - c0) / max(g1 - g0, 1e-6)
            res[cn][mname] = {'heads': [f'{L}.{h}' for L, h in ens],
                              'rise_global': round(g1 - g0, 4),
                              'rise_class': round(c1 - c0, 4),
                              'selectivity': round(sel, 2)}
            print(cn, mname, res[cn][mname], flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    n_sel = sum(1 for cn in CLS
                if max(res[cn]['W']['rise_class']
                       / max(res[cn]['W']['rise_global'], 1e-6),
                       res[cn]['B']['rise_class']
                       / max(res[cn]['B']['rise_global'], 1e-6)) >= 2)
    w_wins = sum(1 for cn in CLS
                 if res[cn]['W']['rise_class'] > res[cn]['B']['rise_class'])
    conc = 0
    for cn in CLS:
        s = SW[cn].flatten()
        top = s.argsort(descending=True)
        if float(s[top[0]]) >= 0.5 * float(s[top[:5]].sum()):
            conc += 1
    pa = n_sel >= 7
    pb = w_wins >= 7
    pc = conc >= 1
    out = {'res': res, 'n_selective_2x': n_sel, 'w_wins': w_wins,
           'n_score_concentrated': conc,
           'pred_a_7of12_selective': bool(pa), 'pred_b_w_wins_7': bool(pb),
           'pred_c_single_head_class': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"selective {n_sel}/12 w_wins {w_wins} concentrated {conc}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
