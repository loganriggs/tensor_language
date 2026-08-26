# compression_rank: THE COMPREHENSIVE DISCOVERY-COMPRESSION RANKING (user: "do we
# actually have that compression ranking?" — we did not; scattered evidence only).
# Fixed suite: 8 verified circuits (comma, question, semicolon, pronouns, is, the,
# months, close_paren). Three discovery compressions, each proposing a top-5 head
# ensemble from the SAME information budget rules:
#   W  — weights-only: class unembedding composed with head output slices.
#   Tn — sink-normalized logit attribution (data: 96 rows).
#   Tr — RAW logit attribution (data: 96 rows; the S1524-deprecated variant, kept
#        as the ablation of the normalization).
# Grading: removal class-rise recovered as a fraction of the GREEDY-VERIFIED
# reference ensemble's class rise, all at NR=960 on identical masks.
#
# Registered predictions:
#   pred_a W's mean recovered fraction >= .60 of the greedy references.
#   pred_b W beats Tn on >= 5 of 8 circuits.
#   pred_c Tn beats Tr on >= 5 of 8 (the normalization matters).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'compression_rank_results.json'
NR = 960
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
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    AR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    CLS = {'comma': rx(r'^,$'), 'question': rx(r'^\?$| \?$'),
           'semicolon': rx(r'^;$'),
           'pronouns': rx(r'^ (he|she|they|He|She|They)$'),
           'is': rx(r'^ is$| was$| are$'), 'the': rx(r'^ the$| The$|^The$'),
           'months': rx(r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$'),
           'close_paren': rx(r'^\)|^ ?\)$')}
    GREEDY_REF = {  # verified class rises of the registry ensembles (NR>=960)
        'comma': 0.1073, 'question': 1.6353, 'semicolon': 0.4973,
        'pronouns': 0.2162, 'is': 0.067, 'the': 0.0353, 'months': 0.0317,
        'close_paren': 0.6548}
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    UD = {}
    for cn, v in CLS.items():
        u = WU[v.to(DEV)].mean(0)
        UD[cn] = u / u.norm()

    # method W scores
    SW = {cn: torch.zeros(18, 9) for cn in CLS}
    for L in range(18):
        Wp = H[L].attn.c_proj.weight.float().to(DEV)
        for hh in range(9):
            blk_ = Wp[:, hh * 128:(hh + 1) * 128]
            for cn in CLS:
                SW[cn][L, hh] = float((UD[cn] @ blk_).norm())
    print("W scored", flush=True)

    # methods Tn / Tr: one capture pass
    STn = {cn: torch.zeros(18, 9) for cn in CLS}
    STr = {cn: torch.zeros(18, 9) for cn in CLS}
    NP = {cn: 0 for cn in CLS}
    for i in range(0, 96, 4):
        bb = AR[i:i + 4]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
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
            for cn in CLS:
                pm = CLS[cn].to(DEV)[tg]
                pm[:, :64] = False
                if int(pm.sum()) == 0:
                    continue
                yy = y.float()[pm]
                for hh in range(9):
                    ov = yy[:, hh, :] @ Wp[:, hh * 128:(hh + 1) * 128].T
                    raw = ov @ UD[cn]
                    STr[cn][L, hh] += float(raw.sum())
                    STn[cn][L, hh] += float((raw / ov.norm(dim=-1)
                                             .clamp_min(1e-6)).sum())
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        for cn in CLS:
            pm = CLS[cn].to(DEV)[tg]; pm[:, :64] = False
            NP[cn] += int(pm.sum())
    print("Tn/Tr scored", flush=True)

    def top5(S):
        return [(int(i) // 9, int(i) % 9)
                for i in S.flatten().argsort(descending=True)[:5]]

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in range(18)]

    def class_rise(hset, mask_v):
        def run():
            cs = 0.0; cn_ = 0
            for i in range(0, NR, 8):
                bb = EVR[i:i + 8].to(DEV)
                idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
                lo = fwd(idx).float()
                ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]),
                                     tg.reshape(-1),
                                     reduction='none').view(tg.shape)
                mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
                cm = mask_v.to(DEV)[tg] & mk
                cs += float(ce[cm].sum()); cn_ += int(cm.sum())
            return cs / max(cn_, 1)
        HSET['set'] = []
        c0 = run()
        HSET['set'] = hset
        c1 = run()
        HSET['set'] = []
        return c1 - c0

    res = {}
    fr = {mth: {} for mth in ('W', 'Tn', 'Tr')}
    for cn in CLS:
        res[cn] = {}
        for mth, S in (('W', SW[cn]), ('Tn', STn[cn]), ('Tr', STr[cn])):
            ens = top5(S)
            rise = class_rise(ens, CLS[cn])
            frac = rise / max(GREEDY_REF[cn], 1e-6)
            fr[mth][cn] = frac
            res[cn][mth] = {'heads': [f'{L}.{h}' for L, h in ens],
                            'class_rise': round(rise, 4),
                            'frac_of_ref': round(frac, 3)}
            print(cn, mth, res[cn][mth]['frac_of_ref'], flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    meanW = sum(fr['W'].values()) / 8
    w_beats = sum(1 for cn in CLS if fr['W'][cn] > fr['Tn'][cn])
    n_beats = sum(1 for cn in CLS if fr['Tn'][cn] > fr['Tr'][cn])
    pa = meanW >= 0.60
    pb = w_beats >= 5
    pc = n_beats >= 5
    out = {'res': res, 'mean_frac': {mth: round(sum(fr[mth].values()) / 8, 3)
                                     for mth in fr},
           'W_beats_Tn': w_beats, 'Tn_beats_Tr': n_beats,
           'pred_a_W_mean_60': bool(pa), 'pred_b_W_beats_Tn_5': bool(pb),
           'pred_c_norm_matters_5': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out['mean_frac'])
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
