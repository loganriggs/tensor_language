# circuit_screen3: CONTEXTUAL CIRCUITS (the frontier past target-token classes:
# classes defined by CONTEXT patterns). Classes per position j:
#   copy      — target occurred in the previous 64 context tokens AND is an
#               INFREQUENT token (eval-count rank > 500): content copying, not
#               function-word recurrence (S1520 fix).
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
# circuit_match_attn: DOES THE COMMITTEE ATTEND TO THE MATCH? (S1528: the pipeline
# story failed — the committee must do its own context-matching.) One capture pass:
# per-head attention mass on context positions holding the CURRENT TARGET token,
# averaged over (a) cap_copied positions, (b) copy_noncap positions, (c) all other
# positions (base rate). Reported for the 13 committee heads, head 5.5, and the
# 162-head average.
# Registered predictions:
#   pred_a committee-mean attention-to-match at cap_copied positions >= 5x the
#          committee's base rate at other positions.
#   pred_b committee-mean at cap_copied >= 2x the all-head mean at cap_copied
#          (the committee is SPECIFICALLY the matcher, not just everyone).
#   pred_c head 5.5's match rate at copy_noncap >= its rate at cap_copied
#          (5.5 serves the generic channel, per S1528).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_match_attn_results.json'
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
FREQOK = None   # infrequent-token mask (rank > 500 by eval count)


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
    global FREQOK
    ecnt = torch.zeros(50257)
    for i in range(0, NR, 8):
        ecnt.index_add_(0, EVR[i:i + 8, :-1].reshape(-1), torch.ones(8 * T))
    thresh = ecnt.sort(descending=True).values[500]
    FREQOK = (ecnt < thresh).to(DEV)
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in range(18)]

    def masks_for(idx, tg):
        mm = class_masks(idx, tg)
        mm['copy'] = mm['copy'] & FREQOK[tg]
        mm['novel_cap'] = CAPV[tg] & ~mm['copy']
        for k in mm:
            mm[k][:, :64] = False
        return mm

    COMM = [(13, 0), (13, 5), (14, 4), (14, 6), (14, 7), (15, 3), (16, 0),
            (16, 3), (16, 4), (16, 5), (17, 0), (17, 1), (17, 2)]
    NRm = 240
    acc = torch.zeros(18, 9, 3)   # [head, class: cap_copied/copy_noncap/other]
    cnt = torch.zeros(3)
    for i in range(0, NRm, 4):
        bb = EVR[i:i + 4].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
        B = idx.shape[0]
        mm0 = class_masks(idx, tg)
        capm = CAPV[tg].clone(); capm[:, :64] = False
        cls_masks = {0: capm & mm0['copy'],
                     1: mm0['copy'] & FREQOK[tg] & ~capm}
        other = torch.ones_like(capm); other[:, :64] = False
        other &= ~(cls_masks[0] | cls_masks[1])
        cls_masks[2] = other
        # source-match mask
        src_match = torch.zeros(B, T, T, dtype=torch.bool, device=DEV)
        for j_lag in range(1, 65):
            past = torch.roll(idx, j_lag, dims=1)
            past[:, :j_lag] = -1
            eq = past == tg
            jj = torch.arange(T, device=DEV)
            kk = jj - j_lag
            ok = kk >= 0
            src_match[:, jj[ok], kk[ok]] |= eq[:, jj[ok]]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
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
            wmatch = (pat.abs() * src_match.unsqueeze(1).float()).sum(-1) \
                / pat.abs().sum(-1).clamp_min(1e-9)          # [B, 9, T]
            for ci in (0, 1, 2):
                pm = cls_masks[ci]
                if int(pm.sum()) == 0:
                    continue
                for hh in range(9):
                    acc[L, hh, ci] += float(wmatch[:, hh, :][pm].sum())
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        for ci in (0, 1, 2):
            cnt[ci] += int(cls_masks[ci].sum())
    rate = acc / cnt.clamp_min(1)
    comm_rate = {ci: float(sum(rate[L, h, ci] for L, h in COMM) / len(COMM))
                 for ci in (0, 1, 2)}
    all_rate = {ci: float(rate[:, :, ci].mean()) for ci in (0, 1, 2)}
    r55 = {ci: float(rate[5, 5, ci]) for ci in (0, 1, 2)}
    top_matchers = sorted(((float(rate[L, h, 0]), f'{L}.{h}')
                           for L in range(18) for h in range(9)), reverse=True)[:10]

    pa = comm_rate[0] >= 5 * max(comm_rate[2], 1e-9)
    pb = comm_rate[0] >= 2 * max(all_rate[0], 1e-9)
    pc = r55[1] >= r55[0]
    out = {'committee_rate': {k: round(v, 4) for k, v in comm_rate.items()},
           'all_head_rate': {k: round(v, 4) for k, v in all_rate.items()},
           'head55_rate': {k: round(v, 4) for k, v in r55.items()},
           'top10_matchers_at_cap_copied': [[n, round(v, 4)]
                                            for v, n in top_matchers],
           'classes': '0=cap_copied 1=copy_noncap 2=other',
           'pred_a_comm_5x_base': bool(pa), 'pred_b_comm_2x_all': bool(pb),
           'pred_c_55_generic': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(out)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
