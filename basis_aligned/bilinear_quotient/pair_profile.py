# pair_profile: §1295 registered question — the annotator pair reads DIFFERENTLY per the
# atlas (1.1: 80% self; 1.8: 82% diffuse-other) yet each alone suffices to write the same
# annotation. Measure both heads' |pattern| offset profile AT match-source positions
# (the annotation sites of §1295), and the pair's consumer-concentration under whole-head
# mean ablation.
#
# Registered predictions:
#   pred_a 1.1 IS A SELF-MARKER AT SOURCES: self+prev share of its |pattern| mass at
#          source positions >= 0.6.
#   pred_b 1.8 STAYS DIFFUSE EVEN AT SOURCES: self+prev share <= 0.4 (the atlas shape is
#          not context-gated; two genuinely different algorithms).
#   pred_c PAIR IS CONSUMER-CONCENTRATED: mean-ablating both heads everywhere costs >= 5x
#          more at induction targets than elsewhere.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pair_profile_results.json'
NR = 192; NMEAN = 24; LQ = 1
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h


@torch.no_grad()
def pattern_at(idx, LSTOP):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
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
        if L == LSTOP:
            return pat.abs()
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return None


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    Wd = 128
    TGT = torch.zeros_like(toks, dtype=torch.bool)
    SRC = torch.zeros_like(toks, dtype=torch.bool)
    for b0 in range(0, NR, 64):
        tb = toks[b0:b0 + 64]; gb = tgt_all[b0:b0 + 64]
        eq = (tb.unsqueeze(1) == tb.unsqueeze(2)) & (gb.unsqueeze(1) == gb.unsqueeze(2))
        q_i = torch.arange(T).view(1, T, 1); p_i = torch.arange(T).view(1, 1, T)
        band = (q_i < p_i) & (q_i >= p_i - Wd)
        rel = eq & band
        TGT[b0:b0 + 64] = rel.any(1)
        SRC[b0:b0 + 64] = rel.any(2)
    TGT[:, :16] = False
    ELSE = ~TGT; ELSE[:, :16] = False
    print(f"sources {int(SRC.sum())} | targets {int(TGT.sum())}", flush=True)

    # offset profiles of 1.1 / 1.8 at source positions
    prof = {1: {'self': 0.0, 'prev': 0.0, 'near': 0.0, 'far': 0.0, 'n': 0},
            8: {'self': 0.0, 'prev': 0.0, 'near': 0.0, 'far': 0.0, 'n': 0}}
    for i in range(0, NR, 4):
        idx = toks[i:i + 4].to(DEV).contiguous()
        pat = pattern_at(idx, LQ)
        sm = SRC[i:i + 4].to(DEV)
        for h in (1, 8):
            p = pat[:, h]
            tot = p.sum(-1).clamp_min(1e-9)
            qpos = torch.arange(T, device=DEV).view(1, T, 1)
            kpos = torch.arange(T, device=DEV).view(1, 1, T)
            off = qpos - kpos
            for name, lo, hi in (('self', 0, 0), ('prev', 1, 1), ('near', 2, 8), ('far', 9, 100000)):
                band_m = (off >= lo) & (off <= hi)
                share = (p * band_m.float()).sum(-1) / tot
                prof[h][name] += float(share[sm].sum())
            prof[h]['n'] += int(sm.sum())
    for h in (1, 8):
        n = max(prof[h]['n'], 1)
        for kk in ('self', 'prev', 'near', 'far'):
            prof[h][kk] = round(prof[h][kk] / n, 4)
        print(f"1.{h} at sources: {prof[h]}", flush=True)

    # pair whole-head mean ablation, consumer concentration
    caps = []
    hk = H[LQ].attn.c_proj.register_forward_pre_hook(
        lambda mod, args: caps.append(args[0].detach().float().mean((0, 1))))
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    hk.remove()
    ymean = torch.stack(caps).mean(0)
    SEL = {'on': False, 'heads': (1, 8)}

    def hook(mod, args):
        if not SEL['on']:
            return args
        y = args[0].clone()
        for hh in SEL['heads']:
            y[:, :, hh * 128:(hh + 1) * 128] = ymean[hh * 128:(hh + 1) * 128].to(y.dtype)
        return (y,)

    hk = H[LQ].attn.c_proj.register_forward_pre_hook(hook)

    def ce_sets():
        tots = {'t': 0.0, 'e': 0.0}; ns = {'t': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TGT), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    SEL['on'] = False; base = ce_sets()
    SEL['on'] = True; abl = ce_sets()
    hk.remove()
    d_t = abl['t'] - base['t']; d_e = abl['e'] - base['e']
    conc = d_t / max(d_e, 1e-4)
    print(f"pair ablation: ind dmg {d_t:.4f} else {d_e:.4f} conc {conc:.2f}", flush=True)

    sp1 = prof[1]['self'] + prof[1]['prev']
    sp8 = prof[8]['self'] + prof[8]['prev']
    pa = sp1 >= 0.6
    pb = sp8 <= 0.4
    pc = conc >= 5
    out = {'n_rows': NR, 'profiles': {'1.1': prof[1], '1.8': prof[8]},
           'pair_ablation': {'d_ind': round(d_t, 4), 'd_else': round(d_e, 4), 'conc': round(conc, 2)},
           'pred_a_11_selfmarker': bool(pa), 'pred_b_18_diffuse': bool(pb),
           'pred_c_consumer_concentrated': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a 1.1-self {pa} ({sp1:.3f}) | pred_b 1.8-diffuse {pb} ({sp8:.3f}) | pred_c conc {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
