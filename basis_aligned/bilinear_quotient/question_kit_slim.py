# question_kit_slim: THE USER'S REFINEMENT of the question kit (§1336). The closed
# description gates WHOLE LAYERS (a02 + L3-5) — 45 heads of weights for a capability the
# per-head §1336 stage showed is carried unevenly (L4's top heads at 0.031-0.034, tail
# near zero). The user's point: a subcomponent should do — most of those heads are dead
# weight in the kit. This run slims the kit to HEAD grain.
#
# Base kit for slimming: route + 10.5 + clause-gated {a02's 27 heads + L4's 9} = the
# §1336 'a02+L4' arm (0.651 target recovery; L3/L5's +0.063 of redundant carriage is
# deliberately dropped — slimming targets the 36-head core, registered choice).
#
# Stage 1 (ranking, NR=960): DROP-ONE from the 36-head gated kit — cost of removing each
#   head from the kit. Drop-cost handles the crowd's redundancy better than solo-adds
#   (a redundant head's drop is ~0 even when its solo-add is positive). Ranking only —
#   never reported as scores.
# Stage 2 (scored, NR=1920): nested kits keeping the top-k heads by drop cost,
#   k in {16, 12, 8, 4}, plus the full 36-head anchor and route/ymean/full.
#
# Registered predictions:
#   pred_a A 16-HEAD KIT IS THE 36-HEAD KIT: top-16 target recovery >= 0.63 (within 0.02
#          of the 36-head 0.651).
#   pred_b AN 8-HEAD KIT BEATS THE WHOLE GATED FRONT BAND: top-8 >= 0.55 (§1334's
#          27-head a02 arm reached 0.545 — a quarter of the heads, more capability).
#   pred_c SELECTIVITY SURVIVES SLIMMING: every kit arm's elsewhere recovery within
#          0.05 of route's.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'question_kit_slim_results.json'
NMEAN = 24; NR1 = 960; NR2 = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
KEEPQ = {(10, 5)}
CORE = [(L, h) for L in (0, 1, 2, 4) for h in range(9)]     # 36 heads
WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
      'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
GATE = {'heads': set()}


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, gatemask):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    glayers = {L for (L, h) in GATE['heads']}
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
        if arm == 'full':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        elif arm == 'ymean':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            for h in range(9):
                y[:, :, h] = ymeans[L][h].to(y.dtype)
        else:
            vr = v.clone()
            for h in range(9):
                if not (arm != 'route' and (L, h) in KEEPQ):
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
            vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
            if arm == 'circ_gate' and L in glayers:
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                gm = gatemask.view(B, T, 1)
                for h in range(9):
                    if (L, h) in GATE['heads']:
                        y[:, :, h] = torch.where(gm, y_live[:, :, h], y[:, :, h])
        yo = at.c_proj(y.reshape(B, T, D))
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def qstate_masks(toks, se_t, wh_t):
    is_end = torch.isin(toks, se_t); is_wh = torch.isin(toks, wh_t)
    B2, T2 = toks.shape
    state = torch.zeros(B2, dtype=torch.bool)
    QSTATE = torch.zeros_like(toks, dtype=torch.bool)
    rec = torch.full((B2,), 99, dtype=torch.long)
    for p in range(T2):
        op = is_wh[:, p] & (rec <= 2)
        state = torch.where(is_end[:, p], torch.zeros_like(state), state | op)
        QSTATE[:, p] = state
        rec = torch.where(is_end[:, p], torch.zeros_like(rec), rec + 1)
    return QSTATE


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    qm = set(); sent_end = set(); wh = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '?' in d:
            qm.add(tok)
        if any(c in d for c in '.!?'):
            sent_end.add(tok)
        if d.strip() in WH:
            wh.add(tok)
    qm_t = torch.tensor(sorted(qm)); se_t = torch.tensor(sorted(sent_end))
    wh_t = torch.tensor(sorted(wh))

    ROWS = cl.fineweb_rows(NMEAN + NR2)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    vs = [[[] for _ in range(9)] for _ in range(18)]
    ys = [[[] for _ in range(9)] for _ in range(18)]
    for i in range(0, NMEAN, 4):
        idx = MEANR[i:i + 4, :-1].to(DEV).contiguous()
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
            for h in range(9):
                vs[L][h].append(v[:, :, h].float().mean((0, 1)).cpu())
                ys[L][h].append(y[:, :, h].float().mean((0, 1)).cpu())
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    vmeans = [torch.stack([torch.stack(vs[L][h]).mean(0) for h in range(9)]).to(DEV)
              for L in range(18)]
    ymeans = [torch.stack([torch.stack(ys[L][h]).mean(0) for h in range(9)]).to(DEV)
              for L in range(18)]

    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    QSTATE = qstate_masks(toks, se_t, wh_t)
    TARGET = torch.isin(tgt_all, qm_t) & QSTATE
    TARGET[:, :64] = False
    ELSE = ~TARGET; ELSE[:, :64] = False
    print(f"targets {int(TARGET.sum())}", flush=True)

    def ce_cond(arm, nr):
        st = se = 0.0; nt = ne = 0
        for i in range(0, nr, 8):
            bb = EVR[i:i + 8]
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].to(DEV).contiguous()
            qs = qstate_masks(idx, se_t, wh_t)
            lo = fwd_arm(idx.to(DEV), arm, vmeans, ymeans, qs.to(DEV)).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            st += float(ce[mt].sum()); nt += int(mt.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
        return st / max(nt, 1), se / max(ne, 1)

    # stage 1: drop-one ranking at NR1
    GATE['heads'] = set(CORE)
    base36, _ = ce_cond('circ_gate', NR1)
    print(f"36-head kit target CE (NR1) {base36:.4f}", flush=True)
    drop = {}
    for hd in CORE:
        GATE['heads'] = set(CORE) - {hd}
        tce, _ = ce_cond('circ_gate', NR1)
        drop[hd] = tce - base36
        print(f"drop L{hd[0]}.{hd[1]}: +{drop[hd]:.4f}", flush=True)
        json.dump({'partial': True,
                   'drop': {f'{a}.{b}': round(v, 4) for (a, b), v in drop.items()}},
                  open(OUT, 'w'), indent=1)
    ranked = sorted(CORE, key=lambda hd: -drop[hd])

    # stage 2: nested kits at NR2
    res = {}
    for arm in ('full', 'ymean', 'route'):
        tce, ece = ce_cond(arm, NR2)
        res[arm] = {'target': round(tce, 4), 'else': round(ece, 4)}
        print(f"{arm}: target {tce:.4f} | else {ece:.4f}", flush=True)
    GATE['heads'] = set(CORE)
    tce, ece = ce_cond('circ_gate', NR2)
    res['kit36'] = {'target': round(tce, 4), 'else': round(ece, 4)}
    print(f"kit36: target {tce:.4f} | else {ece:.4f}", flush=True)
    for kk in (16, 12, 8, 4):
        GATE['heads'] = set(ranked[:kk])
        tce, ece = ce_cond('circ_gate', NR2)
        res[f'kit{kk}'] = {'target': round(tce, 4), 'else': round(ece, 4)}
        print(f"kit{kk}: target {tce:.4f} | else {ece:.4f}", flush=True)

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'target': round((res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    re_ = rec['route']['else']
    pa = rec['kit16']['target'] >= 0.63
    pb = rec['kit8']['target'] >= 0.55
    pc = all(abs(rec[f'kit{kk}']['else'] - re_) <= 0.05 for kk in (36, 16, 12, 8, 4))
    out = {'n_rows_rank': NR1, 'n_rows_score': NR2, 'ce': res, 'recovery': rec,
           'drop_costs': {f'{a}.{b}': round(v, 4) for (a, b), v in drop.items()},
           'ranked_heads': [f'{a}.{b}' for a, b in ranked],
           'pred_a_16_is_36': bool(pa), 'pred_b_8_beats_band': bool(pb),
           'pred_c_selective': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nranked top-8: {[f'{a}.{b}' for a, b in ranked[:8]]}")
    print(f"rec: kit36 {rec['kit36']['target']} kit16 {rec['kit16']['target']} "
          f"kit12 {rec['kit12']['target']} kit8 {rec['kit8']['target']} kit4 {rec['kit4']['target']}")
    print(f"pred_a 16=36 {pa} | pred_b 8>band {pb} | pred_c selective {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
