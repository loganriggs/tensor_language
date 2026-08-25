# shared_band_kit: THE CAPSTONE (§1367). One 22-head annotation COMMONS (the union of
# question's and the closer's independently drop-cost-ranked top-16 bands) serving THREE
# kits at once — question (clause gate + 10.5), comparative (two-window gate + 8.1 +
# refine {10.5,12.8,11.7,11.6}), closer (bracket/quote union gate + 13.8) — each scored
# on its own target family with the SAME live commons. The never-gate-a-constant rule
# applies: 5.7 is routed everywhere (it was never in either top-16 anyway).
#
# Registered predictions:
#   pred_a each kit lands within 0.04 BELOW its own-band closing number (one-sided,
#          bar-lesson #4): question >= 0.601, comparative >= 0.738, closer brackets
#          >= 0.617 and quotes >= 0.590.
#   pred_b the commons is <= 22 heads (checked from the committed jsons at runtime).
#   pred_c selectivity holds for all three kits (elsewhere within 0.05 of route).
#
# If pred_a sweeps: ONE 22-head commons + ONE specialist head per capability + zero-bit
# gates ~ two-thirds of every measured capability — the program's headline MDL statement.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'shared_band_kit_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
      'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
COMP = ['bigger', 'smaller', 'better', 'worse', 'larger', 'greater', 'higher', 'lower',
        'faster', 'slower', 'older', 'younger', 'stronger', 'weaker', 'easier', 'harder',
        'longer', 'shorter', 'cheaper', 'richer', 'more', 'less', 'fewer', 'rather']

# the commons, from the committed drop-cost rankings
_c = json.load(open(PT + 'closer_band_slim_results.json'))['ranked_top16']
_q = json.load(open(PT + 'question_kit_slim_results.json'))['ranked_heads'][:16]
COMMONS = {tuple(int(x) for x in h.split('.')) for h in set(_c) | set(_q)}
SPEC = {'question': {(10, 5)},
        'comparative': {(8, 1), (10, 5), (12, 8), (11, 7), (11, 6)},
        'closer': {(13, 8)}}
CUR = {'kit': None}          # which kit's specialists are live


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, gatemask):
    """arm full|ymean|route|kit. kit: COMMONS heads live inside gatemask; the current
    kit's specialist heads live everywhere; everything else v1-routed."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    spec = SPEC.get(CUR['kit'], set())
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
                if not (arm == 'kit' and (L, h) in spec):
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
            vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
            if arm == 'kit' and any((L, h) in COMMONS for h in range(9)):
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                gm = gatemask.view(B, T, 1)
                for h in range(9):
                    if (L, h) in COMMONS:
                        y[:, :, h] = torch.where(gm, y_live[:, :, h], y[:, :, h])
        yo = at.c_proj(y.reshape(B, T, D))
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def qstate(toks, se_t, wh_t):
    is_end = torch.isin(toks, se_t); is_wh = torch.isin(toks, wh_t)
    B2, T2 = toks.shape
    state = torch.zeros(B2, dtype=torch.bool)
    QS = torch.zeros_like(toks, dtype=torch.bool)
    rec = torch.full((B2,), 99, dtype=torch.long)
    for p in range(T2):
        op = is_wh[:, p] & (rec <= 2)
        state = torch.where(is_end[:, p], torch.zeros_like(state), state | op)
        QS[:, p] = state
        rec = torch.where(is_end[:, p], torch.zeros_like(rec), rec + 1)
    return QS


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    qm = set(); se = set(); wh = set(); than = set(); comp = set()
    close_t = set(); open_t = set(); qq = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        ds = d.strip()
        if '?' in d:
            qm.add(tok)
        if any(c in d for c in '.!?'):
            se.add(tok)
        if ds in WH:
            wh.add(tok)
        if ds.lower() == 'than':
            than.add(tok)
        if ds.lower() in COMP:
            comp.add(tok)
        if ')' in d:
            close_t.add(tok)
        if '(' in d:
            open_t.add(tok)
        if '"' in d:
            qq.add(tok)
    tt = lambda s: torch.tensor(sorted(s))
    qm_t, se_t, wh_t, than_t, comp_t = map(tt, (qm, se, wh, than, comp))
    close_i, open_i, q_i = map(tt, (close_t, open_t, qq))
    print(f"commons {len(COMMONS)} heads", flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
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

    # target families + gates on EVR
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]
    QS = qstate(toks, se_t, wh_t)
    T_q = torch.isin(tgt, qm_t) & QS
    is_comp = torch.isin(toks, comp_t)
    ctx = torch.zeros_like(is_comp)
    for w in range(2, 21):
        sh = torch.zeros_like(is_comp)
        sh[:, w:] = is_comp[:, :-w]
        ctx |= sh
    T_c = torch.isin(tgt, than_t) & ctx
    is_open = torch.isin(toks, open_i); is_close = torch.isin(toks, close_i)
    depth = torch.zeros_like(toks)
    dr = torch.zeros(toks.shape[0], dtype=torch.long)
    for p in range(toks.shape[1]):
        dr = (dr + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
        depth[:, p] = dr
    isq = torch.isin(toks, q_i)
    par = (isq.long().cumsum(1) % 2) == 1
    T_b = torch.isin(tgt, close_i) & (depth > 0)
    T_qc = torch.isin(tgt, q_i) & par
    for M in (T_q, T_c, T_b, T_qc):
        M[:, :64] = False
    ALL = T_q | T_c | T_b | T_qc
    ELSE = ~ALL; ELSE[:, :64] = False
    print(f"targets: q {int(T_q.sum())} | than {int(T_c.sum())} | "
          f"bracket {int(T_b.sum())} | quote {int(T_qc.sum())}", flush=True)

    def gates_for(kit, toks_b):
        if kit == 'question':
            return qstate(toks_b, se_t, wh_t)
        if kit == 'comparative':
            ic = torch.isin(toks_b, comp_t)
            cx = torch.zeros_like(ic)
            for w in range(2, 21):
                sh = torch.zeros_like(ic)
                sh[:, w:] = ic[:, :-w]
                cx |= sh
            return ic | cx
        # closer: depth>0 OR odd parity
        io = torch.isin(toks_b, open_i); icl = torch.isin(toks_b, close_i)
        dep = torch.zeros_like(toks_b)
        dr2 = torch.zeros(toks_b.shape[0], dtype=torch.long)
        for p in range(toks_b.shape[1]):
            dr2 = (dr2 + io[:, p].long() - icl[:, p].long()).clamp_min(0)
            dep[:, p] = dr2
        iq = torch.isin(toks_b, q_i)
        pr = (iq.long().cumsum(1) % 2) == 1
        return (dep > 0) | pr

    MASKS = {'question': T_q, 'comparative': T_c, 'closer_b': T_b, 'closer_q': T_qc}

    def ce_run(arm, kit=None):
        CUR['kit'] = kit
        sums = {k: 0.0 for k in MASKS}; ns = {k: 0 for k in MASKS}
        se_ = 0.0; ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg2 = bb[:, 1:].contiguous()
            gm = gates_for(kit, idx.cpu()).to(DEV) if kit else None
            lo = fwd_arm(idx, arm, vmeans, ymeans, gm).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg2.reshape(-1),
                                 reduction='none').view(tg2.shape)
            for k in MASKS:
                mm = MASKS[k][i:i + 8].to(DEV)
                sums[k] += float(ce[mm].sum()); ns[k] += int(mm.sum())
            me = ELSE[i:i + 8].to(DEV)
            se_ += float(ce[me].sum()); ne += int(me.sum())
        return ({k: sums[k] / max(ns[k], 1) for k in MASKS}, se_ / max(ne, 1))

    res = {}
    for arm in ('full', 'ymean', 'route'):
        r, e = ce_run(arm)
        res[arm] = {**{k: round(v, 4) for k, v in r.items()}, 'else': round(e, 4)}
        print(f"{arm}: " + " ".join(f"{k} {r[k]:.3f}" for k in MASKS) + f" | else {e:.4f}",
              flush=True)
    for kit in ('question', 'comparative', 'closer'):
        r, e = ce_run('kit', kit)
        res[f'kit_{kit}'] = {**{k: round(v, 4) for k, v in r.items()}, 'else': round(e, 4)}
        print(f"kit_{kit}: " + " ".join(f"{k} {r[k]:.3f}" for k in MASKS)
              + f" | else {e:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    def rec(arm, fam):
        g = res['ymean'][fam] - res['full'][fam]
        return (res['ymean'][fam] - res[arm][fam]) / max(g, 1e-6)
    def rece(arm):
        g = res['ymean']['else'] - res['full']['else']
        return (res['ymean']['else'] - res[arm]['else']) / max(g, 1e-6)

    scores = {'question': rec('kit_question', 'question'),
              'comparative': rec('kit_comparative', 'comparative'),
              'closer_b': rec('kit_closer', 'closer_b'),
              'closer_q': rec('kit_closer', 'closer_q')}
    bars = {'question': 0.601, 'comparative': 0.738, 'closer_b': 0.617, 'closer_q': 0.590}
    pa = all(scores[k] >= bars[k] for k in bars)
    pb = len(COMMONS) <= 22
    re_route = rece('route')
    pc = all(abs(rece(f'kit_{k}') - re_route) <= 0.05
             for k in ('question', 'comparative', 'closer'))
    out = {'commons_heads': sorted(f'{a}.{b}' for a, b in COMMONS),
           'n_commons': len(COMMONS), 'ce': res,
           'scores': {k: round(v, 4) for k, v in scores.items()}, 'bars': bars,
           'pred_a_all_within': bool(pa), 'pred_b_commons_22': bool(pb),
           'pred_c_selective': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nscores: " + " ".join(f"{k} {scores[k]:.3f} (bar {bars[k]})" for k in bars))
    print(f"pred_a all-within {pa} | pred_b commons<=22 {pb} ({len(COMMONS)}) | "
          f"pred_c selective {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
