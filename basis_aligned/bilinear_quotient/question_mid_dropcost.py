# question_mid_dropcost: WHICH MIDS DOES QUESTION BUY BEYOND 4-7? (§1393 found keep-{4,5,6,7}
# loses 0.094 recovery on question — 3x the bar — while riding on comparative/closer.)
# Inside the QUESTION kit (commons gated + 10.5 everywhere, sequential-refit gmean
# stand-ins fit UNDER this kit): (1) drop-cost of each single mid L in 4..15 from
# all-mid-live; (2) greedy build-back from keep-{4,5,6,7} adding pruned mids by drop-cost
# rank until within 0.03 of all-live question recovery.
#
# Registered predictions:
#   pred_a the top pruned mid (highest question drop-cost among 8..15) is in mlp8-11.
#   pred_b keep-4 + top-2 pruned mids reaches within 0.03 of all-live question recovery.
#   pred_c the top pruned mid is question-selective: its question dCE >= 2x its else dCE.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'question_mid_dropcost_results.json'
NFITT = 960
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
      'Did', 'Can', 'Could', 'Will', 'Would', 'Should']

_c = json.load(open(PT + 'closer_band_slim_results.json'))['ranked_top16']
_q = json.load(open(PT + 'question_kit_slim_results.json'))['ranked_heads'][:16]
COMMONS = {tuple(int(x) for x in h.split('.')) for h in set(_c) | set(_q)}
SPEC = {(10, 5)}
MLPMODE = {'cur': None}
MLPTAB = {'gmean': {}, 'installed': set(), 'live_set': set()}
CAPL = {'cur': None}
MIDL = tuple(range(4, 16))


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, gatemask):
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
        if arm == 'full':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        elif arm == 'ymean':
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            for h in range(9):
                y[:, :, h] = ymeans[L][h].to(y.dtype)
        else:  # kit
            vr = v.clone()
            for h in range(9):
                if (L, h) not in SPEC:
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
        mlp_in = F.rms_norm(x, (D,))
        mode = MLPMODE['cur']
        if mode == 'refit_fit' and 4 <= L <= 15:
            if L == CAPL['cur']:
                CAPL['out'] = blk.mlp(mlp_in).detach().float()
            if L in MLPTAB['installed']:
                x = x + MLPTAB['gmean'][L].to(x.dtype)
            else:
                x = x + blk.mlp(mlp_in)
        elif mode is None or arm in ('full', 'ymean') or not (4 <= L <= 15):
            x = x + blk.mlp(mlp_in)
        elif mode == 'mid_live_set':
            if L in MLPTAB['live_set']:
                x = x + blk.mlp(mlp_in)
            else:
                x = x + MLPTAB['gmean'][L].to(x.dtype)
        else:
            x = x + blk.mlp(mlp_in)
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
    qm = set(); se = set(); wh = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '?' in d:
            qm.add(tok)
        if any(c in d for c in '.!?'):
            se.add(tok)
        if d.strip() in WH:
            wh.add(tok)
    tt = lambda s: torch.tensor(sorted(s))
    qm_t, se_t, wh_t = map(tt, (qm, se, wh))
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

    # sequential refit of gmean stand-ins, fit UNDER the question kit (§105)
    FITT = cl.fineweb_rows(NFITT, skip=80)[:, :T + 1].contiguous()
    MLPMODE['cur'] = 'refit_fit'
    for L in MIDL:
        CAPL['cur'] = L
        S = torch.zeros(D, device=DEV); npos = 0
        for i in range(0, NFITT, 8):
            idx3 = FITT[i:i + 8, :-1].to(DEV).contiguous()
            gm = qstate(idx3.cpu(), se_t, wh_t).to(DEV)
            fwd_arm(idx3, 'kit', vmeans, ymeans, gm)
            of = CAPL['out'].reshape(-1, D)
            S += of.sum(0); npos += of.shape[0]
        MLPTAB['gmean'][L] = S / npos
        MLPTAB['installed'].add(L)
        print(f"refit mlp{L} installed", flush=True)
    CAPL['cur'] = None
    MLPMODE['cur'] = None

    toks = EVR[:, :-1]; tgt = EVR[:, 1:]
    QS = qstate(toks, se_t, wh_t)
    T_q = torch.isin(tgt, qm_t) & QS
    T_q[:, :64] = False
    ELSE = ~T_q; ELSE[:, :64] = False
    print(f"targets: q {int(T_q.sum())}", flush=True)

    def ce_run(arm):
        sq = 0.0; nq = 0; se_ = 0.0; ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg2 = bb[:, 1:].contiguous()
            gm = qstate(idx.cpu(), se_t, wh_t).to(DEV)
            lo = fwd_arm(idx, arm, vmeans, ymeans, gm).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg2.reshape(-1),
                                 reduction='none').view(tg2.shape)
            mm = T_q[i:i + 8].to(DEV)
            sq += float(ce[mm].sum()); nq += int(mm.sum())
            me = ELSE[i:i + 8].to(DEV)
            se_ += float(ce[me].sum()); ne += int(me.sum())
        return sq / max(nq, 1), se_ / max(ne, 1)

    res = {}
    MLPMODE['cur'] = None
    for arm in ('full', 'ymean'):
        q, e = ce_run(arm)
        res[arm] = {'q': round(q, 4), 'else': round(e, 4)}
        print(f"{arm}: q {q:.4f} else {e:.4f}", flush=True)

    def kit_run(live_set, tag):
        MLPMODE['cur'] = 'mid_live_set'
        MLPTAB['live_set'] = set(live_set)
        q, e = ce_run('kit')
        res[tag] = {'q': round(q, 4), 'else': round(e, 4)}
        MLPMODE['cur'] = None
        print(f"{tag}: q {q:.4f} else {e:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
        return q, e

    q_live, e_live = kit_run(set(MIDL), 'live')
    q_k4, e_k4 = kit_run({4, 5, 6, 7}, 'keep4')

    # single-drop costs from all-live
    drops = {}
    for L in MIDL:
        qd, ed = kit_run(set(MIDL) - {L}, f'drop{L}')
        drops[L] = {'q': round(qd - q_live, 4), 'else': round(ed - e_live, 4)}
    ranked = sorted([L for L in MIDL if L >= 8], key=lambda L: -drops[L]['q'])
    print("pruned-mid ranking: " + " ".join(f"mlp{L}({drops[L]['q']:+.3f})"
                                            for L in ranked), flush=True)

    gap = res['ymean']['q'] - res['full']['q']
    rec = lambda q: (res['ymean']['q'] - q) / max(gap, 1e-6)
    rl = rec(q_live)
    # greedy build-back
    build = {4, 5, 6, 7}; trail = []
    for L in ranked[:4]:
        build.add(L)
        qb, eb = kit_run(sorted(build), 'keep4+' + '+'.join(str(x) for x in sorted(build - {4, 5, 6, 7})))
        trail.append({'added': L, 'live_set': sorted(build), 'q': round(qb, 4),
                      'rec': round(rec(qb), 4)})
        if rec(qb) >= rl - 0.03:
            break

    top = ranked[0]
    pa = top in (8, 9, 10, 11)
    two = trail[1] if len(trail) > 1 else trail[-1]
    pb = (len(trail) <= 2) and (trail[-1]['rec'] >= rl - 0.03)
    pc = drops[top]['q'] >= 2 * abs(drops[top]['else'])
    out = {'res': res, 'drops': {str(L): drops[L] for L in MIDL},
           'ranked_pruned': [int(x) for x in ranked], 'build_trail': trail,
           'rec_live': round(rl, 4), 'rec_keep4': round(rec(q_k4), 4),
           'pred_a_top_in_8_11': bool(pa), 'pred_b_two_adds_suffice': bool(pb),
           'pred_c_top_selective': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top pruned mlp{top} | pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
