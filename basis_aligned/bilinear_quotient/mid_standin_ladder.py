# mid_standin_ladder: THE USER'S PUSH ON THE §1371 WALL — "dynamic middle" only means
# A CONSTANT CAN'T CARRY IT; the mean is the dumbest stand-in. Ladder of cheap stand-ins
# for mlp4-15 inside the comparative kit (front/top MLPs LIVE — the middle isolated):
#   mid_mean     out = gmean_L                       (the §1371 anchor, ~0.540)
#   mid_gain     out = alpha_pos * gmean_L, alpha = rms(x_pos)/rms_ref  (OOD/calibration:
#                one reference scalar per layer — is the miss just SCALE?)
#   mid_rank1    out = gmean_L + pc1_L * (x_hat . beta_L)  (one direction + one
#                input-computed coefficient per layer — the §650 calibration shape)
#   mid_outlier  out = gmean_L on all dims EXCEPT the outlier/massive dims (per-layer
#                top-8 |gmean| dims + the known 645/990/981), which stay TRUE-live.
#                HYPOTHESIS-TESTING arm (the real MLP runs; description not reduced):
#                if the middle's in-kit service is gain/baseline maintenance on a few
#                dims, this recovers most of the 0.334 and the cheap surrogate for
#                those dims is the follow-up. (USER HYPOTHESIS, registered in his name.)
# Fit: one-shot on mid-LIVE kit-forward captures (disclosed: not layer-sequential;
# the §158 refit upgrade is the logged next step if arms land between mean and live).
#
# Registered predictions (comparative-target recovery; anchor kit_attn ~0.874,
# mid_mean ~0.540, gap 0.334):
#   pred_a (USER, outlier dims): mid_outlier >= 0.74  (>= 60% of the gap).
#   pred_b (gain/OOD):          mid_gain    >= 0.62  (>= 25% of the gap).
#   pred_c (rank-1):            mid_rank1   >= 0.71  (>= 50% of the gap).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mid_standin_ladder_results.json'
NFITT = 960
V = 50257
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
CUR = {'kit': None}
MLPMODE = {'cur': None}
MLPTAB = {'gmean': {}, 'rmsref': {}, 'pc1': {}, 'beta': {}, 'outdims': {}}
MIDL = tuple(range(4, 16))
KNOWN_OUT = [645, 990, 981]


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
        mlp_in = F.rms_norm(x, (D,))
        mode = MLPMODE['cur']
        if mode is None or arm in ('full', 'ymean') or not (4 <= L <= 15):
            x = x + blk.mlp(mlp_in)
        elif mode == 'mid_mean':
            x = x + MLPTAB['gmean'][L].to(x.dtype)
        elif mode == 'mid_gain':
            alpha = (x.float().pow(2).mean(-1, keepdim=True).sqrt()
                     / MLPTAB['rmsref'][L]).to(x.dtype)
            x = x + alpha * MLPTAB['gmean'][L].to(x.dtype)
        elif mode == 'mid_rank1':
            coef = (mlp_in.float() @ MLPTAB['beta'][L]).unsqueeze(-1)
            x = x + (MLPTAB['gmean'][L] + coef * MLPTAB['pc1'][L]).to(x.dtype)
        elif mode == 'mid_outlier':
            true_out = blk.mlp(mlp_in).float()
            rep = MLPTAB['gmean'][L].expand_as(true_out).clone()
            od = MLPTAB['outdims'][L]
            rep[..., od] = true_out[..., od]
            x = x + rep.to(x.dtype)
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

    # ---- fit mid stand-ins on mid-LIVE kit forwards (one-shot, disclosed)
    FITT = cl.fineweb_rows(NFITT, skip=80)[:, :T + 1].contiguous()
    caps = {}
    hs = []
    for L in MIDL:
        def mkc(L):
            def h(mod, args, out):
                caps[L] = (args[0].detach().float(), out.detach().float())
                return out
            return h
        hs.append(H[L].mlp.register_forward_hook(mkc(L)))
    S = {L: torch.zeros(D, device=DEV) for L in MIDL}
    CC = {L: torch.zeros(D, D, device=DEV) for L in MIDL}
    XT = {L: torch.zeros(D, D, device=DEV) for L in MIDL}
    XO = {L: torch.zeros(D, D, device=DEV) for L in MIDL}
    SX = {L: torch.zeros(D, device=DEV) for L in MIDL}
    RS = {L: 0.0 for L in MIDL}
    npos = 0
    MLPMODE['cur'] = None
    CUR['kit'] = 'comparative'
    for i in range(0, NFITT, 8):
        idx3 = FITT[i:i + 8, :-1].to(DEV).contiguous()
        km = torch.isin(idx3.cpu(), comp_t).to(DEV)
        qm_ = torch.zeros_like(km)
        for w in range(2, 21):
            qm_[:, w:] |= km[:, :-w]
        fwd_arm(idx3, 'kit', vmeans, ymeans, km, qm_)
        for L in MIDL:
            xin, out = caps[L]
            xf = xin.reshape(-1, D); of = out.reshape(-1, D)
            S[L] += of.sum(0)
            CC[L] += of.T @ of
            XT[L] += xf.T @ xf
            XO[L] += xf.T @ of
            SX[L] += xf.sum(0)
            RS[L] += float(xin.reshape(-1, D).pow(2).mean(-1).sqrt().sum())
        npos += idx3.numel()
    for h in hs:
        h.remove()
    for L in MIDL:
        gmean = S[L] / npos
        MLPTAB['gmean'][L] = gmean
        MLPTAB['rmsref'][L] = RS[L] / npos
        cov = CC[L] / npos - torch.outer(gmean, gmean)
        evals, evecs = torch.linalg.eigh(cov)
        pc1 = evecs[:, -1]
        MLPTAB['pc1'][L] = pc1
        A = XT[L] + 10.0 * npos / D * torch.eye(D, device=DEV)
        rhs = XO[L] @ pc1 - SX[L] * float(gmean @ pc1)
        MLPTAB['beta'][L] = torch.linalg.solve(A, rhs)
        top8 = gmean.abs().topk(8).indices.tolist()
        MLPTAB['outdims'][L] = torch.tensor(sorted(set(top8 + KNOWN_OUT)), device=DEV)
    print('mid stand-ins fit', flush=True)

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
        MLPMODE['cur'] = None
        r, e = ce_run(arm)
        res[arm] = {**{k: round(v, 4) for k, v in r.items()}, 'else': round(e, 4)}
        print(f"{arm}: " + " ".join(f"{k} {r[k]:.3f}" for k in MASKS) + f" | else {e:.4f}",
              flush=True)
    for mode, name in ((None, 'kit_attn'), ('mid_mean', 'kit_mid_mean'),
                       ('mid_gain', 'kit_mid_gain'), ('mid_rank1', 'kit_mid_rank1'),
                       ('mid_outlier', 'kit_mid_outlier')):
        MLPMODE['cur'] = mode
        r, e = ce_run('kit', 'comparative')
        res[name] = {**{k: round(v, 4) for k, v in r.items()}, 'else': round(e, 4)}
        print(f"{name}: " + " ".join(f"{k} {r[k]:.3f}" for k in MASKS)
              + f" | else {e:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    MLPMODE['cur'] = None

    def rec(arm, fam):
        g = res['ymean'][fam] - res['full'][fam]
        return (res['ymean'][fam] - res[arm][fam]) / max(g, 1e-6)
    def rece(arm):
        g = res['ymean']['else'] - res['full']['else']
        return (res['ymean']['else'] - res[arm]['else']) / max(g, 1e-6)

    r_attn = rec('kit_attn', 'comparative')
    r_mean = rec('kit_mid_mean', 'comparative')
    r_gain = rec('kit_mid_gain', 'comparative')
    r_r1 = rec('kit_mid_rank1', 'comparative')
    r_out = rec('kit_mid_outlier', 'comparative')
    pa = r_out >= 0.74
    pb = r_gain >= 0.62
    pc = r_r1 >= 0.71
    out = {'ce': res,
           'comparative_recovery': {'kit_attn': round(r_attn, 4), 'mid_mean': round(r_mean, 4),
                                    'mid_gain': round(r_gain, 4), 'mid_rank1': round(r_r1, 4),
                                    'mid_outlier': round(r_out, 4)},
           'else_recovery': {a: round(rece(a), 4) for a in res if a != 'ymean'},
           'pred_a_user_outlier': bool(pa), 'pred_b_gain': bool(pb),
           'pred_c_rank1': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nmid ladder: mean {r_mean:.3f} | gain {r_gain:.3f} | rank1 {r_r1:.3f} | "
          f"outlier {r_out:.3f} | live {r_attn:.3f}")
    print(f"pred_a outlier {pa} | pred_b gain {pb} | pred_c rank1 {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
