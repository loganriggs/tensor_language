# unified_bill_final: CERTIFY THE FINAL BILL CONFIG (§1407: mlp11 buys question .693,
# mlp13 buys capitalized .771 — as SINGLE adds; §1404 says marginals don't compose, so
# the joint sets must be measured). Arms: keep5 (ref), keep6 = {4,6,7,8,9,11},
# keep7 = {4,6,7,8,9,11,13}, all under the unified ungated 28-head kit.
#
# Registered predictions (one-sided):
#   pred_a keep7 question >= 0.68 (the +11 gain survives joint measurement).
#   pred_b keep7 capitalized >= 0.78 (the +13 gain stacks on +11's smaller one).
#   pred_c keep7 comparative >= 0.97 (the record survives two more live mids).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'unified_bill_final_results.json'
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
SPEC = {'unified': {(8, 1), (10, 5), (12, 8), (11, 7), (11, 6), (13, 8)}}
CUR = {'kit': None}
MLPMODE = {'cur': None}
MLPTAB = {'gmean': {}, 'rmsref': {}, 'pc1': {}, 'beta': {}, 'outdims': {},
          'xmean': {}, 'cclip': {}, 'installed': set(), 'live_set': set()}
CAPL = {'cur': None}
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
        prenorm_rms = x.float().pow(2).mean(-1, keepdim=True).sqrt()
        if mode == 'refit_fit' and 4 <= L <= 15:
            if L == CAPL['cur']:
                CAPL['xin'] = mlp_in.detach().float()
                CAPL['out'] = blk.mlp(mlp_in).detach().float()
                CAPL['prms'] = prenorm_rms.detach().float()
            if L in MLPTAB['installed']:
                x = x + MLPTAB['gmean'][L].to(x.dtype)
            else:
                x = x + blk.mlp(mlp_in)
        elif mode is None or arm in ('full', 'ymean') or not (4 <= L <= 15):
            x = x + blk.mlp(mlp_in)
        elif mode == 'mid_mean':
            x = x + MLPTAB['gmean'][L].to(x.dtype)
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

    # ---- SEQUENTIAL refit of mid stand-ins (upstream stand-ins installed)
    FITT = cl.fineweb_rows(NFITT, skip=80)[:, :T + 1].contiguous()
    MLPMODE['cur'] = 'refit_fit'
    CUR['kit'] = 'unified'
    for L in MIDL:
        CAPL['cur'] = L
        S = torch.zeros(D, device=DEV); CC = torch.zeros(D, D, device=DEV)
        XT = torch.zeros(D, D, device=DEV); XO = torch.zeros(D, D, device=DEV)
        SX = torch.zeros(D, device=DEV); RSs = 0.0; npos = 0
        for i in range(0, NFITT, 8):
            idx3 = FITT[i:i + 8, :-1].to(DEV).contiguous()
            fwd_arm(idx3, 'kit', vmeans, ymeans,
                    torch.ones_like(idx3, dtype=torch.bool))
            xf = CAPL['xin'].reshape(-1, D); of = CAPL['out'].reshape(-1, D)
            S += of.sum(0); CC += of.T @ of
            XT += xf.T @ xf; XO += xf.T @ of; SX += xf.sum(0)
            RSs += float(CAPL['prms'].sum()); npos += xf.shape[0]
        gmean = S / npos
        MLPTAB['gmean'][L] = gmean
        MLPTAB['rmsref'][L] = RSs / npos
        MLPTAB['xmean'][L] = SX / npos
        cov = CC / npos - torch.outer(gmean, gmean)
        evals, evecs = torch.linalg.eigh(cov)
        pc1 = evecs[:, -1]
        MLPTAB['pc1'][L] = pc1
        Xc = XT - torch.outer(SX, SX) / npos
        rhs = XO @ pc1 - SX * float(gmean @ pc1)
        lam = 0.01 * float(torch.diagonal(Xc).mean())
        beta = torch.linalg.solve(Xc + lam * torch.eye(D, device=DEV), rhs)
        MLPTAB['beta'][L] = beta
        # coefficient clip range from a fit subsample
        coefs = ((xf - MLPTAB['xmean'][L]) @ beta)
        MLPTAB['cclip'][L] = (float(coefs.quantile(0.01)), float(coefs.quantile(0.99)))
        top8 = gmean.abs().topk(8).indices.tolist()
        MLPTAB['outdims'][L] = torch.tensor(sorted(set(top8 + KNOWN_OUT)), device=DEV)
        MLPTAB['installed'].add(L)
        print(f"refit mlp{L} installed (rmsref {MLPTAB['rmsref'][L]:.1f})", flush=True)
    CAPL['cur'] = None
    MLPMODE['cur'] = None

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
    cap_v = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if len(d) >= 2 and d[0] == ' ' and d[1].isupper() and d[1:].isalpha():
            cap_v.add(tok)
    T_cap = torch.isin(tgt, torch.tensor(sorted(cap_v)))
    for M in (T_q, T_c, T_b, T_qc, T_cap):
        M[:, :64] = False
    ALL = T_q | T_c | T_b | T_qc | T_cap
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

    MASKS = {'question': T_q, 'comparative': T_c, 'closer_b': T_b, 'capitalized': T_cap}

    def ce_run(arm, kit=None):
        CUR['kit'] = kit
        sums = {k: 0.0 for k in MASKS}; ns = {k: 0 for k in MASKS}
        se_ = 0.0; ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg2 = bb[:, 1:].contiguous()
            gm = torch.ones_like(idx, dtype=torch.bool) if kit else None
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
    for arm in ('full', 'ymean'):
        MLPMODE['cur'] = None
        r, e = ce_run(arm)
        res[arm] = {**{k: round(v, 4) for k, v in r.items()}, 'else': round(e, 4)}
        print(f"{arm}: " + " ".join(f"{k} {r[k]:.3f}" for k in MASKS) + f" | else {e:.4f}",
              flush=True)
    KEEP5 = {4, 6, 7, 8, 9}
    sets = [('keep5', KEEP5), ('keep6', KEEP5 | {11}), ('keep7', KEEP5 | {11, 13})]
    for tag, live in sets:
        MLPMODE['cur'] = 'mid_live_set'
        MLPTAB['live_set'] = set(live)
        r, e = ce_run('kit', 'unified')
        res[f'unified_{tag}'] = {**{k: round(v, 4) for k, v in r.items()},
                                 'else': round(e, 4)}
        print(f"unified_{tag}: " + " ".join(f"{k} {r[k]:.3f}" for k in MASKS)
              + f" | else {e:.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    MLPMODE['cur'] = None

    def rec(arm, fam):
        g = res['ymean'][fam] - res['full'][fam]
        return (res['ymean'][fam] - res[arm][fam]) / max(g, 1e-6)
    def rece(arm):
        g = res['ymean']['else'] - res['full']['else']
        return (res['ymean']['else'] - res[arm]['else']) / max(g, 1e-6)

    table = {}
    for tag, _ in sets:
        a = f'unified_{tag}'
        table[tag] = {**{f: round(rec(a, f), 4) for f in MASKS},
                      'else': round(rece(a), 4)}
        print(f"{tag}: {table[tag]}", flush=True)
    pa = table['keep7']['question'] >= 0.68
    pb = table['keep7']['capitalized'] >= 0.78
    pc = table['keep7']['comparative'] >= 0.97
    out = {'ce': res, 'recovery_table': table,
           'pred_a_q_68': bool(pa), 'pred_b_cap_78': bool(pb),
           'pred_c_comp_97': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
