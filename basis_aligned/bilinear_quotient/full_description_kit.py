# full_description_kit: THE MLP SIDE, PRICED AT LAST (§1343/§1369 frontier; the user's
# "did you mean ablate the mlps too?" made the gap explicit). Every extraction so far ran
# ATTENTION-ONLY with all 18 MLPs live. This run replaces the MLPs with module-ladder
# stand-ins INSIDE the comparative kit (commons + 8.1 + refine, two-window gates):
#   mlp0-3    per-token mean tables (ladder ceilings 0.86/0.94/0.72/0.59, §1322-26)
#   mlp4-15   constant mean (the inert middle: 0.64 nats total, §1326)
#   mlp16-17  per-token tables (ceilings ~0.49; the 16x16 context upgrade of §1332 is
#             logged as the follow-up, registered choice: tables first)
# Arms: full | ymean | kit_attn (MLPs live — the §1368 anchor) | kit_mid_mean (only
# mlp4-15 meaned) | kit_full_desc (all MLP stand-ins).
#
# Registered predictions:
#   pred_a THE MIDDLE IS FREE HERE TOO: kit_mid_mean within 0.03 of kit_attn on
#          comparative targets (one-sided: not more than 0.03 BELOW).
#   pred_b THE FULL DESCRIPTION CARRIES: kit_full_desc >= 0.55 comparative-target
#          recovery (from kit_attn's 0.874 — the tables' cost is real and unknown).
#   pred_c IT REMAINS A LANGUAGE MODEL: kit_full_desc elsewhere recovery >= 0.30.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'full_description_kit_results.json'
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
MLPTAB = {'tabs': {}, 'gmean': {}}
TABLE_LAYERS = (0, 1, 2, 3, 16, 17)


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
        if mode is None or arm in ('full', 'ymean'):
            x = x + blk.mlp(mlp_in)
        elif mode == 'mid_mean' and 4 <= L <= 15:
            x = x + MLPTAB['gmean'][L].to(x.dtype)
        elif mode == 'full_desc':
            if 4 <= L <= 15:
                x = x + MLPTAB['gmean'][L].to(x.dtype)
            elif L in MLPTAB['tabs']:
                x = x + MLPTAB['tabs'][L][idx].to(x.dtype)
            else:
                x = x + blk.mlp(mlp_in)
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

    # ---- fit MLP tables on FITT rows (FULL model — the ladder convention §1322;
    # the §105 capture/apply caveat is DISCLOSED: tables are fit on the intact model and
    # applied inside the kit; sequential refit is the logged upgrade, not run here)
    FITT = cl.fineweb_rows(NFITT, skip=80)[:, :T + 1].contiguous()
    sums = {L: torch.zeros(V, D, device=DEV) for L in TABLE_LAYERS}
    cnts = torch.zeros(V, device=DEV)
    allsums = {L: torch.zeros(D, device=DEV) for L in range(18)}
    npos = 0
    caps = {}
    hooks = []
    for L in range(18):
        def mk(L):
            def h(mod, args, out):
                caps[L] = out.detach().float()
                return out
            return h
        hooks.append(H[L].mlp.register_forward_hook(mk(L)))
    MLPMODE['cur'] = None
    for i in range(0, NFITT, 8):
        idx2 = FITT[i:i + 8, :-1].to(DEV).contiguous()
        x2 = F.rms_norm(m.transformer.wte(idx2), (D,)); x02 = x2; v12 = None
        for blk2 in H:
            x2, v12 = blk2(x2, v12, x02)
        ft = idx2.reshape(-1)
        for L in TABLE_LAYERS:
            sums[L].index_add_(0, ft, caps[L].reshape(-1, D))
        for L in range(18):
            allsums[L] += caps[L].reshape(-1, D).sum(0)
        cnts.index_add_(0, ft, torch.ones_like(ft, dtype=torch.float))
        npos += ft.numel()
    for h in hooks:
        h.remove()
    for L in range(18):
        MLPTAB['gmean'][L] = allsums[L] / max(npos, 1)
    for L in TABLE_LAYERS:
        gm = MLPTAB['gmean'][L]
        MLPTAB['tabs'][L] = torch.where(cnts.unsqueeze(1) > 0,
                                        sums[L] / cnts.clamp_min(1).unsqueeze(1),
                                        gm.unsqueeze(0))
    print('MLP tables fit', flush=True)

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
                       ('full_desc', 'kit_full_desc')):
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
    r_mid = rec('kit_mid_mean', 'comparative')
    r_full = rec('kit_full_desc', 'comparative')
    pa = r_mid >= r_attn - 0.03
    pb = r_full >= 0.55
    pc = rece('kit_full_desc') >= 0.30
    out = {'ce': res,
           'comparative_recovery': {'kit_attn': round(r_attn, 4),
                                    'kit_mid_mean': round(r_mid, 4),
                                    'kit_full_desc': round(r_full, 4)},
           'else_recovery': {a: round(rece(a), 4)
                             for a in ('route', 'kit_attn', 'kit_mid_mean', 'kit_full_desc')},
           'pred_a_middle_free': bool(pa), 'pred_b_full_desc_carries': bool(pb),
           'pred_c_still_an_lm': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\ncomparative rec: attn {r_attn:.3f} | mid-mean {r_mid:.3f} | full-desc {r_full:.3f}")
    print(f"pred_a middle-free {pa} | pred_b carries {pb} | pred_c still-LM {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
