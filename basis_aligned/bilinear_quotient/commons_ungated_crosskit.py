# commons_ungated_crosskit: SHOULD THE TWO-TIER BILL DROP ITS GATES? (§1401: ungating
# the commons lifted capitalized .644->.813 AND improved elsewhere.) For each family
# (comparative / question / closer / capitalized): kit arm with commons GATED on the
# family gate vs commons UNGATED (live everywhere). Specialists per family live ungated
# (SPEC sets from the §1381-era kits; capitalized gets NO specialist — §1401 showed the
# committee adds nothing to construction). All MLPs live. NR=1920. Assumptions
# registered: family gates as in mid_owners_crosskit / capitalized_kit.
#
# Registered predictions:
#   pred_a ungating improves target recovery for ALL FOUR families (4/4 positive).
#   pred_b mean target-recovery gain across families >= 0.05.
#   pred_c elsewhere recovery gain from ungating >= 0.10 (averaged over the four
#          family runs — commons is generic service, else benefits at least as much).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'commons_ungated_crosskit_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
      'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
COMP = ['bigger', 'smaller', 'better', 'worse', 'larger', 'greater', 'higher', 'lower',
        'faster', 'slower', 'older', 'younger', 'stronger', 'weaker', 'easier', 'harder',
        'longer', 'shorter', 'cheaper', 'richer', 'more', 'less', 'fewer', 'rather']

_c = json.load(open(PT + 'closer_band_slim_results.json'))['ranked_top16']
_q = json.load(open(PT + 'question_kit_slim_results.json'))['ranked_heads'][:16]
COMMONS = {tuple(int(x) for x in h.split('.')) for h in set(_c) | set(_q)}
SPEC = {'question': {(10, 5)},
        'comparative': {(8, 1), (10, 5), (12, 8), (11, 7), (11, 6)},
        'closer': {(13, 8)},
        'capitalized': set()}
CUR = {'kit': None}


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, gatemask):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    spec = SPEC.get(CUR['kit'] or '', set())
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
                if (L, h) not in spec:
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
            vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
            if any((L, h) in COMMONS for h in range(9)):
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                gm = gatemask.view(B, T, 1)
                for h in range(9):
                    if (L, h) in COMMONS:
                        y[:, :, h] = torch.where(gm, y_live[:, :, h], y[:, :, h])
        x = xm + at.c_proj(y.reshape(B, T, D))
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
    close_t = set(); open_t = set(); cap = set(); gate_tok = set()
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
        if len(d) >= 2 and d[0] == ' ' and d[1].isupper() and d[1:].isalpha():
            cap.add(tok)
        if any(c in d for c in '.!?\n:"') or (len(d) >= 1 and d.lstrip(' ')[:1].isupper()):
            gate_tok.add(tok)
    tt = lambda s: torch.tensor(sorted(s))
    qm_t, se_t, wh_t, than_t, comp_t = map(tt, (qm, se, wh, than, comp))
    close_i, open_i, cap_i, gate_i = map(tt, (close_t, open_t, cap, gate_tok))

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
    T_b = torch.isin(tgt, close_i) & (depth > 0)
    T_cap = torch.isin(tgt, cap_i)
    for M in (T_q, T_c, T_b, T_cap):
        M[:, :64] = False
    ALL = T_q | T_c | T_b | T_cap
    ELSE = ~ALL; ELSE[:, :64] = False
    MASKS = {'question': T_q, 'comparative': T_c, 'closer': T_b, 'capitalized': T_cap}
    print("targets: " + " ".join(f"{k} {int(v.sum())}" for k, v in MASKS.items()), flush=True)

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
        if kit == 'closer':
            io = torch.isin(toks_b, open_i); icl = torch.isin(toks_b, close_i)
            dep = torch.zeros_like(toks_b)
            dr2 = torch.zeros(toks_b.shape[0], dtype=torch.long)
            for p in range(toks_b.shape[1]):
                dr2 = (dr2 + io[:, p].long() - icl[:, p].long()).clamp_min(0)
                dep[:, p] = dr2
            return dep > 0
        return torch.isin(toks_b, gate_i)   # capitalized

    def ce_run(arm, kit=None, ungated=False):
        CUR['kit'] = kit
        sums = {k: 0.0 for k in MASKS}; ns = {k: 0 for k in MASKS}
        se_ = 0.0; ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg2 = bb[:, 1:].contiguous()
            if kit:
                gm = (torch.ones_like(idx, dtype=torch.bool) if ungated
                      else gates_for(kit, idx.cpu()).to(DEV))
            else:
                gm = None
            lo = fwd_arm(idx, arm, vmeans, ymeans, gm).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg2.reshape(-1),
                                 reduction='none').view(tg2.shape)
            for kk in MASKS:
                mm = MASKS[kk][i:i + 8].to(DEV)
                sums[kk] += float(ce[mm].sum()); ns[kk] += int(mm.sum())
            me = ELSE[i:i + 8].to(DEV)
            se_ += float(ce[me].sum()); ne += int(me.sum())
        return ({kk: sums[kk] / max(ns[kk], 1) for kk in MASKS}, se_ / max(ne, 1))

    res = {}
    for arm in ('full', 'ymean'):
        r, e = ce_run(arm)
        res[arm] = {**{kk: round(v, 4) for kk, v in r.items()}, 'else': round(e, 4)}
        print(f"{arm}: " + " ".join(f"{kk} {r[kk]:.3f}" for kk in r) + f" | else {e:.4f}",
              flush=True)
    for kit in MASKS:
        for tag, ug in (('gated', False), ('ungated', True)):
            r, e = ce_run('kit', kit, ug)
            res[f'{kit}_{tag}'] = {**{kk: round(v, 4) for kk, v in r.items()},
                                   'else': round(e, 4)}
            print(f"{kit}_{tag}: " + " ".join(f"{kk} {r[kk]:.3f}" for kk in r)
                  + f" | else {e:.4f}", flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    def rec(a, fam):
        g = res['ymean'][fam] - res['full'][fam]
        return (res['ymean'][fam] - res[a][fam]) / max(g, 1e-6)
    def rece(a):
        g = res['ymean']['else'] - res['full']['else']
        return (res['ymean']['else'] - res[a]['else']) / max(g, 1e-6)

    table = {}; gains = []; egains = []
    for kit in MASKS:
        gg = rec(f'{kit}_gated', kit); uu = rec(f'{kit}_ungated', kit)
        table[kit] = {'gated': round(gg, 4), 'ungated': round(uu, 4),
                      'gain': round(uu - gg, 4),
                      'else_gated': round(rece(f'{kit}_gated'), 4),
                      'else_ungated': round(rece(f'{kit}_ungated'), 4)}
        gains.append(uu - gg)
        egains.append(rece(f'{kit}_ungated') - rece(f'{kit}_gated'))
        print(f"{kit}: gated {gg:.3f} ungated {uu:.3f} gain {uu - gg:+.3f}", flush=True)
    pa = all(g > 0 for g in gains)
    pb = sum(gains) / len(gains) >= 0.05
    pc = sum(egains) / len(egains) >= 0.10
    out = {'res': res, 'table': table,
           'pred_a_all_four_improve': bool(pa), 'pred_b_mean_gain_05': bool(pb),
           'pred_c_else_gain_10': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
