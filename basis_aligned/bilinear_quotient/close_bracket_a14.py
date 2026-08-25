# close_bracket_a14: THE a14 RUNG, SCORED PER SUBTYPE (user request: show the computation
# for ")," / ")." / ")\"" separately). §1341: a14 assists compound closes (+0.17..0.20)
# and is NEGATIVE on plain ")" (-0.060). §1346 closed the core kit at 0.657. This rung
# adds a14 two ways — gated to depth>0 positions, and fully live — and scores every arm
# per target-token subtype. The depth gate CANNOT separate plain from compound closes
# (both live inside brackets), so pred_b is a genuine risk, registered as such.
#
# Registered predictions:
#   pred_a GATED a14 LIFTS THE COMPOUNDS: comma and period subtypes each gain >= +0.05
#          recovery over the kit.
#   pred_b NET-POSITIVE OVERALL: kit+a14_gated >= kit + 0.03 on all targets (risk: the
#          gate also exposes plain closes to a14's negativity).
#   pred_c COMPOUND-CONCENTRATED: comma/period gains each >= 2x the plain gain.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'close_bracket_a14_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
A02L = (0, 1, 2)
A14 = {'mode': None}   # None | 'gated' | 'full'
KEEPQ = {(13, 8)}


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
        else:
            vr = v.clone()
            for h in range(9):
                if not (arm != 'route' and (L, h) in KEEPQ):
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
            vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
            if L in A02L and arm in ('circ_band', 'circ_gate'):
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                if arm == 'circ_band':
                    y = y_live
                else:
                    gm = gatemask.view(B, T, 1, 1)
                    y = torch.where(gm, y_live, y)
            if L == 14 and arm == 'circ_gate' and A14['mode'] is not None:
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                if A14['mode'] == 'full':
                    y = y_live
                else:
                    gm = gatemask.view(B, T, 1, 1)
                    y = torch.where(gm, y_live, y)
        yo = at.c_proj(y.reshape(B, T, D))
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def bracket_masks(toks, open_ids, close_ids):
    """(INSIDE, OPENPOS): depth>0 positions and open-paren positions."""
    is_open = torch.isin(toks, open_ids); is_close = torch.isin(toks, close_ids)
    B2, T2 = toks.shape
    depth = torch.zeros_like(toks)
    d_run = torch.zeros(B2, dtype=torch.long)
    for p in range(T2):
        d_run = (d_run + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
        depth[:, p] = d_run
    return depth > 0, is_open


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    close_t = set(); open_t = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if ')' in d:
            close_t.add(tok)
        if '(' in d:
            open_t.add(tok)
    close_ids = torch.tensor(sorted(close_t)); open_ids = torch.tensor(sorted(open_t))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # per-head fresh-v and y means (same estimator as §1329/31/33)
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

    # targets on EVR
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    QSTATE, WHPOS = bracket_masks(toks, open_ids, close_ids)
    TARGET = torch.isin(tgt_all, close_ids) & QSTATE
    TARGET[:, :64] = False
    ELSE = ~TARGET; ELSE[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"targets {ntar}", flush=True)

    def ce_cond(arm, gate_kind=None):
        st = se = 0.0; nt = ne = 0
        ssub = {s: 0.0 for s in SUBS}; nsub = {s: 0 for s in SUBS}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8]
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].to(DEV).contiguous()
            qs, wp = bracket_masks(idx, open_ids, close_ids)
            gm = {'key': wp, 'qry': qs, None: qs}[gate_kind]
            lo = fwd_arm(idx.to(DEV), arm, vmeans, ymeans, gm.to(DEV)).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            st += float(ce[mt].sum()); nt += int(mt.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
            for s in SUBS:
                mm = SUBMASK[s][i:i + 8].to(DEV)
                ssub[s] += float(ce[mm].sum()); nsub[s] += int(mm.sum())
        return (st / max(nt, 1), se / max(ne, 1),
                {s: ssub[s] / max(nsub[s], 1) for s in SUBS}, dict(nsub))

    res = {}; subce = {}
    for arm, gk, a14mode in (('full', None, None), ('ymean', None, None),
                             ('route', None, None), ('kit', 'qry', None),
                             ('kit_a14g', 'qry', 'gated'), ('kit_a14f', 'qry', 'full')):
        real_arm = 'circ_gate' if arm.startswith('kit') else arm
        A14['mode'] = a14mode
        tce, ece, sc, ns = ce_cond(real_arm, gk)
        res[arm] = {'target': round(tce, 4), 'else': round(ece, 4)}
        subce[arm] = {s: round(v, 4) for s, v in sc.items()}
        print(f"{arm}: target {tce:.4f} | else {ece:.4f} | " +
              " ".join(f"{s} {sc[s]:.3f}" for s in SUBS), flush=True)
    A14['mode'] = None

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'target': round((res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    # per-subtype recovery gains of a14 arms over kit
    subrec = {}
    for s in SUBS:
        g = subce['ymean'][s] - subce['full'][s]
        subrec[s] = {a: round((subce['ymean'][s] - subce[a][s]) / max(g, 1e-6), 4)
                     for a in res if a != 'ymean'}
    gain_g = {s: round(subrec[s]['kit_a14g'] - subrec[s]['kit'], 4) for s in SUBS}
    pa = gain_g['comma'] >= 0.05 and gain_g['period'] >= 0.05
    pb = rec['kit_a14g']['target'] >= rec['kit']['target'] + 0.03
    pc = (gain_g['comma'] >= 2 * max(gain_g['plain'], 1e-4) and
          gain_g['period'] >= 2 * max(gain_g['plain'], 1e-4))
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': res, 'subtype_ce': subce,
           'recovery': rec, 'subtype_recovery': subrec, 'a14_gated_gain': gain_g,
           'pred_a_compounds_lift': bool(pa), 'pred_b_net_positive': bool(pb),
           'pred_c_compound_concentrated': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\na14 gated gains: " + " ".join(f"{s} {gain_g[s]:+.3f}" for s in SUBS))
    print(f"kit {rec['kit']['target']} -> +a14g {rec['kit_a14g']['target']} "
          f"+a14f {rec['kit_a14f']['target']}")
    print(f"pred_a compounds {pa} | pred_b net {pb} | pred_c concentrated {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
