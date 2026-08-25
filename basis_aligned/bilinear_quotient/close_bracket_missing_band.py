# close_bracket_missing_band: THE §1344 ANALOGUE FOR BRACKETS — which band's LIVE service
# carries the kit's remaining ~0.34, measured BY CONSTRUCTION (the §1347 rule: ablation
# names roles, construction prices kits). Arms: kit | kit + one fully-live band
# (L3-5 / L6-9 / L10-12 / L14-17; L13 excluded — the owner is already live).
# Per-subtype diagnostics retained.
#
# Registered predictions:
#   pred_a SOME BAND CARRIES IT: best band adds >= 0.08 target recovery over the kit.
#   pred_b SPECIFICITY: the winner's target increment >= 1.5x its elsewhere increment
#          (comparative's refine passed at 1.8x).
#   pred_c DOWNSTREAM: the winner is L14-17 — annotate->fetch->REFINE predicts
#          post-fetch service above the L13 owner, as it found L10-12 above the L8 owner.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'close_bracket_missing_band_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
A02L = (0, 1, 2)
BANDS = {'b35': (3, 4, 5), 'b69': (6, 7, 8, 9), 'b1012': (10, 11, 12), 'b1417': (14, 15, 16, 17)}
CURBAND = {'layers': (), 'on': False}
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
            if arm == 'circ_gate' and CURBAND['on'] and L in CURBAND['layers']:
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
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
    # subtype lookup (§1341)
    sub_of = {}
    for tok in close_ids.tolist():
        ds = enc.decode([tok])
        if ds.strip() == ')':
            sub_of[tok] = 'plain'
        elif '))' in ds:
            sub_of[tok] = 'double'
        elif '),' in ds:
            sub_of[tok] = 'comma'
        elif ').' in ds:
            sub_of[tok] = 'period'
        elif ')"' in ds or ")'" in ds or '")' in ds or "')" in ds:
            sub_of[tok] = 'quote'
        else:
            sub_of[tok] = 'other'
    SUBS = ('plain', 'comma', 'period', 'quote', 'other', 'double')
    names = {s: i + 1 for i, s in enumerate(SUBS)}
    sub_lookup = torch.zeros(50257, dtype=torch.long)
    for tok, s2 in sub_of.items():
        sub_lookup[tok] = names[s2]
    tgt_sub = sub_lookup[tgt_all]
    SUBMASK = {s2: TARGET & (tgt_sub == names[s2]) for s2 in SUBS}
    for s2 in SUBS:
        print(f"{s2}: n {int(SUBMASK[s2].sum())}", flush=True)

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
    arms = [('full', None, None), ('ymean', None, None), ('route', None, None),
            ('kit', 'qry', None)] + [(f'kit+{b}', 'qry', b) for b in BANDS]
    for arm, gk, band in arms:
        real_arm = 'circ_gate' if arm.startswith('kit') else arm
        CURBAND['on'] = band is not None
        CURBAND['layers'] = BANDS.get(band, ())
        tce, ece, sc, ns = ce_cond(real_arm, gk)
        res[arm] = {'target': round(tce, 4), 'else': round(ece, 4)}
        subce[arm] = {s: round(v, 4) for s, v in sc.items()}
        print(f"{arm}: target {tce:.4f} | else {ece:.4f} | " +
              " ".join(f"{s} {sc[s]:.3f}" for s in SUBS), flush=True)
    CURBAND['on'] = False

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'target': round((res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    kt, ke = rec['kit']['target'], rec['kit']['else']
    inc = {b: {'target': round(rec[f'kit+{b}']['target'] - kt, 4),
               'else': round(rec[f'kit+{b}']['else'] - ke, 4)} for b in BANDS}
    winner = max(inc, key=lambda b: inc[b]['target'])
    pa = inc[winner]['target'] >= 0.08
    pb = inc[winner]['target'] >= 1.5 * max(inc[winner]['else'], 1e-4)
    pc = winner == 'b1417'
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': res, 'subtype_ce': subce,
           'recovery': rec, 'band_increments': inc, 'winner': winner,
           'pred_a_band_carries': bool(pa), 'pred_b_specific': bool(pb),
           'pred_c_downstream': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nincrements: " + " ".join(f"{b} t{inc[b]['target']:+.3f}/e{inc[b]['else']:+.3f}"
                                        for b in BANDS))
    print(f"winner {winner} | pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
