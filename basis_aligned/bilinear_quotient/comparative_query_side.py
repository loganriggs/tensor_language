# comparative_query_side: THE §1331 DECOMPOSITION'S DIRECT TEST. Position-gating a02 at
# comparative (KEY-side) positions kept only 27% of the band's comparative increment
# (+0.070 of +0.263 over route); the inference was that the dominant service is
# QUERY-side — the front band's processing at the positions where the "than"-prediction
# is actually made. This run gates the a02 band live at:
#   circ_key   comparative positions only            (§1331's circ_pos, re-run as anchor)
#   circ_qry   query-side positions only             (ctx window: a comparative 2-20 back)
#   circ_both  key + query positions
# 8.1 live in every circ arm; route grain everywhere (§1314/16); anchors recomputed.
#
# Registered predictions:
#   pred_a BOTH SIDES RECOVER THE BAND: circ_both target recovery >= 0.65 (band ~0.69).
#   pred_b QUERY DOMINATES: circ_qry target recovery >= circ_key's + 0.05.
#   pred_c ALL SELECTIVE: every circ_* arm's elsewhere recovery within 0.05 of route's.
# Note the query-side gate is a token-property of the input stream (comparative 2-20
# back), so like the key gate it costs ~zero description bits.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_query_side_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
A02L = (0, 1, 2)
KEEP81 = {(8, 1)}


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, gatemask):
    """gatemask (B,T) bool: where the a02 band runs live (circ_key/qry/both).
    arm full|ymean|route|circ_band|circ_gate. 8.1 live in circ_*."""
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
                if not (arm != 'route' and (L, h) in KEEP81):
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
        yo = at.c_proj(y.reshape(B, T, D))
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    COMP = ['bigger', 'smaller', 'better', 'worse', 'larger', 'greater', 'higher',
            'lower', 'faster', 'slower', 'older', 'younger', 'stronger', 'weaker',
            'easier', 'harder', 'longer', 'shorter', 'cheaper', 'richer', 'more', 'less',
            'fewer', 'rather']
    than = set(); comp = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if d.strip().lower() == 'than':
            than.add(tok)
        if d.strip().lower() in COMP:
            comp.add(tok)
    than_t = torch.tensor(sorted(than)); comp_t = torch.tensor(sorted(comp))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # per-head fresh-v and y means (same estimator as §1329/§1331)
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

    # masks on EVR
    tgt_all = EVR[:, 1:]; toks = EVR[:, :-1]
    is_comp = torch.isin(toks, comp_t)
    ctx = torch.zeros_like(is_comp)
    for w in range(2, 21):
        sh = torch.zeros_like(is_comp)
        sh[:, w:] = is_comp[:, :-w]
        ctx |= sh
    TARGET = torch.isin(tgt_all, than_t) & ctx
    TARGET[:, :64] = False
    ELSE = ~TARGET; ELSE[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"targets {ntar}", flush=True)

    def gates(idx_cpu):
        key = torch.isin(idx_cpu, comp_t)
        qry = torch.zeros_like(key)
        for w in range(2, 21):
            sh = torch.zeros_like(key)
            sh[:, w:] = key[:, :-w]
            qry |= sh
        return key, qry

    def ce_cond(arm, gate_kind=None):
        st = se = 0.0; nt = ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8]
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].to(DEV).contiguous()
            key, qry = gates(idx)
            gm = {'key': key, 'qry': qry, 'both': key | qry, None: key}[gate_kind]
            lo = fwd_arm(idx.to(DEV), arm, vmeans, ymeans, gm.to(DEV)).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            st += float(ce[mt].sum()); nt += int(mt.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
        return st / max(nt, 1), se / max(ne, 1)

    res = {}
    for arm, gk in (('full', None), ('ymean', None), ('route', None), ('circ_band', None),
                    ('circ_key', 'key'), ('circ_qry', 'qry'), ('circ_both', 'both')):
        real_arm = 'circ_gate' if arm.startswith('circ_') and arm != 'circ_band' else arm
        tce, ece = ce_cond(real_arm, gk)
        res[arm] = {'target': round(tce, 4), 'else': round(ece, 4)}
        print(f"{arm}: target {tce:.4f} | else {ece:.4f}", flush=True)

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'target': round((res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    re_ = rec['route']['else']
    pa = rec['circ_both']['target'] >= 0.65
    pb = rec['circ_qry']['target'] >= rec['circ_key']['target'] + 0.05
    pc = all(abs(rec[a]['else'] - re_) <= 0.05
             for a in ('circ_key', 'circ_qry', 'circ_both'))
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': res, 'recovery': rec,
           'gap_target': round(gt, 4), 'gap_else': round(ge, 4),
           'pred_a_both_recover_band': bool(pa), 'pred_b_query_dominates': bool(pb),
           'pred_c_all_selective': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nrec target: band {rec['circ_band']['target']} key {rec['circ_key']['target']} "
          f"qry {rec['circ_qry']['target']} both {rec['circ_both']['target']} "
          f"route {rec['route']['target']}")
    print(f"pred_a both {pa} | pred_b query {pb} | pred_c selective {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
