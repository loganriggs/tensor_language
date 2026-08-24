# question_mid_heads: THE LAST LOCALIZATION RUNG OF THE QUESTION THREAD. §1335 named the
# question annotator as L0-5 attention (clause-gated a05 = 0.714) with the mid half (L3-5)
# carrying the service the front lacks. This run asks whether that mid service is
# head-concentrated (a hidden 8.1-style owner) or crowd-carried (the §1093/§1099 prior for
# the mid band).
#
# Stage 1 (per-layer): clause-gated a02 + ONE of L3/L4/L5 fully live (+10.5 + route),
#   against the a02 and a05 anchors. The layer increment = arm - a02.
# Stage 2 (per-head, winner layer): clause-gated a02 + ONE head of the winning layer.
#   Head increment = arm - a02. All stage-2 arms share the stage-1 batch means.
#
# Registered predictions:
#   pred_a PER-LAYER LOCALIZATION: one of L3/L4/L5 carries >= 60% of the a05-over-a02
#          increment.
#   pred_b DIFFUSE BET (the program's collective-pooling prior, §1093 — registered AGAINST
#          a §1304-style single owner): within the winning layer, NO single head carries
#          >= 50% of that layer's increment.
#   pred_c SELECTIVITY THROUGHOUT: every arm's elsewhere recovery within 0.05 of route's.
# Diagnostic: the three layer increments summed vs the joint a05-a02 increment
# (§1335 saw sub-additivity at band grain; expect it again at layer grain).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'question_mid_heads_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
KEEPQ = {(10, 5)}
WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
      'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
GATE = {'heads': set()}          # {(L,h)} live inside the clause gate


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, gatemask):
    """arm full|ymean|route|circ_gate. circ_gate: heads in GATE['heads'] live inside
    gatemask positions; 10.5 live everywhere; all else v1-routed (§1314/16 grain)."""
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


A02 = {(L, h) for L in (0, 1, 2) for h in range(9)}


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

    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    QSTATE = qstate_masks(toks, se_t, wh_t)
    TARGET = torch.isin(tgt_all, qm_t) & QSTATE
    TARGET[:, :64] = False
    ELSE = ~TARGET; ELSE[:, :64] = False
    print(f"targets {int(TARGET.sum())}", flush=True)

    def ce_cond(arm):
        st = se = 0.0; nt = ne = 0
        for i in range(0, NR, 8):
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

    res = {}
    def run(name, heads=None, arm=None):
        if heads is not None:
            GATE['heads'] = heads; arm_ = 'circ_gate'
        else:
            arm_ = arm
        tce, ece = ce_cond(arm_)
        res[name] = {'target': round(tce, 4), 'else': round(ece, 4)}
        print(f"{name}: target {tce:.4f} | else {ece:.4f}", flush=True)
        json.dump({'res': res, 'partial': True}, open(OUT, 'w'), indent=1)

    run('full', arm='full'); run('ymean', arm='ymean'); run('route', arm='route')
    run('a02', heads=set(A02))
    run('a05', heads=A02 | {(L, h) for L in (3, 4, 5) for h in range(9)})
    for L in (3, 4, 5):
        run(f'a02+L{L}', heads=A02 | {(L, h) for h in range(9)})

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    def rt(n): return (res['ymean']['target'] - res[n]['target']) / max(gt, 1e-6)
    def re(n): return (res['ymean']['else'] - res[n]['else']) / max(ge, 1e-6)
    band_inc = rt('a05') - rt('a02')
    layer_inc = {L: rt(f'a02+L{L}') - rt('a02') for L in (3, 4, 5)}
    winner = max(layer_inc, key=layer_inc.get)
    print(f"layer increments {layer_inc} | winner L{winner} | band inc {band_inc:.4f}",
          flush=True)

    # stage 2: single heads of the winner layer
    for h in range(9):
        run(f'L{winner}.{h}', heads=A02 | {(winner, h)})
    head_inc = {h: rt(f'L{winner}.{h}') - rt('a02') for h in range(9)}
    top_h = max(head_inc, key=head_inc.get)

    pa = max(layer_inc.values()) >= 0.60 * max(band_inc, 1e-6)
    pb = head_inc[top_h] < 0.50 * max(layer_inc[winner], 1e-6)
    arms = [a for a in res if a not in ('full', 'ymean')]
    pc = all(abs(re(a) - re('route')) <= 0.05 for a in arms if a != 'route')
    rec = {a: {'target': round(rt(a), 4), 'else': round(re(a), 4)} for a in arms}
    out = {'n_targets': int(TARGET.sum()), 'n_rows': NR, 'ce': res, 'recovery': rec,
           'band_increment': round(band_inc, 4),
           'layer_increments': {str(L): round(v, 4) for L, v in layer_inc.items()},
           'winner_layer': winner,
           'head_increments': {str(h): round(v, 4) for h, v in head_inc.items()},
           'top_head': int(top_h),
           'pred_a_layer_localizes': bool(pa), 'pred_b_diffuse': bool(pb),
           'pred_c_selective': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nhead increments {head_inc} | top L{winner}.{top_h}")
    print(f"pred_a layer {pa} | pred_b diffuse {pb} | pred_c selective {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
