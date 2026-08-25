# exclaim_gates: THE TEMPLATE ON THE EXCLAMATION PAIR (17.2+17.3, §1315-20) — the last
# named specialist without an extraction. DESIGN NOTE (caught before running): "!" has no
# input-computable capability window — no trigger lexicon (§1315-20: criterion is
# stream-computed with no lexical seed), and "current sentence" is every position, so the
# key/qry gate arms of the template are ILL-POSED for this circuit. The template's gate
# stage is replaced by a band-conditionality stage:
#   circ_solo    route + pair            (conditionality test)
#   circ_band02  route + pair + a02 live (the front-annotator hypothesis)
#   circ_band05  route + pair + a05 live (deeper annotator, per the question precedent)
#
# Registered predictions:
#   pred_a THE EXTRACTION CARRIES AT A LOWER BAR: a05+pair >= 0.50 (L17 owner — nearly
#          the whole model upstream; route expected to carry more than any prior circuit).
#   pred_b THE PAIR IS CONDITIONAL like every owner so far: route+pair <= route + 0.08.
#   pred_c DEPTH MATTERS for a late owner: band05 >= band02 + 0.05 (question precedent:
#          long-scope state needs mid-stack maintenance).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'exclaim_gates_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
BANDL = {'cur': (0, 1, 2)}
KEEPQ = {(17, 2), (17, 3)}


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
            if L in BANDL['cur'] and arm in ('circ_band', 'circ_gate'):
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


def bracket_masks(toks, send_ids, ex_ids):
    """(SENT, INIT): positions inside the current sentence (since last sentence-end)
    and the sentence-initial positions. Gate-disjointness note: INIT is a subset of
    SENT by construction, so the 'both' arm DEGENERATES to 'qry' — known and accepted
    this run (the both arm is reported but carries no separate information)."""
    is_end = torch.isin(toks, send_ids)
    B2, T2 = toks.shape
    SENT = torch.ones_like(toks, dtype=torch.bool)
    INIT = torch.zeros_like(toks, dtype=torch.bool)
    fresh = torch.ones(B2, dtype=torch.bool)
    for p in range(T2):
        INIT[:, p] = fresh
        fresh = is_end[:, p].clone()
    return SENT, INIT


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    ex_t = set(); send_t = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '!' in d:
            ex_t.add(tok)
        if any(c in d for c in '.!?'):
            send_t.add(tok)
    close_ids = torch.tensor(sorted(ex_t)); open_ids = torch.tensor(sorted(send_t))

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
    TARGET = torch.isin(tgt_all, close_ids)
    TARGET[:, :64] = False
    ELSE = ~TARGET; ELSE[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"targets {ntar}", flush=True)

    def ce_cond(arm, gate_kind=None):
        st = se = 0.0; nt = ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8]
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].to(DEV).contiguous()
            qs, wp = bracket_masks(idx, open_ids, close_ids)
            gm = {'key': wp, 'qry': qs, 'both': wp | qs, None: wp}[gate_kind]
            lo = fwd_arm(idx.to(DEV), arm, vmeans, ymeans, gm.to(DEV)).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            st += float(ce[mt].sum()); nt += int(mt.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
        return st / max(nt, 1), se / max(ne, 1)

    res = {}
    for arm, band in (('full', None), ('ymean', None), ('route', None),
                      ('circ_solo', None), ('circ_band02', (0, 1, 2)),
                      ('circ_band05', (0, 1, 2, 3, 4, 5))):
        real_arm = arm
        if arm == 'circ_solo':
            real_arm = 'route_solo'
        elif arm.startswith('circ_band'):
            real_arm = 'circ_band'
            BANDL['cur'] = band
        tce, ece = ce_cond(real_arm, None)
        res[arm] = {'target': round(tce, 4), 'else': round(ece, 4)}
        print(f"{arm}: target {tce:.4f} | else {ece:.4f}", flush=True)

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'target': round((res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    pa = rec['circ_band05']['target'] >= 0.50
    pb = rec['circ_solo']['target'] <= rec['route']['target'] + 0.08
    pc = rec['circ_band05']['target'] >= rec['circ_band02']['target'] + 0.05
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': res, 'recovery': rec,
           'gap_target': round(gt, 4), 'gap_else': round(ge, 4),
           'pred_a_carries': bool(pa), 'pred_b_pair_conditional': bool(pb),
           'pred_c_depth_matters': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nrec: route {rec['route']['target']} solo {rec['circ_solo']['target']} "
          f"b02 {rec['circ_band02']['target']} b05 {rec['circ_band05']['target']}")
    print(f"pred_a carries {pa} | pred_b conditional {pb} | pred_c depth {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
