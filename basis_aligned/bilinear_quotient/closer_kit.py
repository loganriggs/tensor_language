# closer_kit: THE UNION-KIT MDL TEST (§1356). 13.8 = THE CLOSER (brackets + quotes from
# one seat). Does ONE kit serve BOTH capabilities? Arms: route | kit_bracket (depth>0
# gate) | kit_quote (odd-parity gate) | kit_union (either) — each = gate on a02 + 13.8,
# route grain, scored on BOTH target families (bracket-closes and quote-closes).
#
# Registered predictions:
#   pred_a THE QUOTE KIT CARRIES: kit_quote >= 0.60 recovery at quote-close targets.
#   pred_b ONE KIT, TWO CAPABILITIES: kit_union within 0.03 of each dedicated kit on its
#          own family (single description, both capabilities — the function frame cashed).
#   pred_c SELECTIVITY: every kit arm's elsewhere recovery within 0.05 of route's.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'closer_kit_results.json'
NMEAN = 24; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
A02L = (0, 1, 2)
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
    close_t = set(); open_t = set(); q_t = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if ')' in d:
            close_t.add(tok)
        if '(' in d:
            open_t.add(tok)
        if '"' in d:
            q_t.add(tok)
    close_ids = torch.tensor(sorted(close_t)); open_ids = torch.tensor(sorted(open_t))
    q_ids = torch.tensor(sorted(q_t))

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
    isq = torch.isin(toks, q_ids)
    QPAR = (isq.long().cumsum(1) % 2) == 1
    TB = torch.isin(tgt_all, close_ids) & QSTATE
    TQ = torch.isin(tgt_all, q_ids) & QPAR
    TB[:, :64] = False; TQ[:, :64] = False
    ELSE = ~TB & ~TQ; ELSE[:, :64] = False
    print(f"bracket targets {int(TB.sum())} | quote targets {int(TQ.sum())}", flush=True)

    def ce_cond(arm, gate_kind=None):
        sb = sq = se = 0.0; nb = nq = ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8]
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].to(DEV).contiguous()
            qs, wp = bracket_masks(idx, open_ids, close_ids)
            iq = torch.isin(idx, q_ids)
            qp = (iq.long().cumsum(1) % 2) == 1
            gm = {'bracket': qs, 'quote': qp, 'union': qs | qp, None: qs}[gate_kind]
            lo = fwd_arm(idx.to(DEV), arm, vmeans, ymeans, gm.to(DEV)).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mb = TB[i:i + 8].to(DEV); mq = TQ[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            sb += float(ce[mb].sum()); nb += int(mb.sum())
            sq += float(ce[mq].sum()); nq += int(mq.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
        return sb / max(nb, 1), sq / max(nq, 1), se / max(ne, 1)

    res = {}
    for arm, gk in (('full', None), ('ymean', None), ('route', None),
                    ('kit_bracket', 'bracket'), ('kit_quote', 'quote'),
                    ('kit_union', 'union')):
        real_arm = 'circ_gate' if arm.startswith('kit') else arm
        b, q, e = ce_cond(real_arm, gk)
        res[arm] = {'bracket': round(b, 4), 'quote': round(q, 4), 'else': round(e, 4)}
        print(f"{arm}: bracket {b:.4f} | quote {q:.4f} | else {e:.4f}", flush=True)

    gb = res['ymean']['bracket'] - res['full']['bracket']
    gq = res['ymean']['quote'] - res['full']['quote']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'bracket': round((res['ymean']['bracket'] - res[a]['bracket']) / max(gb, 1e-6), 4),
               'quote': round((res['ymean']['quote'] - res[a]['quote']) / max(gq, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    re_ = rec['route']['else']
    pa = rec['kit_quote']['quote'] >= 0.60
    pb = (abs(rec['kit_union']['bracket'] - rec['kit_bracket']['bracket']) <= 0.03 and
          abs(rec['kit_union']['quote'] - rec['kit_quote']['quote']) <= 0.03)
    pc = all(abs(rec[a]['else'] - re_) <= 0.05
             for a in ('kit_bracket', 'kit_quote', 'kit_union'))
    out = {'n_bracket': int(TB.sum()), 'n_quote': int(TQ.sum()), 'n_rows': NR,
           'ce': res, 'recovery': rec,
           'pred_a_quote_kit_carries': bool(pa), 'pred_b_one_kit_two_caps': bool(pb),
           'pred_c_selective': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nrec bracket: dedicated {rec['kit_bracket']['bracket']} union {rec['kit_union']['bracket']}")
    print(f"rec quote:   dedicated {rec['kit_quote']['quote']} union {rec['kit_union']['quote']}")
    print(f"pred_a quote-kit {pa} | pred_b one-kit {pb} | pred_c selective {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
