# close_bracket_gates: THE TEMPLATE ON THE CLOSE-BRACKET CIRCUIT (13.8, §1340-41). Gates:
#   key gate   open-paren positions (where the "a bracket is open" mark is seeded)
#   qry gate   inside-bracket positions (depth > 0 — where the close must be predicted)
# Specialist: 13.8 live in every circ arm. Annotator band assumption: a02 (front writes
# class marks, §1286-88) — registered, not established for brackets; a band-arm miss
# localizes the bracket annotator elsewhere (the §1334 path). a14's compound-close assist
# (§1341) is deliberately NOT in these arms — this rung prices the core close circuit;
# the a14 extension is the next rung.
#
# Registered predictions:
#   pred_a THE EXTRACTION CARRIES: circ_band (a02 + 13.8) target recovery >= 0.60.
#   pred_b QUERY-SIDE DOMINATES (third circuit in a row): qry gate beats key gate by
#          >= 0.05 target recovery.
#   pred_c 13.8 IS ANNOTATOR-CONDITIONAL like 8.1: route+13.8 alone <= route + 0.08
#          (the owner without its mark is near-useless).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'close_bracket_gates_results.json'
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
    for arm, gk in (('full', None), ('ymean', None), ('route', None), ('circ_solo', None),
                    ('circ_band', None), ('circ_key', 'key'), ('circ_qry', 'qry'),
                    ('circ_both', 'both')):
        real_arm = arm
        if arm.startswith('circ_') and arm not in ('circ_band', 'circ_solo'):
            real_arm = 'circ_gate'
        elif arm == 'circ_solo':
            real_arm = 'route_solo'
        tce, ece = ce_cond(real_arm, gk)
        res[arm] = {'target': round(tce, 4), 'else': round(ece, 4)}
        print(f"{arm}: target {tce:.4f} | else {ece:.4f}", flush=True)

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'target': round((res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    re_ = rec['route']['else']
    pa = rec['circ_band']['target'] >= 0.60
    pb = rec['circ_qry']['target'] >= rec['circ_key']['target'] + 0.05
    pc = rec['circ_solo']['target'] <= rec['route']['target'] + 0.08
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': res, 'recovery': rec,
           'gap_target': round(gt, 4), 'gap_else': round(ge, 4),
           'pred_a_extraction_carries': bool(pa), 'pred_b_query_dominates': bool(pb),
           'pred_c_owner_conditional': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nrec target: band {rec['circ_band']['target']} key {rec['circ_key']['target']} "
          f"qry {rec['circ_qry']['target']} both {rec['circ_both']['target']} "
          f"route {rec['route']['target']}")
    print(f"pred_a carries {pa} | pred_b query {pb} | pred_c conditional {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
