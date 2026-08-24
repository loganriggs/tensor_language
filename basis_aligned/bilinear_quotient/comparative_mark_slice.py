# comparative_mark_slice: THE §1329 REFINEMENT — slim the comparative circuit's kept
# description from "the whole a02 band" to a capability-specific slice, and price it.
#
# §1329 measured: route+a02+8.1 reaches 0.697 target recovery, but a02 is the UNIVERSAL
# class-marker (elsewhere recovery 0.679 ~ target's), so goal-1 extraction was not
# capability-specific. The two refinement arms registered there:
#   circ_pos  a02 heads live ONLY AT COMPARATIVE POSITIONS (per-query-position gate on the
#             head output; elsewhere v1-routed like the crowd). Cost in bits: ~zero — the
#             gate is a token-property of the input stream.
#   circ_dir  rank-1: a02 v1-routed everywhere, PLUS at comparative positions the
#             projection of the live-minus-route band output onto ONE fixed direction
#             (the mark direction, estimated as the mean live-route output delta at
#             comparative positions on held-out-from-eval fit rows). 8.1 live in both.
#
# ASSUMPTION REGISTERED, NOT ASKED: "comparative positions" = positions whose input token
# is a comparative (the §1303 lexicon). The mark is taken to be written AT the comparative
# token; if the band writes it on following positions instead, circ_pos underperforms and
# that miss is itself informative (mark-position localization).
#
# Anchors (full/ymean/route/circ_band) recomputed fresh — same instrument, fresh rows.
#
# Registered predictions:
#   pred_a POSITIONAL GATING SUFFICES: circ_pos target recovery >= 0.60 AND its elsewhere
#          recovery <= route's + 0.05 — the capability-specific extraction at ~zero bits.
#   pred_b THE MARK IS LOW-RANK: circ_dir target recovery >= 0.55 (rank-1 carries most of
#          what position-gated full-rank carries).
#   pred_c SELECTIVITY OF BOTH SLICES: each new arm's elsewhere recovery within 0.05 of
#          route's (the slice buys the capability and nothing else).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'comparative_mark_slice_results.json'
NMEAN = 24; NDIR = 96; NR = 1920
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
A02L = (0, 1, 2)
KEEP81 = {(8, 1)}


@torch.no_grad()
def fwd_arm(idx, arm, vmeans, ymeans, compmask, markdir):
    """arm in full|ymean|route|circ_band|circ_pos|circ_dir. Route grain everywhere
    (§1314/16): removed heads keep lambda*v1, fresh values -> per-head mean; patterns
    live. a02 treatment varies by arm; 8.1 live in every circ_* arm."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    cm = compmask.unsqueeze(-1) if compmask is not None else None   # (B,T,1) float
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
            # routed values for every head
            vr = v.clone()
            for h in range(9):
                if not (arm != 'route' and (L, h) in KEEP81):
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
            vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
            if L in A02L and arm in ('circ_band', 'circ_pos', 'circ_dir'):
                vv_live = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
                y_live = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv_live.dtype), vv_live)
                if arm == 'circ_band':
                    y = y_live
                elif arm == 'circ_pos':
                    y = torch.where(cm.unsqueeze(-1).bool(), y_live, y)
                else:                       # circ_dir: rank-1 mark at comp positions
                    dflat = (y_live - y).reshape(B, T, D)
                    md = markdir[L]         # (D,)
                    coef = (dflat * md).sum(-1, keepdim=True)
                    add = (coef * md).view(B, T, 9, 128)
                    y = y + add * cm.unsqueeze(-1).to(y.dtype)
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

    ROWS = cl.fineweb_rows(NMEAN + NDIR + NR)[:, :T + 1].contiguous()
    MEANR = ROWS[:NMEAN]; DIRR = ROWS[NMEAN:NMEAN + NDIR]; EVR = ROWS[NMEAN + NDIR:]

    # ---- per-head fresh-v and y means from MEANR (identical to §1329's estimator)
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

    # ---- mark direction per a02 layer: mean (y_live - y_route) at comparative positions
    # on DIRR rows (disjoint from eval), unit-normalized. Rank-1 per layer.
    dsum = {L: torch.zeros(D, device=DEV) for L in A02L}
    dcnt = {L: 0.0 for L in A02L}
    for i in range(0, NDIR, 4):
        bb = DIRR[i:i + 4]
        idx = bb[:, :-1].to(DEV).contiguous()
        cmask = torch.isin(idx.cpu(), comp_t).to(DEV)
        if not cmask.any():
            continue
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
            if L in A02L:
                vr = v.clone()
                for h in range(9):
                    vr[:, :, h] = vmeans[L][h].to(vr.dtype)
                vvr = (1 - at.lamb) * vr + at.lamb * v1.view_as(vr)
                yr = torch.einsum('bhqk,bkhd->bqhd', pat.to(vvr.dtype), vvr)
                delta = (y - yr).reshape(B, T, D)
                sel = delta[cmask]
                if sel.numel():
                    dsum[L] += sel.float().sum(0)
                    dcnt[L] += sel.shape[0]
            x = xm + at.c_proj(y.reshape(B, T, D))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    markdir = {L: (dsum[L] / max(dcnt[L], 1.0)) for L in A02L}
    for L in A02L:
        markdir[L] = markdir[L] / markdir[L].norm().clamp_min(1e-8)
    print(f"mark-dir samples per layer: {[int(dcnt[L]) for L in A02L]}", flush=True)

    # ---- targets on EVR (§1303 construction)
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

    def ce_cond(arm):
        st = se = 0.0; nt = ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            cmask = torch.isin(idx.cpu(), comp_t).to(DEV).float()
            lo = fwd_arm(idx, arm, vmeans, ymeans, cmask, markdir).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
            st += float(ce[mt].sum()); nt += int(mt.sum())
            se += float(ce[me].sum()); ne += int(me.sum())
        return st / max(nt, 1), se / max(ne, 1)

    res = {}
    for arm in ('full', 'ymean', 'route', 'circ_band', 'circ_pos', 'circ_dir'):
        tce, ece = ce_cond(arm)
        res[arm] = {'target': round(tce, 4), 'else': round(ece, 4)}
        print(f"{arm}: target {tce:.4f} | else {ece:.4f}", flush=True)

    gt = res['ymean']['target'] - res['full']['target']
    ge = res['ymean']['else'] - res['full']['else']
    rec = {a: {'target': round((res['ymean']['target'] - res[a]['target']) / max(gt, 1e-6), 4),
               'else': round((res['ymean']['else'] - res[a]['else']) / max(ge, 1e-6), 4)}
           for a in res if a != 'ymean'}
    re_ = rec['route']['else']
    pa = rec['circ_pos']['target'] >= 0.60 and rec['circ_pos']['else'] <= re_ + 0.05
    pb = rec['circ_dir']['target'] >= 0.55
    pc = (abs(rec['circ_pos']['else'] - re_) <= 0.05 and
          abs(rec['circ_dir']['else'] - re_) <= 0.05)
    out = {'n_targets': ntar, 'n_rows': NR, 'ce': res, 'recovery': rec,
           'gap_target': round(gt, 4), 'gap_else': round(ge, 4),
           'mark_dir_samples': {str(L): int(dcnt[L]) for L in A02L},
           'pred_a_posgate_suffices': bool(pa), 'pred_b_mark_rank1': bool(pb),
           'pred_c_both_selective': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nrec target: band {rec['circ_band']['target']} pos {rec['circ_pos']['target']} "
          f"dir {rec['circ_dir']['target']} route {rec['route']['target']}")
    print(f"rec else:   band {rec['circ_band']['else']} pos {rec['circ_pos']['else']} "
          f"dir {rec['circ_dir']['else']} route {re_}")
    print(f"pred_a posgate {pa} | pred_b rank1 {pb} | pred_c selective {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
