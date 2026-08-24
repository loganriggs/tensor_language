# far_bag_standin: can the POOLING CROWD's product be COMPUTED? The §1204-61 reduction
# replaced the copy front end with weights-computable operations. The content pool is the
# other long-range consumer (prose read budget 0.176 @W64; no compact ensemble §1222;
# synergistic §1223). But its PRODUCT has a candidate closed form: §1076 — content =
# pooled bag of block-0's static per-token c_v codes. Stand-in: read-mask ALL 162 heads
# @W64 (full 0.176 damage) and restore, at each carrying-band entry (L5-9), the CONTENT
# PROJECTION of a computed far bag: bag_t = sum_{k <= t-64} exp(-(t-k)/64) * v1_k,
# projected onto the standard content basis U_c (§1150 idiom), scaled by one global
# fitted scalar (grid on FIT rows).
#
# Registered predictions (prose, scored t>=128, eval rows disjoint from fit):
#   pred_a THE BAG COMPUTES: recovery >= 25% of the all-mask damage.
#   pred_b IT IS THE RIGHT BAG: wrong-text bag (same construction from OTHER rows)
#          recovers <= half of the true bag's recovery.
#   pred_c IT IS THE FAR INFORMATION: a within-window-only bag (k in (t-64, t]) recovers
#          <= half of the far bag's recovery (near information is already visible to the
#          masked model).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'far_bag_standin_results.json'
NFIT = 12; NR = 24; QSTART = 128; WIN = 64; K = 64
NSEQ = 96; SEQ = 256; REF = [8, 10, 12]
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
MASK_W = None
FULL = None


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def content_basis(blocks):
    """standard idiom: top-K PCA of pooled L8-12 mlp-input deviation."""
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); V = int(m.lm_head.weight.shape[0]); devsum = None
    for L in REF:
        X = torch.cat(cap[L], 0); xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
        xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; cap[L] = []; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    return Vt[:K].T.contiguous()




@torch.no_grad()
def far_bag(idx, far=True):
    """Recency-weighted bag of v1 codes. far=True: k <= t-WIN; far=False: t-WIN < k <= t."""
    B = idx.shape[0]
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    v1 = m.transformer.h[0].attn.c_v(x)                       # (B,T,D)
    ar = torch.arange(T, device=DEV)
    dist = (ar[:, None] - ar[None, :]).float()
    wmat = torch.exp(-dist / 64.0)
    valid = (dist >= WIN) if far else ((dist > 0) & (dist < WIN))
    wmat = (wmat * valid.float() * torch.tril(torch.ones(T, T, device=DEV))).to(v1.dtype)
    bag = torch.einsum('qk,bkd->bqd', wmat, v1)
    return bag / wmat.sum(-1).clamp_min(1e-6).unsqueeze(0).unsqueeze(-1)


@torch.no_grad()
def forward_masked_bag(idx, Uc, bagproj, scale):
    """All 162 heads read-masked @WIN; optional content-projected bag added at L5-9 entries."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        if bagproj is not None and 5 <= L <= 9:
            x = x + scale * bagproj.to(x.dtype)
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
        pat = pat.masked_fill(~MASK_W, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_of(rows, Uc, mode, scale):
    """mode: None (mask only) / 'true' / 'wrong' / 'near'."""
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        if mode is None:
            bp = None
        else:
            if mode == 'true':
                bag = far_bag(idx, far=True)
            elif mode == 'near':
                bag = far_bag(idx, far=False)
            else:
                ridx = torch.roll(idx, 1, dims=0)
                bag = far_bag(ridx, far=True)
            bp = (bag.float() @ Uc) @ Uc.T
        lo = forward_masked_bag(idx, Uc, bp, scale).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    return tot / n


@torch.no_grad()
def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    Uc = content_basis(blocks)
    FIT = cl.fineweb_rows(NFIT)[:, :T + 1].contiguous()
    EV = cl.fineweb_rows(NR)[:, :T + 1].contiguous()

    qp = torch.arange(QSTART, T, device=DEV)
    # base on eval
    tot = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = EV[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = fwd(idx).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    base = tot / n
    mask_ce = ce_of(EV, Uc, None, 0.0)
    print(f"base {base:.4f} | mask {mask_ce:.4f} | damage {mask_ce - base:.4f}", flush=True)

    best_s, best_ce = 0.0, 1e9
    for s in (0.1, 0.25, 0.5, 1.0, 2.0):
        cef = ce_of(FIT, Uc, 'true', s)
        print(f"scale {s}: fit CE {cef:.4f}", flush=True)
        if cef < best_ce:
            best_ce, best_s = cef, s
    print(f"chosen scale {best_s}", flush=True)

    CE = {'base': round(base, 4), 'mask': round(mask_ce, 4),
          'bag_true': round(ce_of(EV, Uc, 'true', best_s), 4),
          'bag_wrong': round(ce_of(EV, Uc, 'wrong', best_s), 4),
          'bag_near': round(ce_of(EV, Uc, 'near', best_s), 4)}
    dmg = CE['mask'] - CE['base']
    rec = {k: round((CE['mask'] - CE[k]) / max(dmg, 1e-6), 4) for k in ('bag_true', 'bag_wrong', 'bag_near')}
    out = {'n_rows': NR, 'scale': best_s, 'ce': CE, 'damage': round(dmg, 4), 'recovery': rec,
           'pred_a_bag_computes': bool(rec['bag_true'] >= 0.25),
           'pred_b_right_bag': bool(rec['bag_wrong'] <= 0.5 * max(rec['bag_true'], 1e-6)),
           'pred_c_far_info': bool(rec['bag_near'] <= 0.5 * max(rec['bag_true'], 1e-6)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | recovery {rec}")
    print(f"pred_a computes {out['pred_a_bag_computes']} | pred_b right {out['pred_b_right_bag']} | pred_c far {out['pred_c_far_info']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
