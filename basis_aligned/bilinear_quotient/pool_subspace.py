# pool_subspace: WHERE does the far-pool's 0.176 travel? (§1263 opener.) The clean-minus-
# masked stream delta at block-8 entry is only 6.5% content-basis; characterize the FULL
# delta and causally certify its own top subspace: (1) PCA the per-position deltas (rank
# profile, overlap with known bases); (2) REMOVE the delta's top-16 directions from the
# CLEAN stream at L5-9 entries and price it — if that reproduces a chunk of the all-mask
# damage, the pool's carrier subspace is found and named.
#
# Registered predictions:
#   pred_a MODERATELY LOW-RANK: top-16 PCs >= 50% of delta variance.
#   pred_b CAUSAL: removing the top-16 delta directions from the clean model costs >= 30%
#          of the all-mask damage (0.176).
#   pred_c NULL: random-16 removal <= 10% of it.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pool_subspace_results.json'
NR = 24; QSTART = 128; WIN = 64; K = 64
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
def stream_at8(idx, masked):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        if L == 8:
            return x.detach().float()
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
        pat = pat.masked_fill(~(MASK_W if masked else FULL), 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return None



@torch.no_grad()
def forward_rm16(idx, V16):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for L, blk in enumerate(m.transformer.h):
        if V16 is not None and 5 <= L <= 9:
            x = x - (x.float() @ V16 @ V16.T).to(x.dtype)
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_rm(rows, V16):
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_rm16(idx, V16).float()
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
    FIT = cl.fineweb_rows(12)[:, :T + 1].contiguous()
    EV = cl.fineweb_rows(NR)[:, :T + 1].contiguous()

    # 1) characterize the delta on FIT rows
    DL = []
    for i in range(0, 12, 4):
        idx = FIT[i:i + 4, :-1].to(DEV).contiguous()
        xc = stream_at8(idx, False); xm = stream_at8(idx, True)
        DL.append((xc - xm)[:, QSTART:].reshape(-1, D).cpu())
    DL = torch.cat(DL)
    Dc = DL - DL.mean(0)
    U, S, Vt = torch.pca_lowrank(Dc, q=64)
    var = S ** 2
    share16 = float(var[:16].sum() / (Dc ** 2).sum())
    V16 = Vt[:, :16].to(DEV).float()
    cont_overlap = float((V16.T @ Uc).norm() ** 2 / 16)
    print(f"top16 var {share16:.3f} | content-basis overlap {cont_overlap:.3f}", flush=True)

    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = EV[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = fwd(idx).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    base = tot / n
    g = torch.Generator().manual_seed(21)
    R = torch.randn(D, 16, generator=g)
    R, _ = torch.linalg.qr(R); R = R.to(DEV).float()
    CE = {'base': round(base, 4), 'rm_delta16': round(ce_rm(EV, V16), 4),
          'rm_rand16': round(ce_rm(EV, R), 4)}
    cost_d = CE['rm_delta16'] - base; cost_r = CE['rm_rand16'] - base
    out = {'n_rows': NR, 'top16_var_share': round(share16, 4),
           'content_overlap': round(cont_overlap, 4), 'ce': CE,
           'cost': {'delta16': round(cost_d, 4), 'rand16': round(cost_r, 4)},
           'allmask_ref': 0.1759,
           'pred_a_lowrank': bool(share16 >= 0.50),
           'pred_b_causal': bool(cost_d >= 0.30 * 0.1759),
           'pred_c_null': bool(cost_r <= 0.10 * 0.1759),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | delta16 cost {cost_d:.4f} vs rand {cost_r:.4f} (all-mask 0.176)")
    print(f"pred_a lowrank {out['pred_a_lowrank']} | pred_b causal {out['pred_b_causal']} | pred_c null {out['pred_c_null']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
