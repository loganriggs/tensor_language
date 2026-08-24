# pool_names: WHAT are the 16 far-channel directions? (§1266 closer.) Overlap the certified
# V16 basis with every named object on the books, and price the top few directions singly
# with the §1266 surgical instrument.
#
# Named objects compared: (1) the sink/baseline constant direction (mean residual L6-11
# idiom: mean of clean stream at block-8 entry over positions — §1089's baseline); (2) the
# match-verdict axis (§1249 fit); (3) the content basis U_c (§1150 idiom, K=64); (4) the
# top-16 principal directions of the clean stream itself (the "working coordinates" —
# massive/gain dims live here).
#
# Registered predictions:
#   pred_a BASELINE IS IN THE CHANNEL: |V16-span projection of the baseline direction|
#          >= 0.5 (far pooling includes baseline/constant maintenance — the sink theme).
#   pred_b VERDICT AXIS IS NOT (prose; copy machinery dormant): span projection <= 0.2.
#   pred_c CONCENTRATION AT DIRECTION GRAIN: the top single delta direction's far-removal
#          cost >= 3x the mean of directions 13-16 (the channel has internal structure,
#          not uniform spread).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pool_names_results.json'
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
def forward_far_rm(idx, V16, mode):
    """mode: 'none' (base) / 'full' (remove far component) / 'proj' (remove V16-proj of far)."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
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
        pat = pat.masked_fill(~FULL, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        yo = at.c_proj(y)
        if mode != 'none' and 5 <= L <= 9:
            pat_far = pat.masked_fill(MASK_W, 0.0)            # far = beyond WIN, pos-0 excluded
            yf = at.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat_far.to(vv.dtype), vv).reshape(B, T, D))
            if mode == 'full':
                yo = yo - yf
            else:
                yo = yo - (yf.float() @ V16 @ V16.T).to(yo.dtype)
        x = xm + yo
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_far(rows, V16, mode):
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_far_rm(idx, V16, mode).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    return tot / n



@torch.no_grad()
def ce_far_dirs(rows, dirs):
    """Cost of removing the far-component projection onto EACH single direction (list)."""
    out = []
    for dvec in dirs:
        V1 = dvec.view(-1, 1)
        c = ce_far(rows, V1, 'proj')
        out.append(c)
    return out


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

    DL = []; CLEAN = []
    for i in range(0, 12, 4):
        idx = FIT[i:i + 4, :-1].to(DEV).contiguous()
        xc = stream_at8(idx, False); xm = stream_at8(idx, True)
        DL.append((xc - xm)[:, QSTART:].reshape(-1, D).cpu())
        CLEAN.append(xc[:, QSTART:].reshape(-1, D).cpu())
    DL = torch.cat(DL); CLEAN = torch.cat(CLEAN)
    Dc = DL - DL.mean(0)
    _, S, Vt = torch.pca_lowrank(Dc, q=32)
    V16 = Vt[:, :16].to(DEV).float()

    baseline = CLEAN.mean(0); baseline = (baseline / baseline.norm()).to(DEV)
    Cc = CLEAN - CLEAN.mean(0)
    _, _, Wt = torch.pca_lowrank(Cc, q=16)
    W16 = Wt[:, :16].to(DEV).float()

    sys.path.insert(0, PT)
    import importlib
    ra = importlib.import_module('reencode_axis')
    FITR = cl.fineweb_rows(12)[:, :T + 1].contiguous().clone()
    FITR[:, 128:256] = FITR[:, 0:128]
    axis = ra.fit_axis(FITR)

    def span_proj(vec):
        return float((V16.T @ vec).norm())

    overlaps = {'baseline': round(span_proj(baseline), 4),
                'verdict_axis': round(span_proj(axis), 4),
                'content_basis_energy': round(float((V16.T @ Uc).norm() ** 2 / 16), 4),
                'clean_stream_pc16_energy': round(float((V16.T @ W16).norm() ** 2 / 16), 4)}
    print(f"overlaps {overlaps}", flush=True)

    dirs = [V16[:, j].contiguous() for j in range(16)]
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = EV[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = fwd(idx).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    base = tot / n
    costs = []
    for j in (0, 1, 2, 3, 12, 13, 14, 15):
        c = ce_far(EV, dirs[j].view(-1, 1), 'proj') - base
        costs.append((j, round(c, 5)))
        print(f"dir {j}: far-removal cost {c:.5f}", flush=True)
    top = costs[0][1]
    tailmean = sum(c for j, c in costs if j >= 12) / 4
    out = {'n_rows': NR, 'overlaps': overlaps, 'single_dir_costs': dict((str(j), c) for j, c in costs),
           'pred_a_baseline_in': bool(overlaps['baseline'] >= 0.5),
           'pred_b_verdict_out': bool(overlaps['verdict_axis'] <= 0.2),
           'pred_c_concentrated': bool(top >= 3 * max(tailmean, 1e-6)),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a baseline {out['pred_a_baseline_in']} | pred_b verdict {out['pred_b_verdict_out']} | pred_c conc {out['pred_c_concentrated']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
