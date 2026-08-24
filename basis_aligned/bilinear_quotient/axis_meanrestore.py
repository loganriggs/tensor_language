# axis_meanrestore: the sharper position-binding null for §1250. Full per-position restore
# recovered 95.3%; the shuffled null recovered 40% — weak because repeat-row axis values are
# near-exchangeable. Sharper decomposition: restore with each row's MEAN axis value (one
# scalar per row, no positional pattern). What mean-restore recovers = the "matches are
# happening" LEVEL; the gap to full restore = the per-position PATTERN's contribution.
#
# Registered predictions:
#   pred_a THE PATTERN MATTERS: mean-restore recovers <= 60% of full restore's recovery.
#   pred_b LEVEL EXPLAINS THE SHUFFLE: |mean-restore - shuffled-restore| <= 0.15 x mask
#          damage (the §1250 shuffled recovery was the level, not leakage).
#   pred_c ANCHORS: full restore >= 90% recovery (replicates §1250); sanity base = true.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'axis_meanrestore_results.json'
NFIT = 12; NR = 24; QSTART = 128; QFIT = 160; WIN = 64
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
MASK_W = None
FULL = None


@torch.no_grad()
def fit_axis(rows):
    DL = []
    for i in range(0, NFIT, 4):
        idx = rows[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        B = idx.shape[0]
        wsum = torch.zeros(B, T, D, device=DEV)
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
            tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
            pat = pat.masked_fill(~tril, 0.0)
            v = at.c_v(xin).view(B, T, 9, 128)
            if v1 is None:
                v1 = v
            vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
            y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
            if L in MATCHERS:
                for h in MATCHERS[L]:
                    yh = torch.zeros_like(y)
                    yh[:, :, h, :] = y[:, :, h, :]
                    wsum = wsum + at.c_proj(yh.reshape(B, T, D)).float()
            x = xm + at.c_proj(y.reshape(B, T, D))
            if L == 3:
                full = blk.mlp(F.rms_norm(x, (D,)))
                blind = blk.mlp(F.rms_norm(x - wsum.to(x.dtype), (D,)))
                DL.append((full - blind).float()[:, QFIT:].reshape(-1, D).cpu())
                break
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    DL = torch.cat(DL)
    Dc = DL - DL.mean(0)
    _, _, V = torch.pca_lowrank(Dc, q=4)
    return V[:, 0].to(DEV)




@torch.no_grad()
def forward_gen(idx, mask_matchers, d, restore_vals):
    """restore_vals: dict L -> (B,T) target axis components at block-L entry, or None."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    caps = {}
    for L, blk in enumerate(m.transformer.h):
        if d is not None and 4 <= L <= 8:
            if restore_vals is None:
                caps[L] = (x.float() * d).sum(-1)                     # capture
            else:
                cur = (x.float() * d).sum(-1)
                x = x + ((restore_vals[L] - cur).unsqueeze(-1) * d).to(x.dtype)
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
        if mask_matchers and L in MATCHERS:
            msk = torch.stack([MASK_W if h in MATCHERS[L] else FULL for h in range(9)], 0)
            pat = pat.masked_fill(~msk.unsqueeze(0), 0.0)
        else:
            pat = pat.masked_fill(~FULL, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    return logits, caps


@torch.no_grad()
def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    FIT = cl.fineweb_rows(NFIT)[:, :T + 1].contiguous().clone()
    FIT[:, 128:256] = FIT[:, 0:128]
    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    REP[:, 128:256] = REP[:, 0:128]
    d = fit_axis(FIT)
    qp = torch.arange(QSTART, T, device=DEV)
    g = torch.Generator().manual_seed(12)
    perm = torch.randperm(T, generator=g).to(DEV)

    ce = {c: 0.0 for c in ('base', 'mask', 'restore', 'restore_shuf', 'restore_mean')}
    n = 0
    for i in range(0, NR, 4):
        bb = REP[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo_base, caps = forward_gen(idx, False, d, None)
        caps_shuf = {L: v[:, perm] for L, v in caps.items()}
        caps_mean = {L: v.mean(dim=1, keepdim=True).expand_as(v).contiguous() for L, v in caps.items()}
        conds = {'base': (lo_base, None),
                 'mask': forward_gen(idx, True, None, None)[0:1] + (None,),
                 'restore': forward_gen(idx, True, d, caps)[0:1] + (None,),
                 'restore_shuf': forward_gen(idx, True, d, caps_shuf)[0:1] + (None,),
                 'restore_mean': forward_gen(idx, True, d, caps_mean)[0:1] + (None,)}
        for cname, tup in conds.items():
            lo = tup[0] if not isinstance(tup[0], tuple) else tup[0][0]
            lo = lo.float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    dmg = CE['mask'] - CE['base']
    rec = (CE['mask'] - CE['restore']) / max(dmg, 1e-6)
    rec_s = (CE['mask'] - CE['restore_shuf']) / max(dmg, 1e-6)
    rec_m = (CE['mask'] - CE['restore_mean']) / max(dmg, 1e-6)
    out = {'n_rows': NR, 'ce': CE, 'mask_damage': round(dmg, 4),
           'recovery': {'full': round(rec, 4), 'shuffled': round(rec_s, 4), 'mean': round(rec_m, 4)},
           'pred_a_pattern_matters': bool(rec_m <= 0.6 * rec),
           'pred_b_level_explains_shuffle': bool(abs(CE['restore_mean'] - CE['restore_shuf']) <= 0.15 * dmg),
           'pred_c_anchors': bool(rec >= 0.90),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | recovery full {rec:.3f} shuf {rec_s:.3f} mean {rec_m:.3f}")
    print(f"pred_a pattern {out['pred_a_pattern_matters']} | pred_b level {out['pred_b_level_explains_shuffle']} | pred_c anchor {out['pred_c_anchors']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
