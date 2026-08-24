# payload_live: disambiguate §1259's payload failure. Two confounds were left standing:
# (i) the payload needs the LIVE value (layer-8 vv at the source successor — an activation),
# not the static v1 code; (ii) the injection shape was wrong. This run fixes the shape
# (per-head c_proj slices scaled by the §1239 MEASURED in-vivo pattern coefficients:
# 8.4 -> +0.190, 8.3 -> -0.119) and compares LIVE vv (captured from a clean run) against
# STATIC v1 under the identical injection — separating the confounds.
#
# Conditions (repeat rows, quad read-masked, synthesized axis §1258 always on):
#   axis_only (anchor 0.629); +live (vv_L8[succ] payload); +live_flip (signs flipped);
#   +static (v1[succ] payload, same shape/coeffs).
#
# Registered predictions:
#   pred_a LIVE PAYLOAD WORKS: recovery(+live) >= recovery(axis_only) + 0.10.
#   pred_b SIGNS LOAD-BEARING HERE TOO: recovery(+live_flip) <= recovery(axis_only) - 0.05.
#   pred_c THE §1259 CULPRIT WAS STATIC-vs-LIVE: recovery(+static) <= axis_only + 0.05
#          under the corrected shape (if instead +static WORKS here, §1259's failure was
#          the injection shape and the weights-only reduction is back on the table —
#          logged either way, this is the discriminating cell).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'payload_live_results.json'
NFIT = 12; NR = 24; QSTART = 128; QFIT = 160; WIN = 64
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
QUAD = {2: {5}, 3: {8}, 8: {3, 4}}
COEFF = {3: -0.119, 4: 0.190}
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
def weight_scores_max(idx):
    """Max over far candidates (offset > WIN) of the weights-only match score; also argmax
    offset. §1239: same-token products are NEGATIVE — the discriminating statistic is the
    MOST NEGATIVE product, so we take max of (-score) = -min(score)."""
    B = idx.shape[0]
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    tot = torch.zeros(B, T, T, device=DEV)
    for L, hs in MATCHERS.items():
        at = m.transformer.h[L].attn
        dummy = torch.zeros(1, T, 9, 128, device=DEV)
        cos_t, sin_t = at.rotary(dummy)
        def pipe(lin):
            z = F.rms_norm(lin(x).view(B, T, 9, 128), (128,))
            return are(z, cos_t, sin_t)
        q = pipe(at.c_q); k = pipe(at.c_k); q2 = pipe(at.c_q2); k2 = pipe(at.c_k2)
        for h in hs:
            s1 = torch.einsum('bqd,bkd->bqk', q[:, :, h].float(), k[:, :, h].float()) / 128.0
            s2 = torch.einsum('bqd,bkd->bqk', q2[:, :, h].float(), k2[:, :, h].float()) / 128.0
            tot += s1 * s2
    ar = torch.arange(T, device=DEV)
    far = (ar[:, None] - ar[None, :]) > WIN
    neg = (-tot).masked_fill(~far.unsqueeze(0), float('-inf'))
    val, arg = neg.max(dim=-1)
    off = ar[None, :] - arg                       # (B,T) argmax offsets
    val = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
    return val, off






@torch.no_grad()
def clean_vv8(idx):
    """Layer-8 value-mix vv (B,T,9,128) from a clean forward."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        if L == 8:
            return vv.detach()
        x, v1 = blk(x, v1, x0)
    return None


@torch.no_grad()
def make_payload(idx, succ, source):
    """source: 'live' (clean vv8) or 'static' (v1 code). Payload (B,T,D) via c_proj slices
    with §1239 measured coefficients."""
    B = idx.shape[0]
    at8 = m.transformer.h[8].attn
    if source == 'live':
        vv = clean_vv8(idx)                                    # (B,T,9,128)
    else:
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        vv = m.transformer.h[0].attn.c_v(x).view(B, T, 9, 128)
    y = torch.zeros(B, T, 9, 128, device=DEV, dtype=vv.dtype)
    si = succ.clamp(0, T - 1)
    for h, c in COEFF.items():
        vh = torch.gather(vv[:, :, h, :], 1, si.unsqueeze(-1).expand(-1, -1, 128))
        y[:, :, h, :] = c * vh
    return at8.c_proj(y.reshape(B, T, D))


@torch.no_grad()
def forward_quad(idx, axis_d, axis_vals, payload):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        if axis_d is not None and 4 <= L <= 8:
            cur = (x.float() * axis_d).sum(-1)
            x = x + ((axis_vals[L] - cur).unsqueeze(-1) * axis_d).to(x.dtype)
        if payload is not None and L == 8:
            x = x + payload.to(x.dtype)
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
        if L in QUAD:
            msk = torch.stack([MASK_W if h in QUAD[L] else FULL for h in range(9)], 0)
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
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


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
    A = {}; Bc = {}
    caps_fit = {}; scores_fit = []
    for i in range(0, NFIT, 4):
        idx = FIT[i:i + 4, :-1].to(DEV).contiguous()
        _, caps = forward_gen(idx, False, d, None)
        sc, _ = weight_scores_max(idx)
        scores_fit.append(sc)
        for L, v in caps.items():
            caps_fit.setdefault(L, []).append(v)
    S = torch.cat(scores_fit)[:, QSTART:].reshape(-1)
    for L in caps_fit:
        Y = torch.cat(caps_fit[L])[:, QSTART:].reshape(-1)
        sm, ym = S.mean(), Y.mean()
        beta = float(((S - sm) * (Y - ym)).sum() / ((S - sm) ** 2).sum().clamp_min(1e-9))
        A[L] = beta; Bc[L] = float(ym - beta * sm)

    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in ('base', 'mask', 'axis_only', 'live', 'live_flip', 'static')}
    n = 0
    for i in range(0, NR, 4):
        bb = REP[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        sc, off = weight_scores_max(idx)
        vals = {L: A[L] * sc + Bc[L] for L in A}
        pos = torch.arange(T, device=DEV)[None, :]
        succ = (pos - off + 1).clamp(0, T - 1)
        pl_live = make_payload(idx, succ, 'live')
        pl_static = make_payload(idx, succ, 'static')
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)).float()
        ce['base'] += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, (av, p) in {'mask': (None, None), 'axis_only': (vals, None),
                               'live': (vals, pl_live), 'live_flip': (vals, -pl_live),
                               'static': (vals, pl_static)}.items():
            lo = forward_quad(idx, d if av is not None else None, av, p).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    dmg = CE['mask'] - CE['base']
    rec = {c: round((CE['mask'] - CE[c]) / max(dmg, 1e-6), 4) for c in ('axis_only', 'live', 'live_flip', 'static')}
    out = {'n_rows': NR, 'ce': CE, 'quad_damage': round(dmg, 4), 'recovery': rec,
           'pred_a_live_works': bool(rec['live'] >= rec['axis_only'] + 0.10),
           'pred_b_signs': bool(rec['live_flip'] <= rec['axis_only'] - 0.05),
           'pred_c_static_null': bool(rec['static'] <= rec['axis_only'] + 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | damage {dmg:.4f} | recovery {rec}")
    print(f"pred_a live {out['pred_a_live_works']} | pred_b signs {out['pred_b_signs']} | pred_c static {out['pred_c_static_null']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
