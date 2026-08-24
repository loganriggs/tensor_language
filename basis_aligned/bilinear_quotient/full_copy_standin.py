# full_copy_standin: THE WHOLE NAMEABLE COPY FRONT END AS COMPUTED OPERATIONS. §1258 gave
# the matchers a weights-computed stand-in (0.92-0.97). Extension to the FETCHERS (8.3/8.4):
# their delivered payload is dominated by the successor token's block-0 value code (v1 =
# c_v of rms(wte), weights-computable from the token id, §1076/§1236). Stand-in: read-mask
# the quad (2.20-nat damage, §1207); restore (i) the synthesized verdict axis (§1258
# method) AND (ii) a computed fetch payload — the L8 c_proj image of the successor token's
# v1 code at heads 8.3/8.4's slices, scaled by one fitted scalar per head-pair, where the
# successor = token at (weight-score argmax + 1).
#
# Registered predictions:
#   pred_a FULL STAND-IN: axis + computed payload recovers >= 50% of the quad-mask damage.
#   pred_b PAYLOAD ADDS: >= 10 points over the synthesized-axis-only condition.
#   pred_c WRONG-SUCCESSOR CONTROL: payload from (argmax + 2) recovers <= axis-only + 5
#          points (it is the successor's identity, not generic mass).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'full_copy_standin_results.json'
NFIT = 12; NR = 24; QSTART = 128; QFIT = 160; WIN = 64
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
QUAD = {2: {5}, 3: {8}, 8: {3, 4}}
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
def v1_code(tokens):
    """Block-0 c_v code of each token id: c_v(rms(wte(t))). (B,T,9,128) -> (B,T,D) raw."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    return m.transformer.h[0].attn.c_v(x)                     # (B,T,D) flattened head-major


@torch.no_grad()
def fetch_payload(idx, succ_pos):
    """L8 c_proj image of the successor's v1 code through heads 8.3/8.4's slices. (B,T,D)"""
    at = m.transformer.h[8].attn
    B = idx.shape[0]
    succ_tok = torch.gather(idx, 1, succ_pos.clamp(0, T - 1))
    v1s = v1_code(succ_tok).view(B, T, 9, 128)
    y = torch.zeros(B, T, 9, 128, device=DEV, dtype=v1s.dtype)
    lam = float(at.lamb)
    for h in (3, 4):
        y[:, :, h, :] = lam * v1s[:, :, h, :]
    return at.c_proj(y.reshape(B, T, D))


@torch.no_grad()
def forward_quad(idx, axis_d, axis_vals, payload, pscale):
    """Quad read-masked; optional axis restoration (blocks 4-8) and L8-entry payload add."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        if axis_d is not None and 4 <= L <= 8:
            cur = (x.float() * axis_d).sum(-1)
            x = x + ((axis_vals[L] - cur).unsqueeze(-1) * axis_d).to(x.dtype)
        if payload is not None and L == 8:
            x = x + pscale * payload.to(x.dtype)
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

    # axis 1-D map (max statistic) on FIT rows
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

    # payload scale: coarse fit on FIT rows (grid over pscale, best CE)
    qpf = torch.arange(QSTART, T, device=DEV)
    best_ps, best_ce = 0.0, 1e9
    idxf = FIT[:8, :-1].to(DEV).contiguous(); tgtf = FIT[:8, 1:].to(DEV).contiguous()
    scf, offf = weight_scores_max(idxf)
    valsf = {L: A[L] * scf + Bc[L] for L in A}
    succf = (torch.arange(T, device=DEV)[None, :] - offf + 1).clamp(0, T - 1)
    pf = fetch_payload(idxf, succf)
    for ps in (0.0, 0.25, 0.5, 1.0, 2.0):
        lo = forward_quad(idxf, d, valsf, pf, ps).float()
        cec = float(F.cross_entropy(lo[:, qpf].reshape(-1, lo.shape[-1]), tgtf[:, qpf].reshape(-1)))
        print(f"pscale {ps}: fit CE {cec:.4f}", flush=True)
        if cec < best_ce:
            best_ce, best_ps = cec, ps
    print(f"chosen pscale {best_ps}", flush=True)

    qp = torch.arange(QSTART, T, device=DEV)
    ce = {c: 0.0 for c in ('base', 'mask', 'axis_only', 'full', 'wrong_succ')}
    n = 0
    for i in range(0, NR, 4):
        bb = REP[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        sc, off = weight_scores_max(idx)
        vals = {L: A[L] * sc + Bc[L] for L in A}
        pos = torch.arange(T, device=DEV)[None, :]
        succ = (pos - off + 1).clamp(0, T - 1)
        succ2 = (pos - off + 2).clamp(0, T - 1)
        pl = fetch_payload(idx, succ)
        pl2 = fetch_payload(idx, succ2)
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)).float()
        ce['base'] += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, (av, p, ps) in {'mask': (None, None, 0.0), 'axis_only': (vals, None, 0.0),
                                   'full': (vals, pl, best_ps), 'wrong_succ': (vals, pl2, best_ps)}.items():
            lo = forward_quad(idx, d if av is not None else None, av, p, ps).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    dmg = CE['mask'] - CE['base']
    rec = {c: round((CE['mask'] - CE[c]) / max(dmg, 1e-6), 4) for c in ('axis_only', 'full', 'wrong_succ')}
    out = {'n_rows': NR, 'pscale': best_ps, 'ce': CE, 'quad_damage': round(dmg, 4), 'recovery': rec,
           'pred_a_full_works': bool(rec['full'] >= 0.50),
           'pred_b_payload_adds': bool(rec['full'] >= rec['axis_only'] + 0.10),
           'pred_c_wrong_succ': bool(rec['wrong_succ'] <= rec['axis_only'] + 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | damage {dmg:.4f} | recovery {rec}")
    print(f"pred_a full {out['pred_a_full_works']} | pred_b adds {out['pred_b_payload_adds']} | pred_c wrong {out['pred_c_wrong_succ']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
