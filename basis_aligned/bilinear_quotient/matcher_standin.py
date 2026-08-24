# matcher_standin: a WEIGHTS-COMPUTED STAND-IN for the matcher heads (benchmark-grade).
# Ingredients now certified: the match criterion is computable from weights+embeddings
# (§1238: raw wte codes through 2.5/3.8's q/k pipelines, AUC 1.0 inverted) and the verdict
# rides one axis whose clean restoration = 95.3% (§1250). The stand-in: mask both matchers;
# for each position t on repeat rows compute the WEIGHTS-ONLY match score of token t vs
# token t-128 (both matchers' pipelines at true offsets); map score -> axis value by a 1-D
# linear fit (on FIT rows against captured clean axis values); inject the SYNTHESIZED axis.
# If this works, two attention heads reduce to: an embedding-table lookup + one scalar map
# + one direction — a Type-1 stand-in for the copy circuit's front end.
#
# Registered predictions:
#   pred_a STAND-IN WORKS: synthesized-axis restore recovers >= 50% of the mask damage.
#   pred_b BEATS THE LEVEL: recovery > mean-restore's 71.8% is NOT required (that used
#          oracle per-row levels); the real bar: synthesized > row-mean-of-SYNTHESIZED
#          restore by >= 5 points (the weight-derived per-position signal adds value
#          beyond its own level).
#   pred_c CONTROL: score-shuffled synthesis (same 1-D map, scores permuted across
#          positions) recovers <= its own-level equivalent + 5 points.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_standin_results.json'
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
def weight_scores(idx):
    """Weights-only match score per position: sum over both matchers of s1*s2 between
    rms(wte(tok_t)) at position t and rms(wte(tok_{t-128})) at t-128. Zero for t<128."""
    B = idx.shape[0]
    x = F.rms_norm(m.transformer.wte(idx), (D,))              # (B,T,D) raw codes
    out = torch.zeros(B, T, device=DEV)
    for L, hs in MATCHERS.items():
        at = m.transformer.h[L].attn
        dummy = torch.zeros(1, T, 9, 128, device=DEV)
        cos_t, sin_t = at.rotary(dummy)
        def pipe(lin):
            z = F.rms_norm(lin(x).view(B, T, 9, 128), (128,))
            return are(z, cos_t, sin_t)
        q = pipe(at.c_q); k = pipe(at.c_k); q2 = pipe(at.c_q2); k2 = pipe(at.c_k2)
        for h in hs:
            qt = q[:, 128:, h]; kt = k[:, :T - 128, h]
            q2t = q2[:, 128:, h]; k2t = k2[:, :T - 128, h]
            s1 = (qt.float() * kt.float()).sum(-1) / 128.0
            s2 = (q2t.float() * k2t.float()).sum(-1) / 128.0
            out[:, 128:] += s1 * s2
    return out


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

    # 1-D map: weights-only score -> clean axis value, fit on FIT rows (per capture layer L)
    A = {}; Bc = {}
    caps_fit = {}; scores_fit = []
    for i in range(0, NFIT, 4):
        idx = FIT[i:i + 4, :-1].to(DEV).contiguous()
        _, caps = forward_gen(idx, False, d, None)
        sc = weight_scores(idx)
        scores_fit.append(sc)
        for L, v in caps.items():
            caps_fit.setdefault(L, []).append(v)
    S = torch.cat(scores_fit)[:, QSTART:].reshape(-1)
    for L in caps_fit:
        Y = torch.cat(caps_fit[L])[:, QSTART:].reshape(-1)
        sm, ym = S.mean(), Y.mean()
        beta = float(((S - sm) * (Y - ym)).sum() / ((S - sm) ** 2).sum().clamp_min(1e-9))
        A[L] = beta; Bc[L] = float(ym - beta * sm)
    print(f"1-D maps: {[(L, round(A[L], 4), round(Bc[L], 2)) for L in sorted(A)]}", flush=True)

    qp = torch.arange(QSTART, T, device=DEV)
    g = torch.Generator().manual_seed(13)
    perm = torch.randperm(T, generator=g).to(DEV)
    ce = {c: 0.0 for c in ('base', 'mask', 'synth', 'synth_mean', 'synth_shuf')}
    n = 0
    for i in range(0, NR, 4):
        bb = REP[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        sc = weight_scores(idx)
        vals = {L: A[L] * sc + Bc[L] for L in A}
        vals_mean = {L: v.mean(dim=1, keepdim=True).expand_as(v).contiguous() for L, v in vals.items()}
        sc_shuf = sc[:, perm]
        vals_shuf = {L: A[L] * sc_shuf + Bc[L] for L in A}
        conds = {'base': (False, None, None), 'mask': (True, None, None),
                 'synth': (True, d, vals), 'synth_mean': (True, d, vals_mean),
                 'synth_shuf': (True, d, vals_shuf)}
        for cname, (mk, dd, rv) in conds.items():
            lo, _ = forward_gen(idx, mk, dd, rv)
            lo = lo.float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    dmg = CE['mask'] - CE['base']
    rec = {c: round((CE['mask'] - CE[c]) / max(dmg, 1e-6), 4) for c in ('synth', 'synth_mean', 'synth_shuf')}
    out = {'n_rows': NR, 'ce': CE, 'mask_damage': round(dmg, 4), 'recovery': rec,
           'pred_a_standin_works': bool(rec['synth'] >= 0.50),
           'pred_b_beats_own_level': bool(rec['synth'] >= rec['synth_mean'] + 0.05),
           'pred_c_shuffle_control': bool(rec['synth_shuf'] <= rec['synth_mean'] + 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | damage {dmg:.4f} | recovery {rec}")
    print(f"pred_a works {out['pred_a_standin_works']} | pred_b beats-level {out['pred_b_beats_own_level']} | pred_c shuf {out['pred_c_shuffle_control']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
