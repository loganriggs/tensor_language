# matcher_standin2: the GENERAL-CONTEXT stand-in (§1257's registered extension). No offset
# knowledge: for each position t, compute the weights-only match score against ALL far
# candidates (offsets 65..t) from raw wte codes through the matchers' pipelines, take the
# MAX as the match statistic; 1-D affine map (fit on period-128 rows with the SAME max
# statistic) -> axis injection. Deciding test: PERIOD-160 repeat rows (source at t-160 —
# an offset the stand-in never saw).
#
# Registered predictions:
#   pred_a GENERALIZES: on period-160 rows the stand-in recovers >= 40% of the matcher-mask
#          damage there.
#   pred_b THE SCORE FINDS THE SOURCE: weight-score argmax = the true source offset for
#          >= 70% of scored positions on period-160 rows.
#   pred_c CONSISTENCY: on period-128 rows the max-statistic stand-in lands within 10
#          points of §1257's fixed-offset synth (92.0%).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_standin2_results.json'
NFIT = 12; NR = 24; QSTART = 128; QFIT = 160; WIN = 64
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
MASK_W = None
FULL = None
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_standin2_results.json'
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
def main():
    global MASK_W, FULL
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar = torch.arange(T, device=DEV)
    vis = ((ar[:, None] - ar[None, :]) < WIN) | (ar[None, :] == 0)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL & vis
    FIT = cl.fineweb_rows(NFIT)[:, :T + 1].contiguous().clone()
    FIT[:, 128:256] = FIT[:, 0:128]
    R128 = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    R128[:, 128:256] = R128[:, 0:128]
    R160 = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    R160[:, 160:256] = R160[:, 0:96]              # period-160 repeat (96 repeated tokens)
    d = fit_axis(FIT)

    # fit the 1-D map on FIT rows with the MAX statistic
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

    def eval_rows(ROWS, qstart, true_off):
        qp = torch.arange(qstart, T, device=DEV)
        ce = {c: 0.0 for c in ('base', 'mask', 'synth')}; n = 0
        acc_hits = 0; acc_tot = 0
        for i in range(0, NR, 4):
            bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            sc, off = weight_scores_max(idx)
            if true_off is not None:
                hits = (off[:, qstart:] == true_off)
                acc_hits += int(hits.sum()); acc_tot += hits.numel()
            vals = {L: A[L] * sc + Bc[L] for L in A}
            for cname, (mk, dd, rv) in {'base': (False, None, None), 'mask': (True, None, None),
                                        'synth': (True, d, vals)}.items():
                lo, _ = forward_gen(idx, mk, dd, rv)
                lo = lo.float()
                ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                                   tgt[:, qp].reshape(-1), reduction='sum'))
            n += 4 * len(qp)
        CE = {c: round(v / n, 4) for c, v in ce.items()}
        dmg = CE['mask'] - CE['base']
        rec = (CE['mask'] - CE['synth']) / max(dmg, 1e-6)
        acc = acc_hits / max(acc_tot, 1) if true_off is not None else None
        return CE, round(dmg, 4), round(rec, 4), (round(acc, 4) if acc is not None else None)

    CE128, dmg128, rec128, _ = eval_rows(R128, 128, None)
    print(f"p128: CE {CE128} dmg {dmg128} rec {rec128}", flush=True)
    CE160, dmg160, rec160, acc160 = eval_rows(R160, 160, 160)
    print(f"p160: CE {CE160} dmg {dmg160} rec {rec160} | argmax acc {acc160}", flush=True)

    out = {'n_rows': NR, 'p128': {'ce': CE128, 'damage': dmg128, 'recovery': rec128},
           'p160': {'ce': CE160, 'damage': dmg160, 'recovery': rec160, 'argmax_acc': acc160},
           'pred_a_generalizes': bool(rec160 >= 0.40),
           'pred_b_finds_source': bool(acc160 is not None and acc160 >= 0.70),
           'pred_c_consistency': bool(abs(rec128 - 0.9201) <= 0.10),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a gen {out['pred_a_generalizes']} | pred_b source {out['pred_b_finds_source']} | pred_c consist {out['pred_c_consistency']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
