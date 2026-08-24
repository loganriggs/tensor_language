# swiglu_axis: does the KEY-COMPOSITION sibling have an explicit match-verdict variable?
# bilin18's direct-match design writes a verdict to the stream (the match-evidence axis:
# remove 1.00 nat, restore 95%, §1249-50). swiglu18 matches INSIDE its fetchers' QK
# (§1217, no matcher heads) — the verdict need never be written to the stream at all.
#
# Procedure mirrored: delta = mlp5(x) - mlp5(x - L5H2_write) on repeat rows (L5H2 = its
# main station, §1210); PC1 on 12 fit rows; project the axis out of the stream at block
# entries 6-10; CE on 24 disjoint repeat rows + prose control.
#
# Registered predictions (the DESIGN hypothesis):
#   pred_a NO EXPLICIT VERDICT: axis removal costs <= 0.1 nats on repeat (vs bilin18's
#          1.00) — the key-composition architecture keeps the verdict implicit.
#   pred_b RANDOM NULL <= 0.03.
#   pred_c PIPELINE SANITY: the delta itself is nonzero and stable (split-half PC1 |cos|
#          >= 0.5) — the axis exists as a statistical object; the claim is about its
#          CAUSAL role, so a degenerate/noise axis would void pred_a (logged either way).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'swiglu_axis_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NFIT = 12; NR = 24; QSTART = 128; QFIT = 160
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb
STL, STH = 5, 2


@torch.no_grad()
def block_pass(x, x0, v1, blk, capture_head=None):
    at = blk.attn
    xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
    xin = F.rms_norm(xm, (D,))
    B, Tn = x.shape[0], x.shape[1]
    q = F.rms_norm(at.c_q(xin).view(B, Tn, 9, 128), (128,))
    k = F.rms_norm(at.c_k(xin).view(B, Tn, 9, 128), (128,))
    cos, sin = at.rotary(at.c_q(xin).view(B, Tn, 9, 128))
    q = are(q, cos, sin); k = are(k, cos, sin)
    scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
    tril = torch.tril(torch.ones(Tn, Tn, device=DEV, dtype=torch.bool))
    pat = F.softmax(scores.masked_fill(~tril, float('-inf')), dim=-1)
    v = at.c_v(xin).view(B, Tn, 9, 128)
    if v1 is None:
        v1 = v
    vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
    y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
    w = None
    if capture_head is not None:
        yh = torch.zeros_like(y)
        yh[:, :, capture_head, :] = y[:, :, capture_head, :]
        w = at.c_proj(yh.reshape(B, Tn, D)).float()
    x = xm + at.c_proj(y.reshape(B, Tn, D))
    return x, v1, w


@torch.no_grad()
def fit_axis(rows):
    DL = []
    for i in range(0, NFIT, 4):
        idx = rows[i:i + 4, :-1].to(DEV).contiguous()
        x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for L, blk in enumerate(mdl.transformer.h):
            cap = STH if L == STL else None
            x, v1, w = block_pass(x, x0, v1, blk, cap)
            if L == STL:
                full = blk.mlp(F.rms_norm(x, (D,)))
                blind = blk.mlp(F.rms_norm(x - w.to(x.dtype), (D,)))
                DL.append((full - blind).float()[:, QFIT:].reshape(-1, D).cpu())
                break
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    DL = torch.cat(DL)
    Dc = DL - DL.mean(0)
    half = Dc.shape[0] // 2
    _, _, Va = torch.pca_lowrank(Dc[:half], q=4)
    _, _, Vb = torch.pca_lowrank(Dc[half:], q=4)
    stab = float(F.cosine_similarity(Va[:, 0], Vb[:, 0], dim=0).abs())
    _, _, V = torch.pca_lowrank(Dc, q=4)
    return V[:, 0].to(DEV), stab


@torch.no_grad()
def forward_rm(idx, d):
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for L, blk in enumerate(mdl.transformer.h):
        if d is not None and 6 <= L <= 10:
            x = x - (x * d).sum(-1, keepdim=True) * d
        x, v1, _ = block_pass(x, x0, v1, blk)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def ce_of(rows, d):
    qp = torch.arange(QSTART, T, device=DEV)
    tot = 0.0; n = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_rm(idx, d).float()
        tot += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += idx.shape[0] * len(qp)
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FIT = cl.fineweb_rows(NFIT)[:, :T + 1].contiguous().clone()
    FIT[:, 128:256] = FIT[:, 0:128]
    REP = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    REP[:, 128:256] = REP[:, 0:128]
    PROSE = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    d, stab = fit_axis(FIT)
    g = torch.Generator(device=DEV).manual_seed(8)
    dr = torch.randn(D, device=DEV, generator=g); dr = dr / dr.norm()

    ce = {'base_rep': ce_of(REP, None), 'axis_rep': ce_of(REP, d), 'rand_rep': ce_of(REP, dr),
          'base_pr': ce_of(PROSE, None), 'axis_pr': ce_of(PROSE, d)}
    CE = {k: round(v, 4) for k, v in ce.items()}
    d_rep = CE['axis_rep'] - CE['base_rep']
    d_rand = CE['rand_rep'] - CE['base_rep']
    d_pr = CE['axis_pr'] - CE['base_pr']
    out = {'model': 'swiglu18', 'ce': CE, 'splithalf_pc1': round(stab, 4),
           'deltas': {'axis_repeat': round(d_rep, 4), 'rand_repeat': round(d_rand, 4),
                      'axis_prose': round(d_pr, 4)},
           'bilin18_ref': {'axis_repeat': 1.001},
           'pred_a_no_explicit_verdict': bool(d_rep <= 0.1),
           'pred_b_rand_null': bool(d_rand <= 0.03),
           'pred_c_axis_exists': bool(stab >= 0.5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE} | PC1 stab {stab:.3f} | deltas {out['deltas']}")
    print(f"pred_a noverdict {out['pred_a_no_explicit_verdict']} | pred_b rand {out['pred_b_rand_null']} | pred_c exists {out['pred_c_axis_exists']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
