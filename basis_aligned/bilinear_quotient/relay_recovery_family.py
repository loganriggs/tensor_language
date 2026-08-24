# relay_recovery_family: §1191's relay quantifier on swiglu18. All-18 read-masking at
# W in {16,32,64} vs swiglu18's truncation curve (§1181: 0.6504/0.4070/0.2260).
# Registered: (a) mask <= truncation at every W; (b) recovery <= 0.30 at every W;
# (c) monotone in W.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'relay_recovery_family_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('swiglu18'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; T = 256; NR = 24; WIN = 64; QSTART = 128
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def forward_banded(idx, band, MASK_W, FULL):
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(mdl.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
        k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(q, cos, sin); k = are(k, cos, sin)
        scores = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (128 ** 0.5)
        msk = MASK_W if L in band else FULL
        scores = scores.masked_fill(~msk, float('-inf'))
        pat = F.softmax(scores, dim=-1)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ar_g = torch.arange(T, device=DEV)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MASK_W = FULL
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    ALL = set(range(18))
    qp = torch.arange(QSTART, T, device=DEV)
    WSL = [16, 32, 64]
    ce = {'base': 0.0}; ce.update({f'W{w}': 0.0 for w in WSL}); n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lo = forward_banded(idx, set(), MASK_W, FULL).float()
        ce['base'] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]), tgt[:, qp].reshape(-1), reduction="sum"))
        for w in WSL:
            near = (ar_g[:, None] - ar_g[None, :]) < w
            mk = FULL & (near | (ar_g[None, :] == 0))
            lo = forward_banded(idx, ALL, mk, FULL).float()
            ce[f'W{w}'] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                                 tgt[:, qp].reshape(-1), reduction="sum"))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    cost = {f'W{w}': round(CE[f'W{w}'] - CE['base'], 4) for w in WSL}
    TRUNC = {'W16': 0.6504, 'W32': 0.407, 'W64': 0.226}
    rec = {k: round((TRUNC[k] - cost[k]) / TRUNC[k], 3) for k in cost}
    seq = [cost[f'W{w}'] for w in WSL]
    out = {'model': 'swiglu18', 'n_rows': NR, 'ce': CE, 'readmask_cost': cost,
           'truncation_refs_1181': TRUNC, 'relay_recovery_fraction': rec,
           'bilin18_recovery': {'W16': 0.191, 'W32': 0.168, 'W64': 0.15},
           'pred_a_mask_le_trunc': bool(all(cost[k] <= TRUNC[k] for k in cost)),
           'pred_b_recovery_le_30pct': bool(all(v <= 0.3 for v in rec.values())),
           'pred_c_monotone': bool(seq[0] > seq[1] > seq[2]),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"readmask {cost} vs trunc {TRUNC} | recovery {rec}")
    print(f"pred_a {out['pred_a_mask_le_trunc']} | pred_b {out['pred_b_recovery_le_30pct']} | pred_c {out['pred_c_monotone']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
