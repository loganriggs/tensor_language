# relay_bilin12: §1193 triangulation — does relaying track the ATTENTION NONLINEARITY
# (bilinear yes / softmax no) or depth/scale? bilin12 = 12L, D=768, squared attention like
# bilin18. Truncation AND read-masking measured in one harness at W in {32, 64}, scored
# positions >= 128.
# Registered: (a) BILINEAR FINGERPRINT: recovery = (trunc - mask)/trunc >= 0.10 at both W;
# (b) alternative: recovery <= 0.05 -> depth/scale hypothesis; (c) sanity: unmasked banded
# forward == true model CE +/- 0.005.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
import census_lib as cl

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'relay_bilin12_results.json'
DEV = 'cuda'
mdl, cfg = load_elriggs('bilin12'); mdl = mdl.to(DEV).eval()
D = cfg['n_embd']; NL = cfg['n_layer']; H = cfg['n_head']; HD = D // H
T = 256; NR = 24; QSTART = 128
are = sys.modules[type(mdl.transformer.h[0].attn).__module__].apply_rotary_emb
SQUARED = cfg.get('squared_attn', False)


@torch.no_grad()
def forward_masked(idx, mask):
    """Full model; attention pattern masked with `mask` (bool TxT) at ALL layers."""
    x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(mdl.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, H, HD))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, H, HD), (HD,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, H, HD), (HD,)), cos, sin)
        if SQUARED:
            # bilin12: single-score squared attention, ROW-NORMALIZED (naive_squared_attention)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / HD).square()
            pat = pat.masked_fill(~mask, 0.0)
            pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
        else:
            sc = torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / (HD ** 0.5)
            sc = sc.masked_fill(~mask, float('-inf'))
            pat = F.softmax(sc, dim=-1)
        v = at.c_v(xin).view(B, T, H, HD)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def trunc_forward(idx, w, qpos):
    """Truncation: full model on each scored position's last-w tokens."""
    B = idx.shape[0]
    wins = torch.stack([idx[:, t - w + 1: t + 1] for t in qpos], 1)
    Q = wins.shape[1]
    flat = wins.reshape(B * Q, w)
    outs = []
    step = max(64, 4096 // w)
    for j in range(0, flat.shape[0], step):
        x = F.rms_norm(mdl.transformer.wte(flat[j:j + step]), (D,)); x0 = x; v1 = None
        for blk in mdl.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = 30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)
        outs.append(lg[:, -1].float())
    return torch.cat(outs, 0).reshape(B, Q, -1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    ar = torch.arange(T, device=DEV)
    FULL = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    qpos = list(range(QSTART, T, 2)); qp = torch.tensor(qpos, device=DEV)
    WSL = [32, 64]
    ce = {'base': 0.0, 'true': 0.0}
    for w in WSL:
        ce[f'mask{w}'] = 0.0; ce[f'trunc{w}'] = 0.0
    n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(mdl.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in mdl.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = (30.0 * torch.tanh(mdl.lm_head(F.rms_norm(x, (D,))) / 30.0)).float()
        ce['true'] += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        lo = forward_masked(idx, FULL).float()
        ce['base'] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for w in WSL:
            near = (ar[:, None] - ar[None, :]) < w
            mk = FULL & (near | (ar[None, :] == 0))
            lo = forward_masked(idx, mk).float()
            ce[f'mask{w}'] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                                    tgt[:, qp].reshape(-1), reduction='sum'))
            lw = trunc_forward(idx, w, qpos)
            ce[f'trunc{w}'] += float(F.cross_entropy(lw.reshape(-1, lw.shape[-1]),
                                                     tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qpos)
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    cost = {}
    for w in WSL:
        cost[f'mask{w}'] = round(CE[f'mask{w}'] - CE['base'], 4)
        cost[f'trunc{w}'] = round(CE[f'trunc{w}'] - CE['base'], 4)
    rec = {f'W{w}': round((cost[f'trunc{w}'] - cost[f'mask{w}']) / max(cost[f'trunc{w}'], 1e-6), 3)
           for w in WSL}
    out = {'model': 'bilin12', 'n_rows': NR, 'ce': CE, 'cost': cost, 'recovery': rec,
           'family_refs': {'bilin18': {'W32': 0.168, 'W64': 0.15}, 'swiglu18': {'W32': 0.03, 'W64': 0.013}},
           'sanity': bool(abs(CE['base'] - CE['true']) <= 0.005),
           'pred_a_bilinear_fingerprint': bool(all(v >= 0.10 for v in rec.values())),
           'pred_b_depth_scale': bool(all(v <= 0.05 for v in rec.values())),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CE {CE}")
    print(f"cost {cost} | recovery {rec}")
    print(f"sanity {out['sanity']} | pred_a bilinear {out['pred_a_bilinear_fingerprint']} | pred_b depth/scale {out['pred_b_depth_scale']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
