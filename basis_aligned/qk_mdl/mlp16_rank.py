"""Top-MLP arc probe 2 (the H7 playbook on mlp16): even though L16's MLP
consumes diffuse long-range input (TM-1), its OUTPUT may be low-rank around
token-conditional means. Replace mlp_out at layer 16 by
mean[t] + rank-k projection of the deviation (live coefficients), k in
{1,4,16,64}; audit natural dCE. Same for L13 (most diffuse) as contrast."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
LAYERS = (16, 13)
OUT = f'{QK}/mlp16_rank.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
EST = build_eval_tokens(n_chunks=20 + 256, seq_len=513)[20:][:, :-1]


@torch.no_grad()
def forward(idx, target=None, mean=None, basis=None, grab=False):
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    grabbed = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        if li == target:
            if grab:
                grabbed = mlp_out
            else:
                mu = mean[idx.cpu()].to(DEV)
                dev = mlp_out - mu
                proj = torch.einsum('btd,kd->btk', dev, basis)
                mlp_out = mu + torch.einsum('btk,kd->btd', proj, basis)
        x = x + mlp_out
    if grabbed is not None:
        return grabbed
    xf = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(xf) / 30)


@torch.no_grad()
def ce_eval(target=None, mean=None, basis=None):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = forward(b[:, :-1], target=target, mean=mean, basis=basis).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


base = ce_eval()
res = {'baseline': base}
print(f'baseline {base:.4f}', flush=True)
for L in LAYERS:
    acc = torch.zeros(V, D)
    cnt = torch.zeros(V)
    outs = []
    toks = []
    for i in range(0, len(EST), 4):
        idx = EST[i:i + 4].to(DEV)
        o = forward(idx, target=L, grab=True)
        flat = idx.reshape(-1).cpu()
        acc.index_add_(0, flat, o.reshape(-1, D).float().cpu())
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        if i < 64:
            outs.append(o.reshape(-1, D).float())
            toks.append(flat)
    mean = acc / cnt.clamp_min(1)[:, None]
    mean[cnt == 0] = acc.sum(0) / cnt.sum()
    Osub = torch.cat(outs)
    dev = Osub - mean[torch.cat(toks)].to(DEV)
    C = (dev.T @ dev) / len(dev)
    evals, evecs = torch.linalg.eigh(C)
    evals, evecs = evals.flip(0), evecs.flip(1)
    shares = (evals[:64] / evals.sum()).cpu()
    res[f'L{L}_dev_pc_shares'] = [round(v.item(), 4) for v in shares[:8].tolist()]
    print(f'L{L} dev PC shares: {[round(v.item(), 3) for v in shares[:6]]}', flush=True)
    d0 = ce_eval(target=L, mean=mean, basis=torch.zeros(1, D, device=DEV))
    res[f'L{L} mean-only (rank-0)'] = d0 - base
    print(f'L{L} mean-only: dCE {d0 - base:+.4f}', flush=True)
    for kk in (1, 4, 16, 64):
        basis = evecs[:, :kk].T.contiguous()
        d = ce_eval(target=L, mean=mean, basis=basis) - base
        res[f'L{L} rank{kk}'] = d
        print(f'L{L} rank-{kk}: dCE {d:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)
print('mlp16 rank done', flush=True)
