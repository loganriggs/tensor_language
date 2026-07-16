"""Iterated bottom-up re-estimation (NO training): rebuild each tabled layer's
cond-mean QK tables UNDER THE ALREADY-PATCHED lower stack (menu2 config:
zeros at 8/14/15/17, L5 heads 5,7 live, L0 live-exact). If the composed dCE
comes down from the wall (+0.76 trained / +1.8 untrained) toward sum-of-parts
(~+0.15-0.25) without touching a single trainable value, the wall is
distribution shift, not vq discreteness or table capacity.
Progressive audits after each layer give the cumulative curve."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
ZERO_L = {8, 14, 15, 17}
LIVE5 = (5, 7)
TAB_L = [L for L in range(1, 18) if L not in ZERO_L]
OUT = f'{QK}/iter_reestimate.json'
m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]

tables = {}          # (L, name) -> (V, NH, HD) fp32 CPU, filled bottom-up
NAMES = ('q1', 'k1', 'q2', 'k2')


def cs(Fq, Fk, dtype):
    d = HD // 2
    T = Fq.shape[1]
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosD = torch.einsum('if,jf->ijf', cos, cos) + torch.einsum('if,jf->ijf', sin, sin)
    sinD = torch.einsum('if,jf->ijf', sin, cos) - torch.einsum('if,jf->ijf', cos, sin)
    qa, qb = Fq[..., :d], Fq[..., d:]
    ka, kb = Fk[..., :d], Fk[..., d:]
    s = (torch.einsum('bihf,bjhf,ijf->bhij', qa, ka, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, kb, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, ka, sinD)
         - torch.einsum('bihf,bjhf,ijf->bhij', qa, kb, sinD))
    return (s / HD).to(dtype)


def patch_factory(idx, patched_layers):
    idx_cpu = idx.cpu()

    def patch(li, s1, s2):
        if li in ZERO_L:
            return torch.zeros_like(s1), torch.zeros_like(s2)
        if li not in patched_layers:
            return s1, s2
        n1 = cs(tables[(li, 'q1')][idx_cpu].to(DEV), tables[(li, 'k1')][idx_cpu].to(DEV), s1.dtype)
        n2 = cs(tables[(li, 'q2')][idx_cpu].to(DEV), tables[(li, 'k2')][idx_cpu].to(DEV), s2.dtype)
        if li == 5:
            keep = torch.tensor([h in LIVE5 for h in range(NH)],
                                device=DEV)[None, :, None, None]
            return torch.where(keep, s1, n1), torch.where(keep, s2, n2)
        return n1, n2
    return patch


@torch.no_grad()
def estimate_layer(L, patched_layers):
    """cond-mean factors at layer L under the patched lower stack."""
    acc = {n: torch.zeros(V, NH * HD) for n in NAMES}
    cnt = torch.zeros(V)
    for i in range(0, len(TRAIN), 8):
        idx = TRAIN[i:i + 8, :-1].to(DEV)
        B, T = idx.shape
        patch = patch_factory(idx, patched_layers)
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        flat = idx.reshape(-1).cpu()
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            if li == L:
                for name, lin in (('q1', a.c_q), ('k1', a.c_k), ('q2', a.c_q2), ('k2', a.c_k2)):
                    z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
                    acc[name].index_add_(0, flat, z.reshape(-1, NH * HD).float().cpu())
                break
            qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            s1, s2 = patch(li, s1, s2)
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    seen = cnt > 0
    for n in NAMES:
        t = acc[n] / cnt.clamp_min(1)[:, None]
        t[~seen] = acc[n].sum(0) / cnt.sum()
        t = t.view(V, NH, HD)
        tables[(L, n)] = t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


@torch.no_grad()
def audit_ce(patched_layers):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        logits = reference_forward(m, idx, 'bf16',
                                   score_patch=patch_factory(idx, patched_layers)).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = reference_forward(m, b[:, :-1], 'bf16').float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
base = tot / n
print(f'baseline CE {base:.4f}', flush=True)
res = {'baseline_ce': base, 'cumulative': {}}

done = set()
for L in TAB_L:
    estimate_layer(L, done)
    done.add(L)
    d = audit_ce(done) - base
    res['cumulative'][L] = d
    print(f'after re-estimating L{L} (stack of {len(done)}): dCE {d:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
torch.save({f'{L}_{n}': v.half() for (L, n), v in tables.items()},
           f'{QK}/iter_tables.pt')
print('iter reestimate done', flush=True)
