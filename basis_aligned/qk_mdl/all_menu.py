"""Stage-A flagship: compose the per-layer menu across ALL of bilin18's
attention. One estimation pass captures cond-mean QK factor tables for layers
1-17 (CPU accumulators, ~14GB); audits stream table rows from CPU per batch
(fp32 fidelity, no V-sized tables on GPU). Arms:
  A. all-table   — every layer 1-17 on cond-mean tables (L0 stays live; its
                   fold is exact, so selection is token-indexed EVERYWHERE)
  B. menu        — per-layer argmin from the individual sweeps
                   (zero: L8,L14,L15,L17 · live: L5 · table: rest)
  C. menu-static — menu but L5 tabled too (no live QK anywhere above L0)
Sum-of-parts for B is +0.234; the question is composition."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward

torch.manual_seed(0)
DEV = 'cuda'
LAYERS = list(range(1, 18))
ZERO_L = {8, 14, 15, 17}
LIVE_L = {5}
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/all_menu.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]

acc = {(L, n): torch.zeros(V, NH * HD) for L in LAYERS for n in ('q1', 'k1', 'q2', 'k2')}
cnt = torch.zeros(V)


@torch.no_grad()
def capture(idx):
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    flat = idx.reshape(-1).cpu()
    cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        if li in acc_layers:
            for name, lin in (('q1', a.c_q), ('k1', a.c_k), ('q2', a.c_q2), ('k2', a.c_k2)):
                z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
                acc[(li, name)].index_add_(0, flat, z.reshape(-1, NH * HD).float().cpu())
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cos, sin)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))


acc_layers = set(LAYERS)
for i in range(0, len(TRAIN), 8):
    capture(TRAIN[i:i + 8, :-1].to(DEV))
    if i % 256 == 0:
        print(f'  estimate {i}/{len(TRAIN)}', flush=True)
seen = cnt > 0
tables = {}
for L in LAYERS:
    for n in ('q1', 'k1', 'q2', 'k2'):
        a = acc.pop((L, n))
        t = a / cnt.clamp_min(1)[:, None]
        t[~seen] = a.sum(0) / cnt.sum()
        t = t.view(V, NH, HD)
        tables[(L, n)] = (t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()))
torch.save({f'{L}_{n}': v.half() for (L, n), v in tables.items()}, 'all17_tables.pt')
print('tables built + saved (fp16 on disk, fp32 in RAM)', flush=True)


def cs_scores(Fq, Fk, hd, dtype):
    d = hd // 2
    T = Fq.shape[1]
    cos, sin = rope_tables(T, hd, DEV, torch.float32, 'bf16')
    cosD = torch.einsum('if,jf->ijf', cos, cos) + torch.einsum('if,jf->ijf', sin, sin)
    sinD = torch.einsum('if,jf->ijf', sin, cos) - torch.einsum('if,jf->ijf', cos, sin)
    qa, qb = Fq[..., :d], Fq[..., d:]
    ka, kb = Fk[..., :d], Fk[..., d:]
    s = (torch.einsum('bihf,bjhf,ijf->bhij', qa, ka, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, kb, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', qb, ka, sinD)
         - torch.einsum('bihf,bjhf,ijf->bhij', qa, kb, sinD))
    return (s / hd).to(dtype)


@torch.no_grad()
def audit_ce(mode):
    # mode: dict layer -> 'table' | 'zero' | 'live'
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        idx_cpu = idx.cpu()

        def patch(li, s1, s2):
            mm = mode.get(li, 'live')
            if mm == 'live':
                return s1, s2
            if mm == 'zero':
                return torch.zeros_like(s1), torch.zeros_like(s2)
            outs = []
            for qn_, kn_ in (('q1', 'k1'), ('q2', 'k2')):
                Fq = tables[(li, qn_)][idx_cpu].to(DEV)
                Fk = tables[(li, kn_)][idx_cpu].to(DEV)
                outs.append(cs_scores(Fq, Fk, HD, s1.dtype))
            return outs[0], outs[1]

        logits = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


base = audit_ce({})
print(f'baseline CE {base:.4f}', flush=True)
res = {'baseline_ce': base, 'arms': {}}

arms = {
    'all-table (L1-17)': {L: 'table' for L in LAYERS},
    'menu (zero 8,14,15,17; live 5; table rest)':
        {L: ('zero' if L in ZERO_L else 'live' if L in LIVE_L else 'table')
         for L in LAYERS},
    'menu-static (menu + L5 tabled)':
        {L: ('zero' if L in ZERO_L else 'table') for L in LAYERS},
}
for name, mode in arms.items():
    d = audit_ce(mode) - base
    res['arms'][name] = d
    print(f'{name}: dCE {d:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('all menu done', flush=True)
