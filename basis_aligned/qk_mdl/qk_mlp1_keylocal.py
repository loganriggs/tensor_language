"""TICK 249: key-position-local truncation of block-1's MLP output.

The discriminating payload test: on failure-packet positions (where the memory
pipeline demonstrably carries the prediction), truncate mlp1's output to rank r
AT THE KEY POSITION ONLY and measure the target's log-probability drop. Controls:
same truncation at a random non-key position in the same document (should be ~0),
and zeroing at the key position (ceiling). If key-position writes need the tail
(large drop even at rank 256), block-1's payload role is causal at component
grain; if rank 64 suffices locally, the flat global tail is *aggregate diversity
across positions* rather than per-write high rank — also informative.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, reference_forward
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
tok = AutoTokenizer.from_pretrained('gpt2')
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
SEQS = FINEWEB[:128]
NEUTRAL = tok.encode(' one')[0]
pk = json.load(open(f'{QK}/qk_failure_packets.json'))
docs = [SEQS[i].tolist() for i in range(128)]
dec_cache = [tok.decode(d) for d in docs]


@torch.no_grad()
def forward_kl(idx, spec=None):
    """spec = (positions, proj_or_None): at block-1's MLP output, the given
    positions are projected onto proj rows (or zeroed if proj is None)."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (x.size(-1),)))
        if li == 1 and spec is not None:
            mo = mo.clone()
            pos, proj = spec
            if proj is None:
                mo[:, pos] = 0.0
            else:
                mo[:, pos] = (mo[:, pos].float() @ proj.T @ proj).to(mo.dtype)
        x = x + mo
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def mlp1_out(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for li in (0, 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (x.size(-1),)))
        if li == 1:
            return mo
        x = x + mo


# PCA basis (identical construction to tick 248)
cov = torch.zeros(D, D, device=DEV, dtype=torch.float64)
mu = torch.zeros(D, device=DEV, dtype=torch.float64)
n_tot = 0
with torch.no_grad():
    for i in range(0, 512, 4):
        b = COOC[i:i + 4].to(DEV)
        mo = mlp1_out(b[:, :-1]).float().reshape(-1, D).double()
        cov += mo.T @ mo
        mu += mo.sum(0)
        n_tot += mo.shape[0]
mu /= n_tot
cov = cov / n_tot - torch.outer(mu, mu)
evals, evecs = torch.linalg.eigh(cov)
order = evals.argsort(descending=True)
evecs = evecs[:, order]
P64 = evecs[:, :64].T.float().contiguous().to(DEV)
P256 = evecs[:, :256].T.float().contiguous().to(DEV)
print('basis ready', flush=True)

# failure-packet positions + their keys (standard procedure)
POS = []
for pkt in pk:
    for s in pkt['samples']:
        ctx_tail = s['context'][-60:]
        for di, dtext in enumerate(dec_cache):
            if ctx_tail in dtext:
                seq = docs[di]
                acc = ''
                for p in range(len(seq) - 1):
                    acc += tok.decode([seq[p]])
                    if acc.endswith(ctx_tail):
                        POS.append((di, p))
                        break
                break
POS = POS[:96]
KEYJ = []
for (di, p) in POS:
    tgt = SEQS[di, p + 1]
    idx = SEQS[di:di + 1, :-1]
    offs = list(range(0, min(16, p + 1)))
    variants = [idx.clone()]
    for off in offs:
        v = idx.clone()
        v[0, p - off] = NEUTRAL
        variants.append(v)
    with torch.no_grad():
        lg = reference_forward(m, torch.cat(variants, 0).to(DEV), 'bf16')[:, p].float()
        lps = F.log_softmax(lg, 1)[:, tgt]
    drops = [(off, float(lps[0] - lps[1 + i])) for i, off in enumerate(offs)]
    drops.sort(key=lambda x: -x[1])
    KEYJ.append(p - drops[0][0] if drops[0][1] > 1.0 else None)
sel = [(POS[i][0], POS[i][1], KEYJ[i]) for i in range(len(POS)) if KEYJ[i] is not None][:64]
print(f'{len(sel)} positions', flush=True)

g2 = torch.Generator().manual_seed(3)
agg = {k: [] for k in ('key_r64', 'key_r256', 'key_zero', 'ctrl_r64', 'ctrl_zero')}
for (di, p, j) in sel:
    tgt = int(SEQS[di, p + 1])
    idx = SEQS[di:di + 1, :-1].to(DEV)
    lp_c = float(F.log_softmax(forward_kl(idx)[0, p].float(), 0)[tgt])
    # random non-key control position, != j (lower bound adapts for early p)
    lo = 32 if p > 33 else 1
    if p - lo < 2:
        continue
    while True:
        jc = int(torch.randint(lo, p, (1,), generator=g2))
        if jc != j:
            break
    for name, pos, proj in (('key_r64', [j], P64), ('key_r256', [j], P256),
                            ('key_zero', [j], None), ('ctrl_r64', [jc], P64),
                            ('ctrl_zero', [jc], None)):
        lp = float(F.log_softmax(forward_kl(idx, (pos, proj))[0, p].float(), 0)[tgt])
        agg[name].append(lp_c - lp)
out = {k: {'mean': round(float(np.mean(v)), 3), 'median': round(float(np.median(v)), 3),
           'n': len(v)} for k, v in agg.items()}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(f'{QK}/qk_mlp1_keylocal.json', 'w'), indent=2)
print('MLP1 KEYLOCAL DONE', flush=True)
