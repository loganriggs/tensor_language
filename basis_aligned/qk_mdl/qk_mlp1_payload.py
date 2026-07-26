"""TICK 248: block-1 bilinear MLP — interface rank and the symbol/payload split.

Block-1's MLP is the largest single memory-enrichment writer (tick 240, 0.50).
Question A (interface): what is the causal rank-truncation frontier of its OUTPUT?
Project mlp1's output onto its top-r PCA subspace (fit on the disjoint cooc corpus)
before adding to the residual; audit dCE at r in {4, 16, 64, 256} plus zeroed.
Question B (payload selectivity): under truncation, are strong-key (memory)
predictions hurt disproportionately versus random positions? Per-instance
orthogonal payloads live in the PCA tail by construction, so if block-1 writes
payloads, truncation should damage strong-key targets far more than average.
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
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
NEUTRAL = tok.encode(' one')[0]


@torch.no_grad()
def forward_proj(idx, proj=None):
    """Full 18-block forward; if proj is not None, block-1's MLP output is
    projected onto the row-space of proj (r, D) before being added."""
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
        if li == 1 and proj is not None:
            mo = (mo.float() @ proj.T @ proj).to(mo.dtype)
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


# ---- PCA basis of mlp1 output on the disjoint corpus (streaming covariance) ----
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
evals, evecs = evals[order], evecs[:, order]
frac = (evals / evals.sum()).cumsum(0)
print('mlp1 output PCA cum var:', {r: round(float(frac[r - 1]), 4)
      for r in (4, 16, 64, 256, 512)}, flush=True)

BASE = 3.07630
out = {'cumvar': {str(r): round(float(frac[r - 1]), 4) for r in (4, 16, 64, 256, 512)}}


@torch.no_grad()
def audit(proj, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        lg = forward_proj(b[:, :-1], proj).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n




# ---- Part B (standalone rerun, lean memory): strong-key selectivity ----
g = torch.Generator().manual_seed(7)
POS = []
T = FINEWEB[:128].shape[1]
for _ in range(160):
    di = int(torch.randint(0, 128, (1,), generator=g))
    p = int(torch.randint(32, T - 2, (1,), generator=g))
    POS.append((di, p))
SEQS = FINEWEB[:128]
KEY = []
for (di, p) in POS[:96]:
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
    drops = [float(lps[0] - lps[1 + i]) for i in range(len(offs))]
    KEY.append(max(drops))
proj64 = evecs[:, :64].T.float().contiguous()
proj256 = evecs[:, :256].T.float().contiguous()
res = {}
for pname, proj in (('rank64', proj64), ('rank256', proj256)):
    strong, weak = [], []
    with torch.no_grad():
        docs_needed = sorted(set(di for (di, p) in POS[:96]))
        lpf, lpt = {}, {}
        for di in docs_needed:
            idx = SEQS[di:di + 1, :-1].to(DEV)
            nxt = SEQS[di, 1:].to(DEV)
            Tm = idx.shape[1]
            ar = torch.arange(Tm, device=DEV)
            lg_f = forward_proj(idx, None)[0].float()
            lpf[di] = F.log_softmax(lg_f, -1)[ar, nxt].cpu()
            del lg_f
            lg_t = forward_proj(idx, proj)[0].float()
            lpt[di] = F.log_softmax(lg_t, -1)[ar, nxt].cpu()
            del lg_t
            torch.cuda.empty_cache()
        for (di, p), kd in zip(POS[:96], KEY):
            d = float(lpf[di][p] - lpt[di][p])
            (strong if kd > 1.0 else weak).append(d)
    res[f'strongkey_{pname}_drop'] = {'mean': round(float(np.mean(strong)), 3),
                                      'median': round(float(np.median(strong)), 3), 'n': len(strong)}
    if weak:
        res[f'weakkey_{pname}_drop'] = {'mean': round(float(np.mean(weak)), 3),
                                        'median': round(float(np.median(weak)), 3), 'n': len(weak)}
    # also: average drop over ALL positions >=32 in these docs (the corpus-wide yardstick)
    alld = []
    for di in docs_needed:
        alld.append(float((lpf[di][32:] - lpt[di][32:]).mean()))
    res[f'alldoc_{pname}_mean_drop'] = round(float(np.mean(alld)), 4)
prev = json.load(open(f'{QK}/qk_mlp1_rank.json'))
prev.update(res)
print(json.dumps(res, indent=1), flush=True)
json.dump(prev, open(f'{QK}/qk_mlp1_rank.json', 'w'), indent=2)
print('MLP1 PAYLOAD DONE', flush=True)
