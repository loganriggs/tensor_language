"""TICK 237 (Logan): token- and activation-patching at scale.

PART A — key taxonomy over the failure set: for the worst-192 positions (from the
failure packets' clusters, re-located), substitute each window token (offsets 0-15)
with a neutral token and measure dlogp(target). Classify each position's key:
  single-token (one dominant |dlogp|>1), compositional (>=2 heavy contiguous),
  syntactic (heaviest is punctuation/boundary), diffuse (none heavy).

PART B — activation-patching causal traces on 4 located examples: corrupt the top key
token, then restore CLEAN activations at three sites x two position sets and measure
logp recovery: sites = layer-0 attention output (yh), block-0 MLP output (mo),
layer-1 QK factors; positions = {key token pos} and {key..target span}. This maps the
key's causal route: token -> [attn | MLP] -> layer-1 keys -> prediction.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
tok = AutoTokenizer.from_pretrained('gpt2')
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQS = FINEWEB[:128]
NEUTRAL = tok.encode(' one')[0]
pk = json.load(open(f'{QK}/qk_failure_packets.json'))
docs = [SEQS[i].tolist() for i in range(128)]
dec_cache = [tok.decode(d) for d in docs]

# punctuation/boundary flags
PUNC = torch.zeros(V, dtype=torch.bool)
for t in range(V):
    s = tok.decode([t])
    PUNC[t] = (len(s.strip()) > 0 and not any(c.isalnum() for c in s)) or s.startswith('\n')

# locate up to 192 positions across packets
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
                        POS.append((di, p, pkt['cluster']))
                        break
                break
POS = POS[:192]
print(f'{len(POS)} positions located', flush=True)

# ---- PART A ----
tax = {'single': 0, 'compositional': 0, 'syntactic': 0, 'diffuse': 0}
keylens = []
for (di, p, ci) in POS:
    tgt = SEQS[di, p + 1]
    idx = SEQS[di:di + 1, :-1]
    variants = [idx.clone()]
    offs = list(range(0, min(16, p + 1)))
    for off in offs:
        v = idx.clone()
        v[0, p - off] = NEUTRAL
        variants.append(v)
    batch = torch.cat(variants, 0).to(DEV)
    with torch.no_grad():
        lg = reference_forward(m, batch, 'bf16')[:, p].float()
        lps = F.log_softmax(lg, 1)[:, tgt]
    base = float(lps[0])
    drops = [(off, base - float(lps[1 + i])) for i, off in enumerate(offs)]
    heavy = [(off, d) for off, d in drops if d > 1.0]
    if not heavy:
        tax['diffuse'] += 1
    else:
        heavy.sort(key=lambda x: -x[1])
        top_off = heavy[0][0]
        top_tok = int(SEQS[di, p - top_off])
        if PUNC[top_tok]:
            tax['syntactic'] += 1
        elif len(heavy) >= 2:
            tax['compositional'] += 1
        else:
            tax['single'] += 1
        keylens.append(len(heavy))
print(f'KEY TAXONOMY over {len(POS)} failures: {tax}; '
      f'median heavy-key size {np.median(keylens) if keylens else 0}', flush=True)
out = {'taxonomy': tax, 'n': len(POS),
       'median_key_tokens': float(np.median(keylens)) if keylens else 0}
json.dump(out, open(f'{QK}/qk_patch_program.json', 'w'), indent=2)

# ---- PART B: activation patching on the 4 known examples ----
ex = json.load(open(f'{QK}/qk_key_ablation.json'))


@torch.no_grad()
def run_cached(idx, patch=None, cache=None):
    """Manual 18-layer forward. patch=(site, positions, cache) restores clean acts."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    my_cache = {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin, nm):
            z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
            if li == 1:
                my_cache[nm] = z.clone()
                if patch and patch[0] == 'l1fac':
                    z = z.clone()
                    z[:, patch[1]] = patch[2][nm][:, patch[1]]
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q, 'q1'), qk(a.c_k, 'k1')
        q2, k2 = qk(a.c_q2, 'q2'), qk(a.c_k2, 'k2')
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if li == 0:
            my_cache['yh'] = yh4.clone()
            if patch and patch[0] == 'yh':
                yh4 = yh4.clone()
                yh4[:, patch[1]] = patch[2]['yh'][:, patch[1]]
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (x.size(-1),)))
        if li == 0:
            my_cache['mo'] = mo.clone()
            if patch and patch[0] == 'mo':
                mo = mo.clone()
                mo[:, patch[1]] = patch[2]['mo'][:, patch[1]]
        x = x + mo
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30), my_cache


traces = []
for e in ex:
    ctx = None
    for pkt in pk:
        if pkt['cluster'] == e['cluster']:
            ctx = pkt['samples'][0]['context'][-60:]
    loc = None
    for di, dtext in enumerate(dec_cache):
        if ctx and ctx in dtext:
            seq = docs[di]
            acc = ''
            for p in range(len(seq) - 1):
                acc += tok.decode([seq[p]])
                if acc.endswith(ctx):
                    loc = (di, p)
                    break
            break
    if loc is None:
        continue
    di, p = loc
    tgt = SEQS[di, p + 1]
    key_off = e['key_tokens'][0]['offset']
    j = p - key_off
    idx_c = SEQS[di:di + 1, :-1].clone().to(DEV)
    idx_x = idx_c.clone()
    idx_x[0, j] = NEUTRAL
    lg_c, cache_c = run_cached(idx_c)
    lg_x, _ = run_cached(idx_x)
    lp_c = float(F.log_softmax(lg_c[0, p].float(), 0)[tgt])
    lp_x = float(F.log_softmax(lg_x[0, p].float(), 0)[tgt])
    row = {'cluster': e['cluster'], 'target': e['target'],
           'clean_lp': round(lp_c, 2), 'corrupt_lp': round(lp_x, 2), 'recovery': {}}
    span = list(range(j, p + 1))
    for site in ('yh', 'mo', 'l1fac'):
        for posname, positions in (('key_pos', [j]), ('span', span)):
            lg_p, _ = run_cached(idx_x, patch=(site, positions, cache_c))
            lp_p = float(F.log_softmax(lg_p[0, p].float(), 0)[tgt])
            rec = (lp_p - lp_x) / max(lp_c - lp_x, 1e-6)
            row['recovery'][f'{site}@{posname}'] = round(rec, 2)
    traces.append(row)
    print(f"cluster {e['cluster']} {e['target']!r}: clean {lp_c:.2f} corrupt {lp_x:.2f} "
          f"| recovery {row['recovery']}", flush=True)
out['traces'] = traces
json.dump(out, open(f'{QK}/qk_patch_program.json', 'w'), indent=2)
print('PATCH PROGRAM DONE', flush=True)
