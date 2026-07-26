"""TICK 238: execute the adversarial agents' patch specs (deduped batch)."""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
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
ka = json.load(open(f'{QK}/qk_key_ablation.json'))
docs = [SEQS[i].tolist() for i in range(128)]
dec_cache = [tok.decode(d) for d in docs]

LOCS = {}
for e in ka:
    for pkt in pk:
        if pkt['cluster'] == e['cluster']:
            ctx = pkt['samples'][0]['context'][-60:]
    for di, dtext in enumerate(dec_cache):
        if ctx in dtext:
            seq = docs[di]
            acc = ''
            for p in range(len(seq) - 1):
                acc += tok.decode([seq[p]])
                if acc.endswith(ctx):
                    LOCS[e['cluster']] = (di, p, p - e['key_tokens'][0]['offset'])
                    break
            break
print('located:', LOCS.keys(), flush=True)


@torch.no_grad()
def run(idx, patch=None, caches=None):
    """patch=(site, l0, l1, positions, src_caches). Returns logits and caches."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    C = {'attn_out': {}, 'mlp_out': {}, 'resid': {}}
    for li, blk in enumerate(m.transformer.h):
        C['resid'][li] = x.clone()
        if patch and patch[0] == 'resid' and patch[1] <= li <= patch[2]:
            x = x.clone()
            x[:, patch[3]] = patch[4]['resid'][li][:, patch[3]]
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
        yo = a.c_proj(yh4.reshape(B, T, -1))
        C['attn_out'][li] = yo.clone()
        if patch and patch[0] == 'attn_out' and patch[1] <= li <= patch[2]:
            yo = yo.clone()
            yo[:, patch[3]] = patch[4]['attn_out'][li][:, patch[3]]
        x = x + yo
        mo = blk.mlp(F.rms_norm(x, (x.size(-1),)))
        C['mlp_out'][li] = mo.clone()
        if patch and patch[0] == 'mlp_out' and patch[1] <= li <= patch[2]:
            mo = mo.clone()
            mo[:, patch[3]] = patch[4]['mlp_out'][li][:, patch[3]]
        x = x + mo
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30), C


SPECS = [
    (5, 'attn_out', 4, 12, 'span', 'restore'),
    (5, 'mlp_out', 4, 12, 'span', 'restore'),
    (7, 'attn_out', 4, 12, 'key_pos', 'insert'),
    (7, 'mlp_out', 4, 12, 'target_pos', 'restore'),
    (5, 'attn_out', 7, 13, 'target_pos', 'restore'),
    (5, 'mlp_out', 7, 13, 'target_pos', 'restore'),
    (7, 'attn_out', 7, 13, 'target_pos', 'restore'),
    (7, 'mlp_out', 7, 13, 'target_pos', 'restore'),
    (5, 'attn_out', 6, 12, 'target_pos', 'restore'),
    (5, 'attn_out', 6, 8, 'target_pos', 'restore'),
    (7, 'attn_out', 7, 12, 'target_pos', 'restore'),
    (6, 'attn_out', 6, 12, 'target_pos', 'insert'),
    (5, 'resid', 8, 8, 'key_pos', 'restore'),
    (5, 'resid', 13, 17, 'key_pos', 'insert'),
    (2, 'resid', 2, 17, 'key_pos', 'insert'),
]
results = []
state = {}
for ci in set(s[0] for s in SPECS):
    if ci not in LOCS:
        continue
    di, p, j = LOCS[ci]
    tgt = int(SEQS[di, p + 1])
    idx_c = SEQS[di:di + 1, :-1].clone().to(DEV)
    idx_x = idx_c.clone()
    idx_x[0, j] = NEUTRAL
    lg_c, cc = run(idx_c)
    lg_x, cx = run(idx_x)
    lp_c = float(F.log_softmax(lg_c[0, p].float(), 0)[tgt])
    lp_x = float(F.log_softmax(lg_x[0, p].float(), 0)[tgt])
    state[ci] = (di, p, j, tgt, idx_c, idx_x, cc, cx, lp_c, lp_x)
for (ci, site, l0, l1, posname, direction) in SPECS:
    if ci not in state:
        results.append({'spec': [ci, site, l0, l1, posname, direction], 'result': None})
        continue
    di, p, j, tgt, idx_c, idx_x, cc, cx, lp_c, lp_x = state[ci]
    positions = {'key_pos': [j], 'target_pos': [p], 'span': list(range(j, p + 1))}[posname]
    if direction == 'restore':
        lg, _ = run(idx_x, patch=(site, l0, l1, positions, cc))
        lp = float(F.log_softmax(lg[0, p].float(), 0)[tgt])
        val = (lp - lp_x) / max(lp_c - lp_x, 1e-6)
        kind = 'recovery'
    else:
        lg, _ = run(idx_c, patch=(site, l0, l1, positions, cx))
        lp = float(F.log_softmax(lg[0, p].float(), 0)[tgt])
        val = (lp_c - lp) / max(lp_c - lp_x, 1e-6)
        kind = 'damage'
    results.append({'spec': [ci, site, l0, l1, posname, direction],
                    kind: round(val, 2)})
    print(f'cluster {ci} {site} L{l0}-{l1} {posname} {direction}: {kind} {val:.2f}',
          flush=True)
json.dump(results, open(f'{QK}/qk_spec_results.json', 'w'), indent=1)
print('SPEC EXECUTOR DONE', flush=True)
