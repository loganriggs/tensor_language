"""TICK 237b: depth-sweep causal tracing. For each of the 4 examples: corrupt the top
key token; then for each layer L in 0..17, restore the CLEAN residual stream at the
key->target span as it enters layer L, and measure logp recovery. The recovery-vs-
depth curve locates where the compositional key's binding/retrieval computation lives.
"""
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


@torch.no_grad()
def forward_resid(idx, restore=None):
    """restore=(layer, positions, clean_resids dict): overwrite x at entry of layer."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    resids = {}
    for li, blk in enumerate(m.transformer.h):
        resids[li] = x.clone()
        if restore and restore[0] == li:
            x = x.clone()
            x[:, restore[1]] = restore[2][li][:, restore[1]]
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
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30), resids


out = []
for e in ka:
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
    j = p - e['key_tokens'][0]['offset']
    idx_c = SEQS[di:di + 1, :-1].clone().to(DEV)
    idx_x = idx_c.clone()
    idx_x[0, j] = NEUTRAL
    lg_c, res_c = forward_resid(idx_c)
    lg_x, _ = forward_resid(idx_x)
    lp_c = float(F.log_softmax(lg_c[0, p].float(), 0)[tgt])
    lp_x = float(F.log_softmax(lg_x[0, p].float(), 0)[tgt])
    span = list(range(j, p + 1))
    curve = []
    for L in range(18):
        lg_p, _ = forward_resid(idx_x, restore=(L, span, res_c))
        lp_p = float(F.log_softmax(lg_p[0, p].float(), 0)[tgt])
        curve.append(round((lp_p - lp_x) / max(lp_c - lp_x, 1e-6), 2))
    out.append({'cluster': e['cluster'], 'target': e['target'], 'curve': curve})
    print(f"cluster {e['cluster']} {e['target']!r}: recovery by restore-layer {curve}",
          flush=True)
json.dump(out, open(f'{QK}/qk_depth_sweep.json', 'w'), indent=1)
print('DEPTH SWEEP DONE', flush=True)
