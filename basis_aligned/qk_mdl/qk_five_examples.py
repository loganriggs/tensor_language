"""Five full-context failure examples with input-interaction attribution (Logan).
For each: long context, target, missing channels, and the share of the TRUE layer-1
context correction (on the missing channels) attributable to: emb-only path
(emb x emb through the MLP + direct), attention-direct (no MLP), and the remainder
(attention-through-MLP: emb x attn0 + attn0 x attn0 blocks)."""
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
blk0 = m.transformer.h[0]
a1 = m.transformer.h[1].attn
MAPS = {'q1': a1.c_q, 'k1': a1.c_k, 'q2': a1.c_q2, 'k2': a1.c_k2}

pk = json.load(open(f'{QK}/qk_failure_packets.json'))
# pick 5: clusters 5, 3, 2, 6, 7 first samples
PICK = [(5, 0), (3, 0), (2, 0), (6, 0), (7, 0)]
docs = [SEQS[i].tolist() for i in range(128)]
dec_cache = [tok.decode(d) for d in docs]


@torch.no_grad()
def block1_input_variant(idx, mode):
    """mode: 'full' | 'noattn' (l0 attention zeroed) | 'nomlp' (block-0 MLP zeroed)."""
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    x = blk0.lambdas[0] * x + blk0.lambdas[1] * x0
    a = blk0.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, NH, HD)
        return apply_rot(F.rms_norm(z, (HD,)), cosb, sinb)

    v = a.c_v(hcur).view(B, T, NH, HD)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    if mode == 'noattn':
        pat = pat * 0
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    if mode != 'nomlp':
        x = x + blk0.mlp(F.rms_norm(x, (x.size(-1),)))
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0


# tables (shrunk) for dev computation — reuse saved l1 tables? build quick from mean:
TBL = torch.load(f'{QK}/qk_l1_tables.pt', map_location=DEV)
TABLES = {n: TBL[n].float().to(DEV) for n in ('q1', 'k1', 'q2', 'k2')}

results = []
for ci, si in PICK:
    pkt = [p for p in pk if p['cluster'] == ci][0]
    s = pkt['samples'][si]
    ctx_tail = s['context'][-60:]
    loc = None
    for di, dtext in enumerate(dec_cache):
        if ctx_tail in dtext:
            seq = docs[di]
            acc = ''
            for p in range(len(seq) - 1):
                acc += tok.decode([seq[p]])
                if acc.endswith(ctx_tail):
                    loc = (di, p)
                    break
            break
    if loc is None:
        continue
    di, p = loc
    idx = SEQS[di:di + 1, :-1].to(DEV)
    devs = {}
    for mode in ('full', 'noattn', 'nomlp'):
        xin = block1_input_variant(idx, mode)
        hn = F.rms_norm(xin, (D,))
        row = {}
        for name, lin in MAPS.items():
            fa = F.rms_norm(lin(hn).view(1, -1, NH, HD).float(), (HD,))
            row[name] = fa[0, p] - TABLES[name][int(idx[0, p])]
        devs[mode] = row
    ch = pkt['top_missing_channels'][:2]
    shares = {}
    for chn in ch:
        name, hh = chn.split('_h')
        hh = int(hh)
        d_full = devs['full'][name][hh]
        d_emb = devs['noattn'][name][hh]
        d_nomlp = devs['nomlp'][name][hh]
        n2 = float(d_full.norm()) ** 2
        emb_share = float((d_full @ d_emb)) / max(n2, 1e-9)
        attdir_share = float((d_full @ (d_nomlp - d_emb))) / max(n2, 1e-9)
        rest = 1 - emb_share - attdir_share
        shares[chn] = {'emb_only': round(emb_share, 2),
                       'attn_direct': round(attdir_share, 2),
                       'attn_through_MLP': round(rest, 2)}
    long_ctx = tok.decode(SEQS[di, max(0, p - 150):p + 1].tolist())
    results.append({'cluster': ci, 'dce': s['dce'], 'target': s['target'],
                    'channels': ch, 'shares': shares, 'context': long_ctx})
    print(f"=== cluster {ci} (dCE {s['dce']}) target {s['target']!r} "
          f"channels {ch} ===", flush=True)
    print(f"  shares: {shares}", flush=True)
json.dump(results, open(f'{QK}/qk_five_examples.json', 'w'), indent=1)
print('FIVE EXAMPLES DONE', flush=True)
