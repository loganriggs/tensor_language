"""TICK 239: key-side enrichment at scale (standalone). See LOG for design."""
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
docs = [SEQS[i].tolist() for i in range(128)]
dec_cache = [tok.decode(d) for d in docs]


@torch.no_grad()
def run_p(idx, patch=None):
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


# locate positions + find each's top key token (single substitution batch)
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
print(f'{len(POS)} positions', flush=True)
from tier2_model import reference_forward
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
sel = [(POS[i][0], POS[i][1], KEYJ[i]) for i in range(len(POS)) if KEYJ[i] is not None]
print(f'{len(sel)} with strong keys', flush=True)

LAYERS = [2, 5, 8, 11, 14]
agg = {f'restore_L{L}': [] for L in LAYERS}
agg['late_insert_damage'] = []
agg['mlp_key_5_11'] = []
agg['attn_key_5_11'] = []
for (di, p, j) in sel[:64]:
    tgt = int(SEQS[di, p + 1])
    idx_c = SEQS[di:di + 1, :-1].clone().to(DEV)
    idx_x = idx_c.clone()
    idx_x[0, j] = NEUTRAL
    lg_c, cc = run_p(idx_c)
    lg_x, cx = run_p(idx_x)
    lp_c = float(F.log_softmax(lg_c[0, p].float(), 0)[tgt])
    lp_x = float(F.log_softmax(lg_x[0, p].float(), 0)[tgt])
    den = max(lp_c - lp_x, 1e-6)
    if den < 1.0:
        continue
    for L in LAYERS:
        lg2, _ = run_p(idx_x, patch=('resid', L, L, [j], cc))
        agg[f'restore_L{L}'].append((float(F.log_softmax(lg2[0, p].float(), 0)[tgt]) - lp_x) / den)
    lg2, _ = run_p(idx_c, patch=('resid', 13, 17, [j], cx))
    agg['late_insert_damage'].append((lp_c - float(F.log_softmax(lg2[0, p].float(), 0)[tgt])) / den)
    for site, key in (('mlp_out', 'mlp_key_5_11'), ('attn_out', 'attn_key_5_11')):
        lg2, _ = run_p(idx_x, patch=(site, 5, 11, [j], cc))
        agg[key].append((float(F.log_softmax(lg2[0, p].float(), 0)[tgt]) - lp_x) / den)
out = {k: {'mean': round(float(np.mean(v)), 3), 'median': round(float(np.median(v)), 3),
           'n': len(v)} for k, v in agg.items() if v}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(f'{QK}/qk_enrich_scale.json', 'w'), indent=2)
print('ENRICH SCALE DONE', flush=True)
