"""TICK 260: layer-5 head-7 identity test — is the giant head the compound-key aggregator?

Tick 240 showed the enriched key-position state is context-bound. Probe: find each
failure's primary key token j and secondary key token j2 (both >1 nat by neutral
substitution). Corrupt ONLY j2; then restore the primary key position j's clean
residual at a single layer L in {2,5,8,11,14} (and the transport band 13-17). If
j2's content is absorbed into j's residual by depth L, the restore rescues the
prediction. The recovery-versus-depth curve is the compound-binding depth profile.
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
    C = {'resid': {}}
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
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30), C


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
PAIRS = []
for (di, p) in POS:
    tgt = SEQS[di, p + 1]
    idx = SEQS[di:di + 1, :-1]
    offs = list(range(1, min(16, p + 1)))
    variants = [idx.clone()]
    for off in offs:
        v = idx.clone()
        v[0, p - off] = NEUTRAL
        variants.append(v)
    with torch.no_grad():
        lg = reference_forward(m, torch.cat(variants, 0).to(DEV), 'bf16')[:, p].float()
        lps = F.log_softmax(lg, 1)[:, tgt]
    drops = sorted([(off, float(lps[0] - lps[1 + i])) for i, off in enumerate(offs)],
                   key=lambda x: -x[1])
    if len(drops) >= 2 and drops[0][1] > 1.0 and drops[1][1] > 0.5:
        PAIRS.append((di, p, p - drops[0][0], p - drops[1][0]))
print(f'{len(PAIRS)} with two strong keys', flush=True)


@torch.no_grad()
def run_ab(idx, kill=None):
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
        if kill and kill[1] == li:
            s1 = s1.clone()
            if kill[0] == 'head':
                s1[:, kill[2]] = 0
            else:
                s1[:] = 0
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)

SING = []
for (di, p) in POS:
    if any(pp[0] == di and pp[1] == p for pp in PAIRS):
        continue
    SING.append((di, p))
groups = {'compound': [(di, p) for (di, p, j, j2) in PAIRS][:32], 'single': SING[:32]}
agg = {}
for gname, plist in groups.items():
    for cname, kill in (('h7', ('head', 5, 7)), ('layer5', ('layer', 5))):
        drops = []
        for (di, p) in plist:
            tgt = int(SEQS[di, p + 1])
            idx = SEQS[di:di + 1, :-1].to(DEV)
            lp_c = float(F.log_softmax(run_ab(idx)[0, p].float(), 0)[tgt])
            lp_a = float(F.log_softmax(run_ab(idx, kill)[0, p].float(), 0)[tgt])
            drops.append(lp_c - lp_a)
        agg[f'{gname}_{cname}'] = {'mean': round(float(np.mean(drops)), 3),
                                   'median': round(float(np.median(drops)), 3),
                                   'n': len(drops)}
print(json.dumps(agg, indent=1), flush=True)
json.dump(agg, open(f'{QK}/qk_l5h7_identity.json', 'w'), indent=2)
print('L5H7 DONE', flush=True)
