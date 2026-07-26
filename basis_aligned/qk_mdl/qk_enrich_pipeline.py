"""TICK 240: inside the enrichment pipeline.

Q1 (which MLPs): restore mlp_out at the key position at SINGLE layers L=0..13 —
per-layer causal share of the enrichment.
Q2 (what is encoded): is the enriched key-side state a context-independent entity
representation? Transplant the key position's residual over the transport band
(layers 13-17) from (a) a minimal synthetic donor context containing the key token,
(b) a different real document containing the key token, into the corrupted run.
Controls: own-clean restore 13-17 (positive, should ~1.0); synthetic donor with
NEUTRAL in place of the key token (negative, should ~0).
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
    """patch = (site, L0, L1, positions, src). site 'resid'/'attn_out'/'mlp_out':
    src is a cache dict from another run at the SAME positions. site 'resid_vec':
    src is {li: (D,) tensor} written at positions (cross-context transplant)."""
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
        if patch and patch[0] == 'resid_vec' and patch[1] <= li <= patch[2]:
            x = x.clone()
            x[:, patch[3]] = patch[4][li].to(x.dtype)
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


# ---- locate positions and their strong keys (same procedure as tick 239) ----
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

DONOR_PRE = tok.encode('The following is an article about')
DONOR_POST = tok.encode(' and other related topics that people often discuss.')
MLP_LAYERS = list(range(14))
agg = {f'mlpL{L}': [] for L in MLP_LAYERS}
for k in ('own_restore_13_17', 'synth_transplant', 'neutral_control', 'real_transplant'):
    agg[k] = []
for (di, p, j) in sel[:48]:
    tgt = int(SEQS[di, p + 1])
    w = int(SEQS[di, j])
    idx_c = SEQS[di:di + 1, :-1].clone().to(DEV)
    idx_x = idx_c.clone()
    idx_x[0, j] = NEUTRAL
    lg_c, cc = run_p(idx_c)
    lg_x, _ = run_p(idx_x)
    lp_c = float(F.log_softmax(lg_c[0, p].float(), 0)[tgt])
    lp_x = float(F.log_softmax(lg_x[0, p].float(), 0)[tgt])
    den = lp_c - lp_x
    if den < 1.0:
        continue

    def rec(lg2):
        return (float(F.log_softmax(lg2[0, p].float(), 0)[tgt]) - lp_x) / den

    # Q1: single-layer mlp_out restore at key position
    for L in MLP_LAYERS:
        lg2, _ = run_p(idx_x, patch=('mlp_out', L, L, [j], cc))
        agg[f'mlpL{L}'].append(rec(lg2))
    # positive control: own clean resid over transport band
    lg2, _ = run_p(idx_x, patch=('resid', 13, 17, [j], cc))
    agg['own_restore_13_17'].append(rec(lg2))
    # Q2a: synthetic donor transplant
    for token_at_key, key_out in ((w, 'synth_transplant'), (NEUTRAL, 'neutral_control')):
        donor = torch.tensor([DONOR_PRE + [token_at_key] + DONOR_POST * 3],
                             device=DEV)
        jd = len(DONOR_PRE)
        _, cd = run_p(donor)
        vecs = {li: cd['resid'][li][0, jd] for li in range(13, 18)}
        lg2, _ = run_p(idx_x, patch=('resid_vec', 13, 17, [j], vecs))
        agg[key_out].append(rec(lg2))
    # Q2b: real donor — the same key token in a different document, position >= 32
    found = None
    for di2 in range(128):
        if di2 == di:
            continue
        hits = (SEQS[di2, 32:2000] == w).nonzero()
        if len(hits):
            found = (di2, int(hits[0]) + 32)
            break
    if found is not None:
        di2, jd2 = found
        _, cd = run_p(SEQS[di2:di2 + 1, :-1].to(DEV))
        vecs = {li: cd['resid'][li][0, jd2] for li in range(13, 18)}
        lg2, _ = run_p(idx_x, patch=('resid_vec', 13, 17, [j], vecs))
        agg['real_transplant'].append(rec(lg2))
out = {k: {'mean': round(float(np.mean(v)), 3), 'median': round(float(np.median(v)), 3),
           'n': len(v)} for k, v in agg.items() if v}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(f'{QK}/qk_enrich_pipeline.json', 'w'), indent=2)
print('ENRICH PIPELINE DONE', flush=True)
