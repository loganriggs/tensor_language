"""TICK 263 (reviewer-2): is "context-bound" really "affinely misaligned"?

Tick 240 cross-context transplant failed (median 0.04 vs self 1.0). Objection: donor
and target contexts may differ by a global coordinate frame that a small affine map
would fix. Test: fit a per-layer global affine (W, b): target_resid ~ W @ donor_resid
+ b on a DISJOINT training set of same-token occurrence pairs (token appears in two
different documents), then apply it to held-out failure positions' donor vectors and
re-measure transplant recovery. If affine-corrected transplants jump toward 1.0,
context-boundedness was a frame artifact; if they stay near 0.04, it holds.
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
BAND = [13, 14, 15, 16, 17]


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
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30), C


# ---- training pairs: same token in two different docs, residuals at BAND ----
print('collecting same-token occurrence pairs...', flush=True)
caches = {}
for di in range(128):
    _, c = run_p(SEQS[di:di + 1, :-1].to(DEV))
    caches[di] = {L: c['resid'][L][0].cpu() for L in BAND}
# map token -> list of (doc, pos) at pos in [32, T-2]
from collections import defaultdict
occ = defaultdict(list)
T = SEQS.shape[1]
for di in range(128):
    for p in range(32, T - 1):
        occ[int(SEQS[di, p])].append((di, p))
pairs = []
for w, lst in occ.items():
    if len(lst) >= 2:
        for a_i in range(0, min(len(lst), 6) - 1, 2):
            (d1, p1), (d2, p2) = lst[a_i], lst[a_i + 1]
            if d1 != d2:
                pairs.append((d1, p1, d2, p2))
print(f'{len(pairs)} training pairs', flush=True)
# fit per-layer ridge affine W,b: target ~ W@donor + b
Wmap = {}
for L in BAND:
    Dn = torch.stack([caches[d1][L][p1] for (d1, p1, d2, p2) in pairs]).float().to(DEV)
    Tg = torch.stack([caches[d2][L][p2] for (d1, p1, d2, p2) in pairs]).float().to(DEV)
    Dn1 = torch.cat([Dn, torch.ones(len(Dn), 1, device=DEV)], 1)
    A = Dn1.T @ Dn1 + 1.0 * torch.eye(D + 1, device=DEV)
    Wmap[L] = torch.linalg.solve(A, Dn1.T @ Tg)          # (D+1, D)
    r2 = 1 - float((Dn1 @ Wmap[L] - Tg).pow(2).sum() / (Tg - Tg.mean(0)).pow(2).sum())
    print(f'layer {L} affine fit R2 {r2:.3f}', flush=True)


def apply_affine(vec, L):
    v1 = torch.cat([vec, torch.ones(1, device=DEV)])
    return v1 @ Wmap[L]


# ---- held-out failure positions (same locate as tick 240) ----
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
agg = {'self': [], 'raw_donor': [], 'affine_donor': []}
for (di, p) in POS:
    tgt = SEQS[di, p + 1]
    idx = SEQS[di:di + 1, :-1]
    offs = list(range(0, min(16, p + 1)))
    variants = [idx.clone()]
    for off in offs:
        v = idx.clone(); v[0, p - off] = NEUTRAL; variants.append(v)
    with torch.no_grad():
        lg = reference_forward(m, torch.cat(variants, 0).to(DEV), 'bf16')[:, p].float()
        lps = F.log_softmax(lg, 1)[:, tgt]
    drops = sorted([(off, float(lps[0] - lps[1 + i])) for i, off in enumerate(offs)],
                   key=lambda x: -x[1])
    if drops[0][1] <= 1.0:
        continue
    j = p - drops[0][0]
    w = int(SEQS[di, j])
    donor = None
    for (d2, p2) in occ[w]:
        if d2 != di:
            donor = (d2, p2); break
    if donor is None:
        continue
    tgt = int(SEQS[di, p + 1])
    idx_c = SEQS[di:di + 1, :-1].to(DEV)
    idx_x = idx_c.clone(); idx_x[0, j] = NEUTRAL
    lg_c, cc = run_p(idx_c)
    lg_x, _ = run_p(idx_x)
    lp_c = float(F.log_softmax(lg_c[0, p].float(), 0)[tgt])
    lp_x = float(F.log_softmax(lg_x[0, p].float(), 0)[tgt])
    den = lp_c - lp_x
    if den < 1.0:
        continue
    d2, p2 = donor

    def rec(vecs):
        lg2, _ = run_p(idx_x, patch=('resid_vec', 13, 17, [j], vecs))
        return (float(F.log_softmax(lg2[0, p].float(), 0)[tgt]) - lp_x) / den
    agg['self'].append(rec({L: cc['resid'][L][0, j] for L in BAND}))
    agg['raw_donor'].append(rec({L: caches[d2][L][p2].to(DEV) for L in BAND}))
    agg['affine_donor'].append(rec({L: apply_affine(caches[d2][L][p2].to(DEV), L) for L in BAND}))
out = {k: {'mean': round(float(np.mean(v)), 3), 'median': round(float(np.median(v)), 3),
           'n': len(v)} for k, v in agg.items() if v}
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(f'{QK}/qk_affine_transplant.json', 'w'), indent=2)
print('AFFINE TRANSPLANT DONE', flush=True)
