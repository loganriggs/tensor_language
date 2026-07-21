"""Cross-term reduction: compress layer-0 OV relative to the layer-1 QK path, data-validated
(Logan 2026-07-21). F30: 18% of layer-1 QK is context (attended tokens via OV). Do the ATTENDED
tokens reduce to equivalence classes for QK-1? Cluster the layer-0 value table V0[token] into K
classes, re-aggregate through the REAL block-0 attention pattern (data validation: only pairs that
actually occur) into the layer-1 QK input, patch layer-1 QK only, ΔCE vs K. Continuous (real values)
is the floor. Compare RAW value clustering vs QK-1-EFFECT clustering (cluster the value's downstream
contribution) — Logan's 'better features than compositions individually'. Decode classes (GPT-2).
Gate: no-cluster path = reference.
"""
import sys, json
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
from transformers import AutoTokenizer
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
Vsz = cfg['vocab_size']
tk = AutoTokenizer.from_pretrained('gpt2')
TOK = build_eval_tokens(n_chunks=16, seq_len=513)[:12]
IDX = TOK[:, :-1].to(DEV); TGT = TOK[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
b0 = m.transformer.h[0]; a0 = b0.attn; b1 = m.transformer.h[1]
mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]


@torch.no_grad()
def block0_pieces(vtable=None):
    """run block 0; if vtable given, use per-token clustered values in the attention aggregation.
    returns layer-1 QK input xin1 (B,T,D)."""
    x = rms(m.transformer.wte(IDX)); x0 = x
    x = b0.lambdas[0] * x + b0.lambdas[1] * x0
    xin = x; h = rms(xin)
    qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
    if vtable is None:
        v = a0.c_v(h).view(Bt, T, NH, HD)
    else:
        v = vtable[IDX].view(Bt, T, NH, HD)               # clustered per-token values
    s1 = torch.einsum('bqnh,bknh->bnqk', qk(a0.c_q), qk(a0.c_k)) / HD
    s2 = torch.einsum('bqnh,bknh->bnqk', qk(a0.c_q2), qk(a0.c_k2)) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)               # REAL block-0 attention (data-validated)
    x = xin + a0.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
    x = x + b0.mlp(rms(x))
    xin1 = b1.lambdas[0] * x + b1.lambdas[1] * x0
    return xin1


@torch.no_grad()
def patch_forward_ce(h_qk_L1):
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn
        hq = h_qk_L1 if li == 1 else rms(xin); hv = rms(xin)
        qkf = lambda lin, hh: apply_rot(F.rms_norm(lin(hh).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        q, k = qkf(a.c_q, hq), qkf(a.c_k, hq); q2, k2 = qkf(a.c_q2, hq), qkf(a.c_k2, hq)
        v = a.c_v(hv).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', q, k) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))
    lg = 30 * torch.tanh(m.lm_head(rms(x)) / 30)
    return F.cross_entropy(lg.float().reshape(-1, Vsz), TGT.reshape(-1)).item()


# per-token layer-0 value table
V0 = a0.c_v(rms(m.transformer.wte.weight.data.float())).float()     # (Vocab, D)
# real layer-1 QK input (gate)
xin1_real = block0_pieces(None)
CE0 = patch_forward_ce(rms(xin1_real))
CEref = F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()
print(f'gate {CE0:.4f} vs ref {CEref:.4f} (Δ {abs(CE0-CEref):.1e})', flush=True)

uniq = torch.unique(IDX)
freq = torch.bincount(IDX.reshape(-1), minlength=Vsz)


def kmeans(X, K, iters=15):
    g = torch.Generator(device='cpu').manual_seed(0)
    C = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = ((X * X).sum(1, True) - 2 * X @ C.T + (C * C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); ct = torch.zeros(K, device=X.device)
        Cn.index_add_(0, a, X); ct.index_add_(0, a, torch.ones(len(X), device=X.device))
        mm = ct > 0; C[mm] = Cn[mm] / ct[mm][:, None]
    return a, C


res = {'baseline_ce': round(CE0, 4), 'gate': round(abs(CE0 - CEref), 5), 'raw': {}, 'qk1_effect': {}}
print('cluster layer-0 VALUE table (attended tokens) into K classes, re-aggregate through REAL '
      'attention, ΔCE on layer-1 QK:', flush=True)


def cluster_and_ce(space_vecs, K):
    """cluster uniq tokens by space_vecs[uniq] into K; build clustered V0 table; ΔCE."""
    a_u, cent_space = kmeans(space_vecs[uniq], K)
    # centroid in VALUE space (average the actual values within each class)
    Vtab = V0.clone()
    for c in range(K):
        members = uniq[a_u == c]
        if len(members) == 0:
            continue
        Vtab[members] = V0[members].mean(0)
    xin1_c = block0_pieces(Vtab)
    return patch_forward_ce(rms(xin1_c)) - CE0, a_u


# QK-1-effect space: value projected through Right-gate reach (a linear proxy for the cross-term's
# read of the value): how the value direction enters the bilinear then QK. Use ‖path‖-weighted value.
Rqk = torch.cat([getattr(b1.attn, n).weight.data.float() for n in ['c_q', 'c_k', 'c_q2', 'c_k2']], 0)
# value -> (as OV output o) enters bilinear via Right(o)/Left(o); its QK-1 reach ~ Rqk@Down@(Right or Left)
reach = (Rqk @ b0.mlp.Down.weight.data.float() @ b0.mlp.Right.weight.data.float())   # (4D, D) value->QK1 (Right branch)
qk1_effect_vecs = V0 @ reach.T                                     # (Vocab, 4D) value's QK-1-effect signature

for K in [16, 64, 256]:
    d_raw, a_raw = cluster_and_ce(V0, K)
    d_eff, _ = cluster_and_ce(qk1_effect_vecs, K)
    res['raw'][K] = round(d_raw, 4); res['qk1_effect'][K] = round(d_eff, 4)
    print(f'  K={K:4d}: raw-value ΔCE {d_raw:+.4f} | QK1-effect ΔCE {d_eff:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_crossterm.json', 'w'), indent=2)

# qualitative: decode the QK-1-effect value classes at K=32
_, a32 = cluster_and_ce(qk1_effect_vecs, 32)
lines = ['# Layer-0 OV value classes relative to layer-1 QK (attended-token equivalence classes)\n']
sizes = torch.bincount(a32, minlength=32)
for c in torch.argsort(sizes, descending=True)[:16].tolist():
    members = uniq[a32 == c]
    top = members[torch.argsort(freq[members], descending=True)][:12]
    lines.append(f'- class {c} ({int(sizes[c])}): ' + ', '.join(repr(tk.decode([t.item()])) for t in top))
open(f'{OUT}/crossterm_value_classes.md', 'w').write('\n'.join(lines) + '\n')
print('\n--- sample QK-1-effect value classes (attended tokens) ---', flush=True)
for c in torch.argsort(sizes, descending=True)[:6].tolist():
    members = uniq[a32 == c]
    top = members[torch.argsort(freq[members], descending=True)][:10]
    print(f'class {c}: ' + ', '.join(repr(tk.decode([t.item()])) for t in top), flush=True)
print('bilin18 crossterm done', flush=True)
