"""Task 2 (Logan 2026-07-21 overnight): can WEIGHT-only info identify which part of the block-0
bilinear layer layer-1 query/key reads (the null of the QK∘bilinear composition)? Bilinear
output M = Down(Left(h)⊙Right(h)); hidden unit i contributes along Down[:,i]. Layer-1 QK reads M
via R=[c_q;c_k;c_q2;c_k2]. WEIGHT-ONLY reachability of unit i = ‖R·Down[:,i]‖ (no data). Rank
hidden units by this, KEEP top-k for the QK path (subtract the dropped units' contribution from
the QK input only), ΔCE vs k. Compare weight-only ranking vs activation-aware (‖R Down_i‖·std(h_i))
vs random. If weight-only lets us drop most units near-free -> weights alone find the QK-null part
of the bilinear layer. Held-out ΔCE on eval tokens via a layer-1 QK-input patch.
"""
import json, sys
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
Vsz = cfg['vocab_size']
TOK = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = TOK[:, :-1].to(DEV); TGT = TOK[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
mlp0 = m.transformer.h[0].mlp
DHID = mlp0.Left.weight.shape[0]
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
R = torch.cat([getattr(m.transformer.h[1].attn, n).weight.data.float() for n in NAMES], 0)  # (4D, D)
Down = mlp0.Down.weight.data.float()                     # (D, DHID)
print(f'd_hidden={DHID}; weight-only QK-reachability ‖R·Down[:,i]‖ per bilinear unit', flush=True)


@torch.no_grad()
def capture():
    """block-0 hidden activations, M, and layer-1 QK input xin1; plus lambda1_0."""
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    blk = m.transformer.h[0]
    x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    xin = x; a = blk.attn; h = rms(xin)
    qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
    v = a.c_v(h).view(Bt, T, NH, HD)
    s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
    s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
    hm = rms(x)
    hidden = (mlp0.Left(hm) * mlp0.Right(hm))            # (B,T,DHID) block-0 bilinear hidden
    M = mlp0.Down(hidden)
    x = x + M
    blk1 = m.transformer.h[1]
    xin1 = blk1.lambdas[0] * x + blk1.lambdas[1] * x0     # layer-1 QK input (pre-norm)
    return hidden.reshape(-1, DHID), xin1.reshape(-1, D), float(blk1.lambdas[0].item())


HID, XIN1, lam = capture()
# scores
w_only = (R @ Down).norm(dim=0)                          # (DHID,) ‖R Down[:,i]‖ weight-only
act_std = HID.std(0)                                     # activation std per unit
act_aware = w_only * act_std                             # weight x activation
g = torch.Generator(device='cpu').manual_seed(0)
rand_score = torch.rand(DHID, generator=g).to(DEV)


@torch.no_grad()
def patch_forward(h_qk_L1):
    """full forward with layer-1 QK input replaced by h_qk_L1 (B,T,D)."""
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn
        h_qk = h_qk_L1 if li == 1 else rms(xin)
        h_v = rms(xin)
        qkf = lambda lin, hh: apply_rot(F.rms_norm(lin(hh).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        q, k = qkf(a.c_q, h_qk), qkf(a.c_k, h_qk); q2, k2 = qkf(a.c_q2, h_qk), qkf(a.c_k2, h_qk)
        v = a.c_v(h_v).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', q, k) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))
    return 30 * torch.tanh(m.lm_head(rms(x)) / 30)


def ce_keep(score, k):
    """keep top-k hidden units by score for the QK path; subtract dropped units' M from xin1."""
    keep = torch.topk(score, k).indices
    dropped = torch.ones(DHID, device=DEV, dtype=torch.bool); dropped[keep] = False
    Mdrop = (HID[:, dropped] @ Down[:, dropped].T)        # (Ntok, D) dropped contribution
    xin1_q = (XIN1 - lam * Mdrop).reshape(Bt, T, D)
    return F.cross_entropy(patch_forward(rms(xin1_q)).float().reshape(-1, Vsz), TGT.reshape(-1)).item()


# baseline: full QK input (no drop) via the patch path (gate: matches reference)
CE_full = F.cross_entropy(patch_forward(rms(XIN1.reshape(Bt, T, D))).float().reshape(-1, Vsz), TGT.reshape(-1)).item()
CEref = F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()
print(f'gate patch-forward vs reference: {CE_full:.4f} vs {CEref:.4f} (Δ {abs(CE_full-CEref):.1e})', flush=True)
res = {'d_hidden': DHID, 'baseline_ce': round(CE_full, 4), 'gate': round(abs(CE_full - CEref), 5), 'keep': {}}
print('\nkeep top-k bilinear hidden units for the QK path, ΔCE (weight-only vs activation-aware vs random):', flush=True)
print('  keep_k | weight-only | act-aware | random', flush=True)
for k in [64, 128, 256, 512, 1024, 2048]:
    if k > DHID:
        continue
    d_w = ce_keep(w_only, k) - CE_full
    d_a = ce_keep(act_aware, k) - CE_full
    d_r = ce_keep(rand_score, k) - CE_full
    res['keep'][k] = {'weight_only': round(d_w, 4), 'act_aware': round(d_a, 4), 'random': round(d_r, 4)}
    print(f'  {k:5d}  |   {d_w:+.4f}   |  {d_a:+.4f}  | {d_r:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_qk1_bilinear_null.json', 'w'), indent=2)
print(f'\n(d_hidden={DHID}) weight-only ‖R·Down‖ ranking vs activation-aware for finding the QK-null of the bilinear layer', flush=True)
print('bilin18 qk1 bilinear null done', flush=True)
