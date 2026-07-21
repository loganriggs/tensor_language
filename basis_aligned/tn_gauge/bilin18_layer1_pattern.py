"""What does the special layer-1 selection DO? (Logan 2026-07-21, interpretive payoff of
F13-F19). F13/F18 established layer-1 (h[1]) query/key selects strongly on the block-0
bilinear output. Characterize the MECHANISM: capture h[1]'s attention pattern over eval
tokens and measure where read-weight goes by relative offset (q-k). Local / previous-token /
diffuse? The attention is UNNORMALIZED bilinear (pat = s1*s2, can be negative), so use
|pat| normalized per query as the read-weight distribution. Also report the per-head peak-
offset distribution. Gate: forward = reference exactly. Per head (9 heads).
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
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
AUD = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = AUD[:, :-1].to(DEV); TGT = AUD[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
LAYER = 1


@torch.no_grad()
def forward(capture=False):
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x
    v1 = None; cap = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if capture and li == LAYER:
            cap = pat.clone()                              # (B, NH, Tq, Tk)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))
    return 30 * torch.tanh(m.lm_head(rms(x)) / 30), cap


with torch.no_grad():
    lg, PAT = forward(capture=True)
    CE0 = F.cross_entropy(lg.float().reshape(-1, Vsz), TGT.reshape(-1)).item()
    CEref = F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()
assert abs(CE0 - CEref) < 0.05
print(f'gate forward=reference: {CE0:.4f} vs {CEref:.4f}; layer-{LAYER} pattern captured {tuple(PAT.shape)}', flush=True)

# read-weight distribution by relative offset (q-k), per head, |pat| normalized per query
A = PAT.abs()                                              # (B,NH,Tq,Tk)
qidx = torch.arange(T, device=DEV)[:, None]
kidx = torch.arange(T, device=DEV)[None, :]
off = (qidx - kidx)                                        # (Tq,Tk), >=0 in causal region
res = {'baseline_ce': round(CE0, 4), 'layer': LAYER, 'per_head_peak_offset': {}, 'offset_profile': {}}
# aggregate over batch+query (query>=8 to avoid boundary), normalized per query
norm = A / A.sum(-1, keepdim=True).clamp_min(1e-9)         # per-query read distribution
print('\nper-head read-weight by relative offset (mean over queries):', flush=True)
print('  head | off0(self) | off1(prev) | off2 | off3-8 | off>8 | peak@1 %', flush=True)
for hh in range(NH):
    nh = norm[:, hh, 8:, :]                                # (B, Tq', Tk)
    o = off[8:, :]
    def band(lo, hi):
        mm = (o >= lo) & (o <= hi)
        return (nh * mm[None].float()).sum().item() / (nh.shape[0] * nh.shape[1])
    b0, b1, b2, b38, bhi = band(0, 0), band(1, 1), band(2, 2), band(3, 8), band(9, T)
    # peak offset per query
    peak = nh.argmax(-1)                                   # (B,Tq') key index
    qpos = torch.arange(8, T, device=DEV)[None, :]
    peakoff = (qpos - peak)
    pk1 = (peakoff == 1).float().mean().item()
    res['per_head_peak_offset'][hh] = {'off0': round(b0, 3), 'off1': round(b1, 3),
                                       'off2': round(b2, 3), 'off3_8': round(b38, 3),
                                       'off_gt8': round(bhi, 3), 'peak_at_prev_frac': round(pk1, 3)}
    print(f'   {hh}   |   {b0:.3f}    |   {b1:.3f}    | {b2:.3f}| {b38:.3f} | {bhi:.3f} |  {pk1:.2f}', flush=True)

# overall mechanism summary
allnorm = norm[:, :, 8:, :]
o = off[8:, :]
Nq = allnorm.shape[0] * allnorm.shape[1] * allnorm.shape[2]   # B * NH * Tq'
prof = {f'off{d}': round((allnorm * (o == d)[None, None].float()).sum().item() / Nq, 4)
        for d in range(0, 9)}
prof['off_gt8'] = round((allnorm * (o > 8)[None, None].float()).sum().item() / Nq, 4)
res['offset_profile'] = prof
loc = prof['off0'] + prof['off1'] + prof['off2']
json.dump(res, open(f'{OUT}/bilin18_layer1_pattern.json', 'w'), indent=2)
res['local_le2'] = round(loc, 3)
print(f'\noverall read-weight within offset<=2 (local): {loc:.3f}; offset>8 (long-range): {prof["off_gt8"]:.3f}', flush=True)
print(f'mechanism: {"LOCAL/previous-token-ish" if loc > 0.4 else "predominantly DIFFUSE/long-range with a few local heads"}', flush=True)
print('bilin18 layer1 pattern done', flush=True)
