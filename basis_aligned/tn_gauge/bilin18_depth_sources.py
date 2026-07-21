"""Does 'selection runs on the immediately-preceding bilinear output' hold ACROSS DEPTH?
(Logan 2026-07-21, bounded generalization of F18). F18 showed bilin18's h[1] query/key
selects on block-0's bilinear (mlp) output (remove it -> +0.68 ΔCE) not its attention
output. Test whether each block's attention selects on the PRECEDING block's mlp write vs
its attn write, across depth. For target block L, ablate from L's QK input only the write
of block L-1 (mlp or attn), entering with coefficient lambdas_L[0]; ΔCE. Gate: forward =
reference exactly. Reuses the value-bus-mixing + embedding-RMSNorm-correct forward from F18.
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
NL = len(m.transformer.h)
Vsz = cfg['vocab_size']
AUD = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = AUD[:, :-1].to(DEV); TGT = AUD[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
TARGETS = [1, 2, 3, 6, 9, 12, 17]


@torch.no_grad()
def forward(target=None, remove=None):
    """ablate block(target-1)'s `remove` write ('mlp'|'attn') from block target's QK input."""
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x
    v1 = None; prev = {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x
        a = blk.attn
        def qkf(lin, hh):
            return apply_rot(F.rms_norm(lin(hh).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        h_qk = rms(xin)
        if target is not None and li == target and remove is not None:
            src = blk.lambdas[0] * prev[remove]        # block li-1's write, as it enters xin
            h_qk = rms(xin - src)
        h_v = rms(xin)
        q, k = qkf(a.c_q, h_qk), qkf(a.c_k, h_qk)
        q2, k2 = qkf(a.c_q2, h_qk), qkf(a.c_k2, h_qk)
        v = a.c_v(h_v).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', q, k) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn_out = a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = xin + attn_out
        mlp_out = blk.mlp(rms(x))
        x = x + mlp_out
        prev = {'attn': attn_out, 'mlp': mlp_out}       # this block's writes -> for next block
    return 30 * torch.tanh(m.lm_head(rms(x)) / 30)


@torch.no_grad()
def ce(**kw):
    return F.cross_entropy(forward(**kw).float().reshape(-1, Vsz), TGT.reshape(-1)).item()


with torch.no_grad():
    CEref = F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()
CE0 = ce()
assert abs(CE0 - CEref) < 0.05, f'forward != reference ({CE0} vs {CEref})'
print(f'gate forward=reference: {CE0:.4f} vs {CEref:.4f} (Δ {abs(CE0-CEref):.1e})', flush=True)
res = {'baseline_ce': round(CE0, 4), 'depth': {}}
print('block L: ablate block(L-1) mlp vs attn write from L QK input, ΔCE:', flush=True)
print('  L | remove preceding MLP | remove preceding ATTN', flush=True)
for L in TARGETS:
    dm = ce(target=L, remove='mlp') - CE0
    da = ce(target=L, remove='attn') - CE0
    res['depth'][L] = {'remove_prev_mlp': round(dm, 4), 'remove_prev_attn': round(da, 4)}
    print(f'  {L:2d} |   {dm:+.4f}            |   {da:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_depth_sources.json', 'w'), indent=2)
mlp_bigger = sum(1 for L in TARGETS if res['depth'][L]['remove_prev_mlp'] > res['depth'][L]['remove_prev_attn'])
res['mlp_dominates_at_n_of_layers'] = f'{mlp_bigger}/{len(TARGETS)}'
json.dump(res, open(f'{OUT}/bilin18_depth_sources.json', 'w'), indent=2)
print(f'\npreceding-MLP matters more than preceding-ATTN at {mlp_bigger}/{len(TARGETS)} probed layers', flush=True)
print('bilin18 depth sources done', flush=True)
