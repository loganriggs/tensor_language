"""Flagship confirmation of F13's INTERPRETABLE claim: does bilin18's layer-1 (h[1])
query/key selection run on the bilinear (mlp) output, not the embedding or block-0
attention output? (Logan 2026-07-21). bilin18 has per-head QK RMSNorm (nonlinear), so the
exact bilinear block decomposition doesn't transfer; use causal SOURCE ABLATION instead.

h[1]'s QK reads rms_norm(xin1), xin1 = E + A + M:
  A = l1_0 * attn_write0      (block-0 attention output)
  M = l1_0 * mlp_write0       (block-0 bilinear/mlp output)
  E = xin1 - A - M            (embedding-path contribution)
Ablate each source from the QK INPUT only (keep V from the full input), recompute h[1]'s
pattern, run the rest of the model, measure ΔCE. Which source's removal hurts selection most?
Gate: E+A+M == xin1; full (no ablation) ΔCE = 0.
"""
import json, sys
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
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


@torch.no_grad()
def forward(remove=None):
    """remove in {None,'E','A','M'} ablates that source from h[1]'s QK input only."""
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x           # embedding is RMS-normed first (tier2_model)
    store = {}; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x                                        # residual entering this block's attn
        a = blk.attn
        def qkf(lin, hh):
            return apply_rot(F.rms_norm(lin(hh).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        h_qk = rms(xin)
        if li == 1 and remove is not None:
            A = blk.lambdas[0] * store['attn0']        # l1_0 = blk.lambdas[0]
            M = blk.lambdas[0] * store['mlp0']
            E = xin - A - M
            src = {'E': E, 'A': A, 'M': M}[remove]
            h_qk = rms(xin - src)
        h_v = rms(xin)
        q, k = qkf(a.c_q, h_qk), qkf(a.c_k, h_qk)
        q2, k2 = qkf(a.c_q2, h_qk), qkf(a.c_k2, h_qk)
        v = a.c_v(h_v).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)   # value bus mixing (tier2_model L87-89)
        s1 = torch.einsum('bqnh,bknh->bnqk', q, k) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn_out = a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = xin + attn_out
        mlp_out = blk.mlp(rms(x))
        x = x + mlp_out
        if li == 0:
            store['attn0'] = attn_out; store['mlp0'] = mlp_out
    xf = rms(x)
    return 30 * torch.tanh(m.lm_head(xf) / 30)


@torch.no_grad()
def ce(remove=None):
    lg = forward(remove).float()
    return F.cross_entropy(lg.reshape(-1, Vsz), TGT.reshape(-1)).item()


# gate: E+A+M == xin1 (reconstruct once, checking the decomposition)
@torch.no_grad()
def gate_decomp():
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    x = rms(m.transformer.wte(IDX)); x0 = x
    store = {}
    for li, blk in enumerate(m.transformer.h[:2]):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x
        if li == 1:
            A = blk.lambdas[0] * store['attn0']; M = blk.lambdas[0] * store['mlp0']
            E = xin - A - M
            return (E + A + M - xin).abs().max().item()
        a = blk.attn
        cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
        h = rms(xin)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(Bt, T, NH, HD)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        store['attn0'] = x - xin
        mo = blk.mlp(rms(x)); x = x + mo; store['mlp0'] = mo


from tier2_model import reference_forward
with torch.no_grad():
    lg_ref = reference_forward(m, IDX, 'bf16').float()
    CEref = F.cross_entropy(lg_ref.reshape(-1, Vsz), TGT.reshape(-1)).item()
g = gate_decomp()
CE0 = ce(None)
fwd_gate = abs(CE0 - CEref)
print(f'GATE E+A+M==xin1: {g:.2e}; forward vs reference CE: {CE0:.4f} vs {CEref:.4f} '
      f'(Δ {fwd_gate:.2e})', flush=True)
assert fwd_gate < 0.05, f'inline forward does not match reference ({CE0} vs {CEref})'
res = {'gate': g, 'baseline_ce': round(CE0, 4), 'ablate_dce': {}}
print('ablate each source from h[1] QK input, ΔCE (higher = more needed for selection):', flush=True)
for src in ['E', 'A', 'M']:
    d = ce(src) - CE0
    res['ablate_dce'][src] = round(d, 4)
    lab = {'E': 'embedding', 'A': 'block-0 attn out', 'M': 'block-0 mlp (bilinear) out'}[src]
    print(f'  remove {src} ({lab}): ΔCE {d:+.4f}', flush=True)
json.dump(res, open(f'{OUT}/bilin18_qk1_sources.json', 'w'), indent=2)
top = max(res['ablate_dce'], key=res['ablate_dce'].get)
print(f"\nmost-needed source for layer-1 selection: {top} "
      f"({'CONFIRMS F13 (bilinear/mlp output)' if top=='M' else 'DIFFERS from toy F13'})", flush=True)
print('bilin18 qk1 sources done', flush=True)
