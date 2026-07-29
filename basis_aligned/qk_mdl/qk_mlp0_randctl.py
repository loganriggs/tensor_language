"""RED-TEAM FIX #6: RANDOM-FEATURE CONTROL. Same as the MLP0 program fit but with the feature
directions A FROZEN AT RANDOM INIT (only U trained). If random-A/trained-U approaches the trained
program, the "named quadratic features" credit is partly a random-features regressor effect.
Original docstring: TN-native explicit-program substitution for MLP 0 (Logan: which methods rely on the tensor
network?). The bilinear MLP folds EXACTLY to a symmetric 3-tensor; a rank-R symmetric CP of that
tensor, fit in function space, is the explicit program
    MLP0(x) ~ TokenTable[token] + sum_r u_r * (a_r . x)^2
= R named quadratic features (directions a_r), squared, scaled out -- the interaction-basis method
proven on the ECG model, now applied to bilin18's MLP0. Arms: table-only; table + R in {64, 256};
R=256 alone. Fit by MSE on cooc activations (weight-faithful; CE-polish previously shown to add
zero). Verify by SUBSTITUTION: replace MLP0's output in the full model, audit dCE on the ledger
subset; understood fraction = 1 - dCE/floor (floor 3.63).
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[:200]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
FLOOR = 3.62671  # ledger mean-input floor for MLP0


@torch.no_grad()
def block0_pairs(idx):
    """(hin, out, token) for block-0 MLP."""
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    blk = m.transformer.h[0]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD)
    q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
    pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
    x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
    return hin.reshape(-1, D), blk.mlp(hin).reshape(-1, D), idx.reshape(-1)

H, Y, TOK = [], [], []
for i in range(0, 400, 8):
    h, y, t = block0_pairs(COOC[i:i+8].to(DEV)[:, :128]); H.append(h); Y.append(y); TOK.append(t)
H = torch.cat(H); Y = torch.cat(Y); TOK = torch.cat(TOK)
print(f"pairs {H.shape[0]}", flush=True)
# token table (conditional mean, shrink to global for rare)
gmean = Y.mean(0)
tsum = torch.zeros(V, D, device=DEV); tcnt = torch.zeros(V, device=DEV)
tsum.index_add_(0, TOK, Y); tcnt.index_add_(0, TOK, torch.ones_like(TOK, dtype=torch.float32))
lam = tcnt.unsqueeze(1) / (tcnt.unsqueeze(1) + 3.0)
TT = lam * (tsum / tcnt.clamp_min(1).unsqueeze(1)) + (1-lam) * gmean

def fit_interaction(target, R, steps=2500, freeze_A=False):
    A = torch.nn.Parameter(torch.randn(R, D, device=DEV) * 0.02)
    U = torch.nn.Parameter(torch.randn(R, D, device=DEV) * 0.02)
    opt = torch.optim.Adam(([U] if freeze_A else [A, U]), lr=3e-3)
    n = H.shape[0]
    for s in range(steps):
        ii = torch.randint(0, n, (8192,), device=DEV)
        f = (H[ii] @ A.T) ** 2                      # (b,R) squared features
        pred = f @ U                                 # (b,D)
        loss = (pred - target[ii]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        f = (H @ A.T)**2; pred = f @ U
        fvu = float((pred - target).pow(2).sum() / (target - target.mean(0)).pow(2).sum())
    return A.detach(), U.detach(), fvu

ARMS = {}
resid = Y - TT[TOK]
for name, tgt, R, fz in [('table_R256', resid, 256, False), ('table_R256_randomA', resid, 256, True)]:
    A, U, fvu = fit_interaction(tgt, R, freeze_A=fz)
    ARMS[name] = (A, U); print(f"{name}: fit FVU {fvu:.3f}", flush=True)


@torch.no_grad()
def audit(arm):
    tot, n = 0.0, 0
    A, U = ARMS[arm]
    for i in range(0, len(FINEWEB), 4):
        full = FINEWEB[i:i+4].to(DEV); idx = full[:, :-1]; B, T = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qkf(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qkf(a.c_q), qkf(a.c_k), qkf(a.c_q2), qkf(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
            if li == 0:
                flat = hin.reshape(-1, D)
                mo = TT[idx.reshape(-1)].view(B, T, D) + (((flat @ A.T)**2) @ U).view(B, T, D)
                x = x + mo.to(x.dtype)
            else:
                x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), full[:, 1:].reshape(-1))
        tot += ce.item()*full[:, 1:].numel(); n += full[:, 1:].numel()
    return tot/n

base = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']
res = {'floor': FLOOR}
for arm in ['table_R256', 'table_R256_randomA']:
    d = audit(arm) - base
    res[arm] = {'dCE': round(d, 5), 'understood_frac': round(1 - d/FLOOR, 3)}
    print(f"{arm}: dCE +{d:.5f} -> understood {1-d/FLOOR:.1%}", flush=True)
json.dump(res, open(f'{QK}/qk_mlp0_randctl.json', 'w'), indent=2)
torch.save({k: v for k, v in ARMS.items() if v[0] is not None}, f'{QK}/qk_mlp0_randctl.pt')
print("QK MLP0 RANDCTL DONE", flush=True)
