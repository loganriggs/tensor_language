"""How UNDERSTANDABLE are the explicit MLP programs? Characterize the R=256 quadratic features of
the MLP0 and MLP1 programs: (1) token-keyed vs contextual (R^2 of token-mean predictor per feature);
(2) sparsity of use (participation ratio -> effective features per position); (3) concentration
(keep only top-64-of-256 by importance, re-audit -> is function concentrated in few features);
(4) alignment with already-named objects (embedding principal components; the six category axes);
(5) human-readable: top tokens for the top-importance features.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[:200]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
SUBBASE = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']
FLOOR = {'mlp0': 3.62671, 'mlp1': 2.15118}
wte = m.transformer.wte.weight.detach().float().to(DEV)
EPC = torch.linalg.svd(F.rms_norm(wte, (D,)) - F.rms_norm(wte, (D,)).mean(0), full_matrices=False).Vh[:96]

m0 = torch.load(f'{QK}/qk_mlp0_interaction.pt', map_location=DEV)
m1 = torch.load(f'{QK}/qk_mlp1_interaction.pt', map_location=DEV)
PROG = {'mlp0': m0['table_R256'], 'mlp1': m1['table_prev_R256']}


@torch.no_grad()
def collect(idx, want):
    """hin at block `want` (0 or 1) + token + prev + next-token id."""
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(want + 1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); hin = F.rms_norm(x, (D,))
        if li == want: return hin.reshape(-1, D), idx.reshape(-1)
        x = x + blk.mlp(hin)

res = {}
for name, want in [('mlp0', 0), ('mlp1', 1)]:
    A, U = PROG[name]; R = A.shape[0]
    Hs, Ts = [], []
    for i in range(0, 240, 8):
        h, t = collect(COOC[i:i+8].to(DEV)[:, :128], want); Hs.append(h); Ts.append(t)
    H = torch.cat(Hs); T = torch.cat(Ts)
    Fq = (H @ A.T) ** 2                                     # (n,R) feature activations
    imp = Fq.mean(0) * U.norm(dim=1)                        # importance per feature
    order = imp.argsort(descending=True)
    # (1) token-keyed vs contextual
    tsum = torch.zeros(V, R, device=DEV); tcnt = torch.zeros(V, device=DEV)
    tsum.index_add_(0, T, Fq); tcnt.index_add_(0, T, torch.ones_like(T, dtype=torch.float32))
    tmean = tsum / tcnt.clamp_min(1).unsqueeze(1)
    pred = tmean[T]
    r2_tok = 1 - (Fq - pred).pow(2).sum(0) / (Fq - Fq.mean(0)).pow(2).sum(0)   # per feature
    frac_tokkeyed = float((r2_tok > 0.8).float().mean())
    # (2) sparsity: participation ratio per position
    pr = (Fq.sum(1) ** 2 / Fq.pow(2).sum(1).clamp_min(1e-9)).mean().item()
    # (4) alignment with embedding PCs (max |cos| per feature direction)
    An = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cos_epc = (An @ EPC.T).abs().max(1).values
    # (5) top tokens for top-8 features
    tops = {}
    for r_ in order[:8].tolist():
        good = (tcnt >= 3)
        vals = tmean[:, r_].clone(); vals[~good] = -1e9
        ids = vals.argsort(descending=True)[:8].tolist()
        tops[f'feat{r_}'] = {'importance_rank': int((order == r_).nonzero()[0, 0]),
                             'r2_token': round(float(r2_tok[r_]), 3),
                             'top_tokens': [tok.convert_ids_to_tokens(i) for i in ids]}
    res[name] = {'frac_token_keyed_r2gt0.8': round(frac_tokkeyed, 3),
                 'median_r2_token': round(float(r2_tok.median()), 3),
                 'effective_features_per_position': round(pr, 1),
                 'importance_top64_share': round(float(imp[order[:64]].sum() / imp.sum()), 3),
                 'max_cos_with_embedding_PCs_median': round(float(cos_epc.median()), 3),
                 'top_features': tops}
    print(f"[{name}] token-keyed(R2>0.8): {frac_tokkeyed:.0%} | median R2_token {float(r2_tok.median()):.2f} | "
          f"eff feats/pos {pr:.1f}/{R} | top-64 importance share {float(imp[order[:64]].sum()/imp.sum()):.0%} | "
          f"median max-cos vs embedding PCs {float(cos_epc.median()):.2f}", flush=True)
    for k, v in tops.items():
        print(f"  {k} (rank {v['importance_rank']}, R2tok {v['r2_token']}): {v['top_tokens']}", flush=True)

# (3) concentration: MLP0 program with only top-64-of-256 features, substitution audit
A0, U0 = PROG['mlp0']
Fq_imp = None
H0s, T0s, Y0s = [], [], []
# rebuild token table for mlp0 quickly
@torch.no_grad()
def block0_out(idx):
    h, t = collect(idx, 0)
    return m.transformer.h[0].mlp(h.view(idx.shape[0], -1, D)).reshape(-1, D), t, h
for i in range(0, 240, 8):
    y, t, h = block0_out(COOC[i:i+8].to(DEV)[:, :128]); Y0s.append(y); T0s.append(t); H0s.append(h)
Y0 = torch.cat(Y0s); T0 = torch.cat(T0s); H0 = torch.cat(H0s)
ts = torch.zeros(V, D, device=DEV); tc = torch.zeros(V, device=DEV)
ts.index_add_(0, T0, Y0); tc.index_add_(0, T0, torch.ones_like(T0, dtype=torch.float32))
lam = tc.unsqueeze(1)/(tc.unsqueeze(1)+3.0); TT0 = lam*(ts/tc.clamp_min(1).unsqueeze(1)) + (1-lam)*Y0.mean(0)
imp0 = ((H0 @ A0.T)**2).mean(0) * U0.norm(dim=1)
KEEPF = imp0.argsort(descending=True)[:64]
A64, U64 = A0[KEEPF].contiguous(), U0[KEEPF].contiguous()

@torch.no_grad()
def audit_top64():
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), 4):
        full = FINEWEB[i:i+4].to(DEV); idx = full[:, :-1]; B, Tn = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(Tn, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(Tn, Tn, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qkf(lin): z = F.rms_norm(lin(hcur).view(B, Tn, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, Tn, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qkf(a.c_q), qkf(a.c_k), qkf(a.c_q2), qkf(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh4.reshape(B, Tn, -1)); hin = F.rms_norm(x, (D,))
            if li == 0:
                flat = hin.reshape(-1, D)
                mo = (TT0[idx.reshape(-1)] + ((flat @ A64.T)**2) @ U64).view(B, Tn, D).to(x.dtype)
                x = x + mo
            else:
                x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), full[:, 1:].reshape(-1))
        tot += ce.item()*full[:, 1:].numel(); n += full[:, 1:].numel()
    return tot/n
d64 = audit_top64() - SUBBASE
res['mlp0_top64of256_substitution'] = {'dCE': round(d64, 5), 'understood': round(1 - d64/FLOOR['mlp0'], 3)}
print(f"MLP0 program with only top-64-of-256 features: dCE +{d64:.5f} -> {1-d64/FLOOR['mlp0']:.1%}", flush=True)
json.dump(res, open(f'{QK}/qk_program_features.json', 'w'), indent=2)
print("QK PROGRAM FEATURES DONE", flush=True)
