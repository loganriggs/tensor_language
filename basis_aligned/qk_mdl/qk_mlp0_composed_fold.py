"""Logan's composed fold: express MLP0 analytically in terms of UPSTREAM features (embedding +
attn0 archetypes), not as a black-box-input quadratic.
KEY IDENTITY (verifies Logan's rms-as-TN hunch for quadratic consumers): the bilinear MLP is
homogeneous degree-2, so  MLP(rms_norm(x)) = D * T(x,x) / ||x||^2 + bias  -- rms folds EXACTLY as a
scalar gauge (ratio of two analytic quadratics). GATE 1 checks this to machine precision.
COMPOSITION: at layer 0, x_pre = (lam0+lam1)*e_t + attn0_out(i), where e_t = rms(wte_t) exactly and
attn0_out is the exactly-folded layer-0 attention. Bilinearity splits
  T(x_pre,x_pre) = T(e,e)[token^2] + 2 T(e,a)[token x attn] + T(a,a)[attn^2]     (GATE 2, exact)
Then attn0_out is truncated to the NAMED 144-dim archetype write-basis (c_proj o per-head archetype
value directions): the interpretable composed program is analytic in (token, 144 archetype
activations) with all coefficient tensors from weights. Measures: archetype capture of attn0_out,
substitution dCE of the composed program, and the three blocks' energy/dCE shares.
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
SUBBASE = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']
FLOOR = 3.62671
blk0 = m.transformer.h[0]; mlp0 = blk0.mlp
Lw = mlp0.Left.weight.detach().float(); Rw = mlp0.Right.weight.detach().float()
Dw = mlp0.Down.weight.detach().float(); bias = mlp0.Down_bias.detach().float()

def T_bilin(u, v):
    """T(u,v) symmetric bilinear evaluation, batched (n,D),(n,D)->(n,D)."""
    return 0.5*(((u @ Lw.T) * (v @ Rw.T)) @ Dw.T + ((v @ Lw.T) * (u @ Rw.T)) @ Dw.T)

# archetype write-basis (144 = 9 heads x 16 value dirs, pushed through c_proj)
mh = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
pol0 = torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV); pol4 = torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)
cw = blk0.attn.c_proj.weight.detach().float()      # (D, D)
Wcols = []
for hh in range(NH):
    if hh in (0, 4):
        bb = pol0 if hh == 0 else pol4; Dv = bb[f'h{hh}_v_Dm'].to(DEV); Dv = Dv / Dv.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dv.T @ bb[f'h{hh}_CJ'][:, :16].to(DEV)
    else:
        Pp = mh[f'h{hh}']; Dn_ = Pp['Dm'].to(DEV); Dn_ = Dn_ / Dn_.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Vdir = Dn_[:, 2*HD:].T @ Pp['U'].to(DEV)[:, :16]
    if Vdir.shape[1] < 16: Vdir = torch.cat([Vdir, torch.zeros(HD, 16-Vdir.shape[1], device=DEV)], 1)
    Wcols.append(cw[:, hh*HD:(hh+1)*HD] @ Vdir)     # (D,16) write directions for head hh
WA = torch.cat(Wcols, 1)                            # (D, 144)
QA, _ = torch.linalg.qr(WA)                         # orthonormal basis of the archetype write span
print("archetype write-basis built (144 dims)", flush=True)


@torch.no_grad()
def collect(idx):
    """per-position: x_pre (pre-rms MLP0 input), attn0_out, token; and true MLP0 out."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    x = blk0.lambdas[0]*x0 + blk0.lambdas[1]*x0; a = blk0.attn; hcur = F.rms_norm(x, (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
    v = a.c_v(hcur).view(B, T, NH, HD)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
    pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
    aout = a.c_proj(yh4.reshape(B, T, -1))
    x_pre = x + aout; hin = F.rms_norm(x_pre, (D,))
    ebar = x                                          # (lam0+lam1)*e_t component
    return (x_pre.reshape(-1, D), aout.reshape(-1, D), ebar.reshape(-1, D),
            mlp0(hin).reshape(-1, D), idx.reshape(-1))

xp, ao, eb, ytrue, tk = collect(COOC[:8].to(DEV)[:, :128])
# GATE 1: rms folds as scalar gauge for the quadratic
rho2 = xp.pow(2).sum(1) / D
y_gauge = T_bilin(xp, xp) / rho2.unsqueeze(1) + bias
g1 = float((y_gauge - ytrue).norm() / ytrue.norm())
print(f"GATE 1 (rms = exact scalar gauge for quadratic consumer): rel-err {g1:.2e}", flush=True)
# GATE 2: bilinear block split is exact
y_blocks = (T_bilin(eb, eb) + 2*T_bilin(eb, ao) + T_bilin(ao, ao)) / rho2.unsqueeze(1) + bias
g2 = float((y_blocks - ytrue).norm() / ytrue.norm())
print(f"GATE 2 (token^2 + 2 token*attn + attn^2 split): rel-err {g2:.2e}", flush=True)
# archetype capture of attn0_out
a_hat = (ao @ QA) @ QA.T
cap = float(1 - (ao - a_hat).pow(2).sum() / ao.pow(2).sum())
print(f"archetype write-basis captures {cap:.1%} of attn0_out energy", flush=True)
# block energy shares (of centered output)
blocks = {'token2': T_bilin(eb, eb)/rho2.unsqueeze(1), 'cross': 2*T_bilin(eb, ao)/rho2.unsqueeze(1),
          'attn2': T_bilin(ao, ao)/rho2.unsqueeze(1)}
tot = (ytrue - ytrue.mean(0))
shares = {k: round(float((v - v.mean(0)).pow(2).sum() / tot.pow(2).sum()), 3) for k, v in blocks.items()}
print("block variance shares:", shares, flush=True)


@torch.no_grad()
def audit(mode):
    """substitute MLP0 with the composed analytic program. mode: 'full_analytic' (exact gauge,
    untruncated attn -- must equal model), 'archetype' (attn truncated to 144-dim named basis,
    gauge from truncated input)."""
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B, T2 = idx.shape
        x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = x0; xf = x0; v1 = None
        cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; a = blk.attn
            x = blk.lambdas[0]*x + blk.lambdas[1]*x0 if li else blk.lambdas[0]*x0 + blk.lambdas[1]*x0
            hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T2, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            aout = a.c_proj(yh4.reshape(B, T2, -1)); x = x + aout; hin = F.rms_norm(x, (D,))
            if li == 0:
                ebf = (blk.lambdas[0]+blk.lambdas[1])*x0
                af = aout.reshape(-1, D)
                if mode == 'archetype': af = (af @ QA) @ QA.T
                xpre = ebf.reshape(-1, D) + af
                r2 = xpre.pow(2).sum(1)/D
                mo = (T_bilin(xpre, xpre)/r2.unsqueeze(1) + bias).view(B, T2, D)
                x = x + mo.to(x.dtype)
            else:
                x = x + blk.mlp(hin)
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

d_full = audit('full_analytic') - SUBBASE
d_arch = audit('archetype') - SUBBASE
print(f"substitution: full-analytic (exactness check) dCE +{d_full:.5f} | archetype-composed (named 144-dim attn) "
      f"+{d_arch:.5f} -> {1-d_arch/FLOOR:.1%} of floor", flush=True)
res = {'gate1_rms_gauge_relerr': g1, 'gate2_block_split_relerr': g2,
       'archetype_capture_of_attn0': round(cap, 4), 'block_variance_shares': shares,
       'full_analytic_dCE': round(d_full, 5), 'archetype_composed_dCE': round(d_arch, 5),
       'archetype_composed_frac': round(1 - d_arch/FLOOR, 4)}
json.dump(res, open(f'{QK}/qk_mlp0_composed_fold.json', 'w'), indent=2)
print("QK MLP0 COMPOSED FOLD DONE", flush=True)
