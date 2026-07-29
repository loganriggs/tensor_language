"""Model -> code -> BACK (Logan): the explicit induction program's 72 scalars (a,b,c x 24 heads),
initialized from the least-squares read-off, FINETUNED by gradient on the task with the model
otherwise frozen. If a scalar-level finetune closes any gap on real data, that part of the model is
replaced by interpretable code. Train on cooc (natural windows CE + shuffled-repeated second-copy
CE); held-out eval on FineWeb: natural-corpus dCE and induction retention (natural + shuffled).
"""
import json, sys, ast
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
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
MINCOMP = json.load(open(f'{QK}/qk_understanding_props.json'))['minimality']['locally_minimal_components']
SUBST = sorted({(li, h) for (t, li, h) in [ast.literal_eval(c) for c in MINCOMP if c.startswith("('h'")] if 2 <= li <= 10})
SIDX = {lh: i for i, lh in enumerate(SUBST)}
print(f"{len(SUBST)} heads; {3*len(SUBST)} scalars", flush=True)

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1)
    eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

# position templates + init params from lstsq on natural repeated eval (as in qk_induction_predicate)
P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P]; EVN = torch.cat([prefN, prefN], 1).to(DEV)
g = torch.Generator().manual_seed(11)
prefS = prefN.clone()
for r in range(NSEQ): prefS[r] = prefS[r][torch.randperm(P, generator=g)]
EVS = torch.cat([prefS, prefS], 1).to(DEV)
FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)

TEMPL = {}; INIT = {}
@torch.no_grad()
def fit_init():
    idx = EVN[:, :-1]; B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); MM = match_matrix(idx)
    for li in range(11):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        for h in range(NH):
            if (li, h) in SIDX:
                Pt = pat[:, h]; Tm = Pt.mean(0); TEMPL[(li, h)] = Tm
                mb = mask.expand(B, T, T)
                Xf = torch.stack([MM[mb], Tm.unsqueeze(0).expand(B, T, T)[mb], torch.ones_like(MM[mb])], 1)
                INIT[(li, h)] = torch.linalg.lstsq(Xf, Pt[mb].unsqueeze(1)).solution.squeeze(1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
fit_init()
theta = torch.nn.Parameter(torch.stack([INIT[lh] for lh in SUBST]))   # (24,3)

def forward(idx, use_theta):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MM = match_matrix(idx) if use_theta else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if use_theta and any((li, h) in SIDX for h in range(NH)):
            pats = []
            for h in range(NH):
                if (li, h) in SIDX:
                    aa, bb, cc = theta[SIDX[(li, h)]]
                    Tm = TEMPL[(li, h)].unsqueeze(0)
                    pats.append((aa*MM + bb*Tm + cc).masked_fill(~mask, 0.0))
                else:
                    pats.append(pat[:, h])
            pat = torch.stack(pats, 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def metrics(use_theta):
    idxN = FINEWEB[:64, :128].to(DEV)
    lg = forward(idxN[:, :-1], use_theta).float()
    ce_nat = F.cross_entropy(lg.reshape(-1, V), idxN[:, 1:].reshape(-1)).item()
    out = {'natural_CE': round(ce_nat, 4)}
    for nm, EV in [('nat', EVN), ('shuf', EVS)]:
        lg = forward(EV[:, :-1], use_theta).float()
        ce = F.cross_entropy(lg.reshape(-1, V), EV[:, 1:].reshape(-1), reduction='none').view(NSEQ, -1)
        out[f'adv_{nm}'] = round(ce[:, FIR].mean().item() - ce[:, SEC].mean().item(), 3)
    return out

m0 = metrics(False); print("model (real patterns):", m0, flush=True)
mpre = metrics(True); print("code, lstsq init      :", mpre, flush=True)

opt = torch.optim.Adam([theta], lr=2e-3)
for step in range(240):
    if step % 2 == 0:
        i = np.random.randint(0, 5000); b = COOC[i:i+2].to(DEV)[:, :128]
        lg = forward(b[:, :-1], True).float()
        loss = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
    else:
        i = np.random.randint(0, 5000); pref = COOC[i:i+2, 1:1+P].clone()
        for r in range(2): pref[r] = pref[r][torch.randperm(P)]
        b = torch.cat([pref, pref], 1).to(DEV)
        lg = forward(b[:, :-1], True).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none').view(2, -1)
        loss = ce[:, SEC].mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 60 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)

mpost = metrics(True); print("code, task-finetuned  :", mpost, flush=True)
res = {'model': m0, 'code_init': mpre, 'code_finetuned': mpost,
       'theta_init_mean_a': round(float(torch.stack([INIT[lh] for lh in SUBST])[:, 0].mean()), 5),
       'theta_post_mean_a': round(float(theta[:, 0].mean()), 5)}
json.dump(res, open(f'{QK}/qk_induction_finetune.json', 'w'), indent=2)
torch.save({'theta': theta.detach(), 'heads': SUBST}, f'{QK}/qk_induction_finetune.pt')
print("QK INDUCTION FINETUNE DONE", flush=True)
