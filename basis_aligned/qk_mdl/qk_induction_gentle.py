"""GENTLE integration of the induction code (Logan: 'is there a way to more gently integrate it?').
Surgical HYBRID: keep each head's OWN pattern (all non-induction function intact) and swap only the
match channel:  pat = pat_model + (a_code - a_readoff) * MATCH.  At init (a_code = a_readoff) this
is EXACTLY the model. Finetune only the 24 match coefficients a_code on a 50/50 natural/shuffled
mix. Prediction: natural CE stays ~model, induction rises to >=100%. Also: full-replacement arm
refinetuned with natural-heavy loss (0.9/0.1) for the Pareto comparison Logan suggested.
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

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P]; EVN = torch.cat([prefN, prefN], 1).to(DEV)
g = torch.Generator().manual_seed(11); prefS = prefN.clone()
for r in range(NSEQ): prefS[r] = prefS[r][torch.randperm(P, generator=g)]
EVS = torch.cat([prefS, prefS], 1).to(DEV)
FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)

# read off a_readoff per head (lstsq on natural repeated eval, MATCH channel with template+const)
AINIT = torch.zeros(len(SUBST), device=DEV)
@torch.no_grad()
def read_a():
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
                Pt = pat[:, h]; Tm = Pt.mean(0); mb = mask.expand(B, T, T)
                Xf = torch.stack([MM[mb], Tm.unsqueeze(0).expand(B, T, T)[mb], torch.ones_like(MM[mb])], 1)
                AINIT[SIDX[(li, h)]] = torch.linalg.lstsq(Xf, Pt[mb].unsqueeze(1)).solution[0, 0]
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
read_a()
acode = torch.nn.Parameter(AINIT.clone())

def forward(idx, hybrid):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MM = match_matrix(idx) if hybrid else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if hybrid:
            pats = []
            for h in range(NH):
                if (li, h) in SIDX:
                    pats.append(pat[:, h] + (acode[SIDX[(li, h)]] - AINIT[SIDX[(li, h)]]) * MM)
                else:
                    pats.append(pat[:, h])
            pat = torch.stack(pats, 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def metrics(hybrid):
    idxN = FINEWEB[:64, :128].to(DEV)
    lg = forward(idxN[:, :-1], hybrid).float()
    ce_nat = F.cross_entropy(lg.reshape(-1, V), idxN[:, 1:].reshape(-1)).item()
    out = {'natural_CE': round(ce_nat, 4)}
    for nm, EV in [('nat', EVN), ('shuf', EVS)]:
        lg = forward(EV[:, :-1], hybrid).float()
        ce = F.cross_entropy(lg.reshape(-1, V), EV[:, 1:].reshape(-1), reduction='none').view(NSEQ, -1)
        out[f'adv_{nm}'] = round(ce[:, FIR].mean().item() - ce[:, SEC].mean().item(), 3)
    return out

m0 = metrics(False); print("model            :", m0, flush=True)
h0 = metrics(True);  print("hybrid @ init    :", h0, "(sanity: must equal model)", flush=True)
opt = torch.optim.Adam([acode], lr=2e-3)
for step in range(200):
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
h1 = metrics(True); print("hybrid finetuned :", h1, flush=True)
res = {'model': m0, 'hybrid_init': h0, 'hybrid_finetuned': h1,
       'mean_a_shift': round(float((acode - AINIT).mean().detach()), 6)}
json.dump(res, open(f'{QK}/qk_induction_gentle.json', 'w'), indent=2)
print("QK INDUCTION GENTLE DONE", flush=True)
