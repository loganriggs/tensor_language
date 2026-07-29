"""RED-TEAM FIX #4: held-out refit of the induction predicate. The flagship numbers were fit on the
scored prefixes; here templates and (a,b,c) are fit ONLY on cooc-derived prefixes, then evaluated on
FRESH FineWeb rows (400-447, never used in any fit or audit) and on Pile -- at TWO periods (48, 64),
templates refit per period on the FIT corpus only. Conditions per (period, corpus): full-model
advantage vs explicit-patterns-in-full advantage (natural + shuffled). Pass = retention comparable
to the provisional 100-107% numbers.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
import ast as _ast
MIN = json.load(open(f'{QK}/qk_understanding_props.json'))['minimality']['locally_minimal_components']
SUBST = sorted({(li, h) for (t, li, h) in [_ast.literal_eval(c) for c in MIN if c.startswith("('h'")] if 2 <= li <= 10})
SIDX = {lh: i for i, lh in enumerate(SUBST)}

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

def rep_eval(rows, P, shuffled=False, seed=11):
    pref = rows[:, 1:1+P].clone()
    if shuffled:
        g = torch.Generator().manual_seed(seed)
        for r in range(pref.shape[0]): pref[r] = pref[r][torch.randperm(P, generator=g)]
    return torch.cat([pref, pref], 1).to(DEV)


@torch.no_grad()
def forward(idx, subst=None, PAR=None, TEM=None, fit_collect=None):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    MM = match_matrix(idx) if (subst or fit_collect is not None) else None
    for li in range(NL):
        blk = m.transformer.h[li]; a = blk.attn
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if fit_collect is not None:
            for h in range(NH):
                if (li, h) in SIDX:
                    Pt = pat[:, h]; Tm = Pt.mean(0); fit_collect['tem'][(li, h)] = Tm
                    mb = mask.expand(B, T, T)
                    Xf = torch.stack([MM[mb], Tm.unsqueeze(0).expand(B, T, T)[mb], torch.ones_like(MM[mb])], 1)
                    fit_collect['par'][(li, h)] = torch.linalg.lstsq(Xf, Pt[mb].unsqueeze(1)).solution.squeeze(1)
        if subst:
            for h in range(NH):
                if (li, h) in subst:
                    aa, bb, cc = PAR[(li, h)]
                    pat[:, h] = (aa*MM + bb*TEM[(li, h)].unsqueeze(0) + cc).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

def adv_of(lg, EV, P):
    FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()

print("building pile rows...", flush=True)
PILE = build_eval_tokens(n_chunks=48, seq_len=130)
res = {}
for P in (48, 64):
    # FIT strictly on cooc-derived natural repeated prefixes
    FIT = rep_eval(COOC[3000:3048], P)
    fc = {'tem': {}, 'par': {}}
    forward(FIT[:, :-1], fit_collect=fc)
    PAR, TEM = fc['par'], fc['tem']
    for corpus, rows in [('fineweb_fresh', FINEWEB[400:448]), ('pile', PILE)]:
        for sh in (False, True):
            EV = rep_eval(rows, P, shuffled=sh)
            af = adv_of(forward(EV[:, :-1]), EV, P)
            ax = adv_of(forward(EV[:, :-1], subst=set(SUBST), PAR=PAR, TEM=TEM), EV, P)
            key = f"P{P}_{corpus}_{'shuf' if sh else 'nat'}"
            res[key] = {'adv_full': round(af, 3), 'adv_explicit': round(ax, 3), 'retention': round(ax/af, 3)}
            print(f"{key}: full {af:+.3f} explicit {ax:+.3f} retention {ax/af:.1%}", flush=True)
json.dump(res, open(f'{QK}/qk_induction_heldout.json', 'w'), indent=2)
print("QK INDUCTION HELDOUT DONE", flush=True)
