"""EDITING CAPSTONE -- TARGETED REDIRECT of the verified induction MATCH channel.
The capability dial (qk_induction_dial) controls induction STRENGTH; this controls the induction
TARGET. Because the induction match is a named, held-out-verified channel (MATCH_prev, 98-111%),
we can surgically REPOINT it: at each naturally-induction-active query, cancel the natural match
(-a_readoff * MM_natural) and install a redirect (+a_readoff * MM_redirect) that points every such
query at a single chosen source position p_C. The model should then copy the token C=idx[p_C]
instead of the true continuation. This is a controlled, interpretability-grounded redirect (a base
LM, NOT a safety-trained target -- honest framing: precision-of-edit demo, not a real jailbreak).
Measured: (1) REACH -- redirect success (argmax==C) and P(C)/P(true-next) at induction queries,
edited vs natural; (2) COLLATERAL -- natural FineWeb CE with the SAME edit applied (induction is
rare in natural text, so a clean localized edit should barely move it). Edit touches ONLY the
census-identified induction heads' match channel; all other head function is left intact.
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
MINCOMP = json.load(open(f'{QK}/qk_understanding_props.json'))['minimality']['locally_minimal_components']
SUBST = sorted({(li, h) for (t, li, h) in [ast.literal_eval(c) for c in MINCOMP if c.startswith("('h'")] if 2 <= li <= 10})
SIDX = {lh: i for i, lh in enumerate(SUBST)}
P_C = 1   # primary redirect target position; C = idx[:, P_C] per sequence
P_C2 = 9  # AIMABILITY control: a second, different target -- model must copy token at P_C2, not a default

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

P = 64; NSEQ = 48
prefN = FINEWEB[:NSEQ, 1:1+P]; EVN = torch.cat([prefN, prefN], 1).to(DEV)
FIR = torch.arange(1, P-1, device=DEV); SEC = torch.arange(P, 2*P-1, device=DEV)  # second-copy induction queries

# --- read off a_readoff per head on natural repeated eval (identical procedure to the dial demo) ---
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
print("a_readoff ready:", [round(float(a), 3) for a in AINIT], flush=True)

@torch.no_grad()
def forward(idx, edit):
    """edit: None=model | 'redirect'=repoint induction match to column P_C for active queries."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    tgt = edit[1] if isinstance(edit, tuple) else P_C
    ekind = edit[0] if isinstance(edit, tuple) else edit
    if ekind in ('redirect', 'hard'):
        MMn = match_matrix(idx)
        active = MMn.sum(-1, keepdim=True) > 0            # queries with a natural induction match
        MMr = torch.zeros_like(MMn); MMr[:, :, tgt] = active.squeeze(-1).float()
        MMr = MMr * mask.float()                          # keep causal
        DELTA = MMr - MMn                                 # cancel natural, install redirect
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if ekind in ('redirect', 'hard'):
            pats = []
            for h in range(NH):
                if (li, h) in SIDX:
                    if ekind == 'redirect':
                        pats.append(pat[:, h] + AINIT[SIDX[(li, h)]] * DELTA)
                    else:  # hard: move each active query's TOTAL attention mass onto column P_C
                        ph = pat[:, h].clone(); rs = ph.sum(-1, keepdim=True)
                        act = active.squeeze(-1)                      # (B,T) bool
                        ph[act] = 0.0; ph[:, :, tgt] = torch.where(act, rs.squeeze(-1), ph[:, :, tgt])
                        pats.append(ph)
                else: pats.append(pat[:, h])
            pat = torch.stack(pats, 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def reach():
    idx = EVN[:, :-1]; B, T = idx.shape
    C = idx[:, P_C]                                       # chosen token per sequence
    true_next = EVN[:, SEC+1]                             # (B, |SEC|) natural continuation (label lives in full EVN)
    C2 = idx[:, P_C2]
    out = {}
    for nm, edit in [('model', None), ('redirect', 'redirect'), ('hard', ('hard', P_C)), ('hard_tgt2', ('hard', P_C2))]:
        lg = forward(idx, edit).float(); pr = lg.softmax(-1)
        sub = pr[:, SEC]                                  # (B, |SEC|, V) at induction queries
        pC = sub.gather(-1, C[:, None, None].expand(-1, SEC.numel(), 1)).squeeze(-1)
        pT = sub.gather(-1, true_next.unsqueeze(-1)).squeeze(-1)
        pC2 = sub.gather(-1, C2[:, None, None].expand(-1, SEC.numel(), 1)).squeeze(-1)
        argm = sub.argmax(-1)
        out[nm] = {'P_C': round(float(pC.mean()), 4), 'P_C2': round(float(pC2.mean()), 4), 'P_true_next': round(float(pT.mean()), 4),
                   'argmax_is_C': round(float((argm == C[:, None]).float().mean()), 4),
                   'argmax_is_C2': round(float((argm == C2[:, None]).float().mean()), 4),
                   'argmax_is_true': round(float((argm == true_next).float().mean()), 4)}
    return out

@torch.no_grad()
def collateral():
    idxN = FINEWEB[64:128, :128].to(DEV)                 # disjoint natural text
    o = {}
    for nm, edit in [('model', None), ('redirect', 'redirect'), ('hard', 'hard')]:
        lg = forward(idxN[:, :-1], edit).float()
        o[nm] = round(F.cross_entropy(lg.reshape(-1, V), idxN[:, 1:].reshape(-1)).item(), 4)
    o['collateral_dCE_redirect'] = round(o['redirect'] - o['model'], 4)
    o['collateral_dCE_hard'] = round(o['hard'] - o['model'], 4)
    return o

res = {'reach': reach(), 'collateral': collateral(), 'n_heads_edited': len(SUBST), 'P_C': P_C}
print("REACH:", json.dumps(res['reach'], indent=2), flush=True)
print("COLLATERAL:", json.dumps(res['collateral'], indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_targeted_redirect.json', 'w'), indent=2)
print("QK TARGETED REDIRECT DONE", flush=True)
