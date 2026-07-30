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
    ekind = edit[0] if isinstance(edit, tuple) else edit
    tgt = edit[1] if (isinstance(edit, tuple) and ekind == 'hard') else P_C
    SOFT_SCALE = edit[1] if (isinstance(edit, tuple) and ekind == 'redirect') else 1.0
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
                        pats.append(pat[:, h] + SOFT_SCALE * AINIT[SIDX[(li, h)]] * DELTA)
                    else:  # hard: move each active query's TOTAL attention mass onto column tgt
                        ph = pat[:, h].clone(); rs = ph.sum(-1, keepdim=True)
                        act = active.squeeze(-1)                      # (B,T) bool
                        ph[act] = 0.0; ph[:, :, tgt] = torch.where(act, rs.squeeze(-1), ph[:, :, tgt])
                        ph = ph * mask.float()                        # F5: re-enforce causality (tgt may exceed q)
                        pats.append(ph)
                else: pats.append(pat[:, h])
            pat = torch.stack(pats, 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

def _tokstats(idx, edit, tgt):
    """P at token@tgt / true_next, argmax breakdown, and residual-argmax destination (F3)."""
    lg = forward(idx, edit).float(); pr = lg.softmax(-1)
    sub = pr[:, SEC]; Ct = idx[:, tgt]
    true_next = EVN[:, SEC+1]
    pTgt = sub.gather(-1, Ct[:, None, None].expand(-1, SEC.numel(), 1)).squeeze(-1)
    pT = sub.gather(-1, true_next.unsqueeze(-1)).squeeze(-1)
    argm = sub.argmax(-1)
    is_tgt = (argm == Ct[:, None]); is_true = (argm == true_next)
    # F3: where does the residual argmax mass (neither target nor true-next) go?
    tok_tm1 = idx[:, max(tgt-1, 0)]; tok_tp1 = EVN[:, tgt+1]   # neighbours of the aimed column
    resid = ~(is_tgt | is_true)
    d = {'P_tgt': round(float(pTgt.mean()), 4), 'P_true_next': round(float(pT.mean()), 4),
         'argmax_is_tgt': round(float(is_tgt.float().mean()), 4),
         'argmax_is_true': round(float(is_true.float().mean()), 4),
         'residual_frac': round(float(resid.float().mean()), 4),
         'resid_is_tgt_nbr': round(float(((argm == tok_tm1[:, None]) | (argm == tok_tp1[:, None]))[resid].float().mean() if resid.any() else 0.0), 4)}
    return d

@torch.no_grad()
def reach():
    idx = EVN[:, :-1]
    C, C2 = idx[:, P_C], idx[:, P_C2]
    true_next = EVN[:, SEC+1]
    out = {}
    # model baseline + double dissociation (aim@P_C vs aim@P_C2), measuring BOTH tokens each time
    for nm, edit in [('model', None), ('hard', ('hard', P_C)), ('hard_tgt2', ('hard', P_C2))]:
        lg = forward(idx, edit).float(); pr = lg.softmax(-1); sub = pr[:, SEC]
        g = lambda t: round(float(sub.gather(-1, t[:, None, None].expand(-1, SEC.numel(), 1)).squeeze(-1).mean()), 4)
        pT = round(float(sub.gather(-1, true_next.unsqueeze(-1)).squeeze(-1).mean()), 4)
        argm = sub.argmax(-1)
        out[nm] = {'P_C': g(C), 'P_C2': g(C2), 'P_true_next': pT,
                   'argmax_is_C': round(float((argm == C[:, None]).float().mean()), 4),
                   'argmax_is_C2': round(float((argm == C2[:, None]).float().mean()), 4),
                   'argmax_is_true': round(float((argm == true_next).float().mean()), 4)}
    return out

@torch.no_grad()
def soft_sweep():
    """F6: does the soft linear repoint EVER engage as amplitude grows? If yes -> 'over-determined' is wrong."""
    idx = EVN[:, :-1]
    return {f'scale={s}': _tokstats(idx, ('redirect', s), P_C) for s in (1, 3, 10, 30, 100)}

@torch.no_grad()
def span():
    """F8: hard repoint across the sequence, not just near the start."""
    idx = EVN[:, :-1]
    return {f'tgt={t}': _tokstats(idx, ('hard', t), t) for t in (1, 9, 30, 55)}

@torch.no_grad()
def rs_stats():
    """F4: sign/magnitude of the relocated row-sum rs over active induction queries (SUBST heads)."""
    idx = EVN[:, :-1]; B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); MMn = match_matrix(idx)
    act = (MMn.sum(-1) > 0); vals = []
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
            if (li, h) in SIDX: vals.append(pat[:, h].sum(-1)[act])
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    rs = torch.cat(vals)
    # also the natural matched-column weight for comparison (what induction actually uses)
    return {'frac_positive': round(float((rs > 0).float().mean()), 3), 'mean': round(float(rs.mean()), 4),
            'mean_abs': round(float(rs.abs().mean()), 4), 'n': int(rs.numel())}

@torch.no_grad()
def collateral():
    idxN = FINEWEB[64:128, :128].to(DEV)                 # disjoint natural text
    lg0 = forward(idxN[:, :-1], None).float()
    base = F.cross_entropy(lg0.reshape(-1, V), idxN[:, 1:].reshape(-1)).item()
    o = {'model_CE': round(base, 4)}
    for nm, edit in [('hard', ('hard', P_C)), ('soft_s10', ('redirect', 10)), ('soft_s30', ('redirect', 30))]:
        lg = forward(idxN[:, :-1], edit).float()
        ce = F.cross_entropy(lg.reshape(-1, V), idxN[:, 1:].reshape(-1)).item()
        o[f'{nm}_CE'] = round(ce, 4); o[f'collateral_dCE_{nm}'] = round(ce - base, 4)
    return o

res = {'reach': reach(), 'soft_amplitude_sweep': soft_sweep(), 'span_targets': span(),
       'rs_stats': rs_stats(), 'collateral': collateral(), 'n_heads_edited': len(SUBST), 'P_C': P_C, 'P_C2': P_C2}
print("REACH (double dissociation):", json.dumps(res['reach'], indent=2), flush=True)
print("SOFT AMPLITUDE SWEEP (F6):", json.dumps(res['soft_amplitude_sweep'], indent=2), flush=True)
print("SPAN TARGETS (F8):", json.dumps(res['span_targets'], indent=2), flush=True)
print("RS STATS (F4):", json.dumps(res['rs_stats'], indent=2), flush=True)
print("COLLATERAL (F7):", json.dumps(res['collateral'], indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_targeted_redirect.json', 'w'), indent=2)
print("QK TARGETED REDIRECT DONE", flush=True)
