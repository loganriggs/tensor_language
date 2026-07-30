"""MECHANISM TEST for §37c: does conditional-redirect reach track LOCAL INDUCTION STRENGTH?
§37c found natural-trigger reach is weak and argued the cause is low baseline induction at natural
trigger queries ("only as much induction as is locally present can be hijacked"). This tests that
directly: on real text, bin induction-active queries by their BASELINE true-next probability (a proxy
for how much induction is present), then fire the SAME redirect (payload = token@1, SCALE=10 -- held
fixed, so payload choice and amplitude are controlled) gated to each bin, and measure reach + SE.
PREDICTION if §37c's mechanism is right: reach rises monotonically with baseline induction strength and
approaches the planted level in the strong bin -- i.e. the natural weakness is weak local induction, NOT
planting or payload choice or amplitude non-transfer.
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
P_C = 1; SCALE = 10.0

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

P = 64; NSEQ = 48
prefBase = FINEWEB[:NSEQ, 1:1+P]; EVbase = torch.cat([prefBase, prefBase], 1).to(DEV)
AINIT = torch.zeros(len(SUBST), device=DEV)
@torch.no_grad()
def read_a(EV):
    idx = EV[:, :-1]; B, T = idx.shape
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
read_a(EVbase)

@torch.no_grad()
def forward(idx, gate):
    """gate: None=model | (B,T) bool mask of queries to redirect (payload = token@P_C, scale x10)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    if gate is not None:
        MMn = match_matrix(idx)
        MMr = torch.zeros_like(MMn); MMr[:, :, P_C] = gate.float(); MMr = MMr * mask.float()
        DELTA = MMr - MMn * gate.unsqueeze(-1).float()
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if gate is not None:
            pat = torch.stack([pat[:, h] + (SCALE*AINIT[SIDX[(li, h)]]*DELTA if (li, h) in SIDX else 0.0) for h in range(NH)], 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

SL = FINEWEB[64:256, :128].to(DEV)
idx = SL[:, :-1]; B, T = idx.shape
MMn = match_matrix(idx); active = MMn.sum(-1) > 0
true_next = SL[:, 1:]; payload = idx[:, P_C]
prm = forward(idx, None).float().softmax(-1)
base_ptn = prm.gather(-1, true_next.unsqueeze(-1)).squeeze(-1)          # baseline true-next P per query

BINS = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]
res = {'note': 'reach vs baseline induction strength; payload=token@1 and scale=10 held fixed'}
res['bins'] = {}
for lo, hi in BINS:
    g = active & (base_ptn >= lo) & (base_ptn < hi)
    n = int(g.sum())
    if n < 5:
        res['bins'][f'{lo}-{hi}'] = {'n': n}; continue
    prc = forward(idx, g).float().softmax(-1)
    pay = prc.gather(-1, payload[:, None, None].expand(-1, T, 1)).squeeze(-1)[g]
    cap = (prc.argmax(-1) == payload[:, None])[g].float()
    ptn_c = prc.gather(-1, true_next.unsqueeze(-1)).squeeze(-1)[g]
    res['bins'][f'{lo}-{hi}'] = {
        'n': n, 'baseline_true_next': round(float(base_ptn[g].mean()), 4),
        'reach_P_payload': round(float(pay.mean()), 4), 'reach_P_payload_SE': round(float(pay.std()/np.sqrt(n)), 4),
        'argmax_capture': round(float(cap.mean()), 4), 'argmax_capture_SE': round(float(cap.std()/np.sqrt(n)), 4),
        'true_next_after': round(float(ptn_c.mean()), 4)}
print("REACH vs INDUCTION STRENGTH:", json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_natural_strong_induction.json', 'w'), indent=2)
print("QK NATURAL STRONG INDUCTION DONE", flush=True)
