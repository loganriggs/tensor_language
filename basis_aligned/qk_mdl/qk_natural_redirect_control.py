"""SETTLING CONTROL for §37c (red-team F2/F3/F4): is the weak natural-trigger reach a recoverable
CALIBRATION limit (fixed by more amplitude) or intrinsic? Primary control = a SCALE SWEEP on natural
triggers: if P_payload climbs toward the planted ~0.8 as SCALE rises (before collateral explodes), the
natural weakness is calibration (an in-the-wild edit that just needs more amplitude); if it plateaus
low, the induction really isn't there to hijack. Also (F2) reports the natural induction match
amplitude at gated queries vs the planted read-off AINIT -- a less-confounded 'how much induction is
present' measure than baseline true-next. Larger slice + triggers chosen for power (F4). Memory-safe:
chunked forward, lm_head only on gated positions.
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
P_C = 1

def match_matrix(idx):
    B, T = idx.shape
    eq = idx.unsqueeze(2) == torch.roll(idx, 1, dims=1).unsqueeze(1); eq[:, :, 0] = False
    return (eq & torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))).float()

# read-off AINIT on planted repeated eval (same as §37)
P = 64; NSEQ = 48
prefBase = FINEWEB[:NSEQ, 1:1+P]; EVbase = torch.cat([prefBase, prefBase], 1).to(DEV)
AINIT = torch.zeros(len(SUBST), device=DEV)
@torch.no_grad()
def _readoff(EV, gatemask=None):
    """lstsq the pattern onto [MATCH, template, const] per SUBST head; if gatemask given, restrict rows
    to gated queries (natural match-amplitude measurement)."""
    idx = EV[:, :-1]; B, T = idx.shape
    coeff = torch.zeros(len(SUBST), device=DEV)
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
                Pt = pat[:, h]; Tm = Pt.mean(0)
                mb = (gatemask.unsqueeze(-1) & mask if gatemask is not None else mask.expand(B, T, T))
                if mb.sum() < 10: continue
                Xf = torch.stack([MM[mb], Tm.unsqueeze(0).expand(B, T, T)[mb], torch.ones_like(MM[mb])], 1)
                coeff[SIDX[(li, h)]] = torch.linalg.lstsq(Xf, Pt[mb].unsqueeze(1)).solution[0, 0]
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return coeff
AINIT = _readoff(EVbase)

@torch.no_grad()
def forward_resid(idx, gate, scale):
    """returns final normed residual (B,T,D); gate=(B,T) bool, redirect payload=token@P_C at scale."""
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
            pat = torch.stack([pat[:, h] + (scale*AINIT[SIDX[(li, h)]]*DELTA if (li, h) in SIDX else 0.0) for h in range(NH)], 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return F.rms_norm(x, (D,))

# larger slice for power; pick triggers with enough induction-active occurrences
SL = FINEWEB[64:320, :128].to(DEV)
idxf = SL[:, :-1]; active_f = match_matrix(idxf).sum(-1) > 0
cnt = torch.bincount(idxf[active_f].cpu(), minlength=V)
rate = torch.bincount(idxf.reshape(-1).cpu(), minlength=V).float() / idxf.numel()
elig = torch.where(cnt >= 40)[0]                         # >=40 induction-active occurrences => real power
elig = elig[rate[elig].argsort()]
TRIGS = {'lowest_freq_powered': int(elig[0]), 'mid_freq': int(elig[len(elig)//2])}
SCALES = [10, 20, 40, 80, 160]

@torch.no_grad()
def gated_logits(idx, gate, scale):
    """lm_head only on gated positions -> (n_gated, V), memory-safe via chunking over sequences."""
    outs = []
    for s in range(0, idx.shape[0], 64):
        ci = idx[s:s+64]; cg = gate[s:s+64]
        if cg.sum() == 0: continue
        r = forward_resid(ci, cg, scale)              # (b,T,D)
        lg = 30*torch.tanh(m.lm_head(r[cg])/30)       # (n_gated_chunk, V)
        outs.append(lg.float())
    return torch.cat(outs, 0) if outs else None

def evaluate(tk):
    gate = active_f & (idxf == tk)
    n = int(gate.sum())
    payload = idxf[:, P_C]
    pay_full = payload[:, None].expand_as(idxf)[gate]              # (n,) payload token per gated position
    natcoeff = _readoff(SL, gatemask=gate)
    out = {'token': tk, 'base_rate': round(float(rate[tk]), 6), 'n': n,
           'match_coeff_ratio_nat_over_planted': round(float((natcoeff.abs().mean()/AINIT.abs().mean())), 3),
           'sweep': {}}
    # model baseline
    lg0 = gated_logits(idxf, gate, 0.0); pr0 = lg0.softmax(-1)
    base_pay = float(pr0.gather(-1, pay_full[:, None]).mean())
    for sc in SCALES:
        lg = gated_logits(idxf, gate, sc); pr = lg.softmax(-1)
        pay = pr.gather(-1, pay_full[:, None]).squeeze(-1)
        cap = (pr.argmax(-1) == pay_full).float()
        out['sweep'][f'scale={sc}'] = {'P_payload': round(float(pay.mean()), 4),
                                       'P_payload_SE': round(float(pay.std()/np.sqrt(n)), 4),
                                       'argmax_capture': round(float(cap.mean()), 4)}
    out['model_P_payload'] = round(base_pay, 4)
    return out

res = {'triggers': {k: evaluate(v) for k, v in TRIGS.items()}, 'planted_ref': {'P_payload': 0.833, 'capture': 0.958}}
print("NATURAL SCALE-SWEEP CONTROL:", json.dumps(res, indent=2), flush=True)
json.dump(res, open(f'{QK}/qk_natural_redirect_control.json', 'w'), indent=2)
print("QK NATURAL REDIRECT CONTROL DONE", flush=True)
