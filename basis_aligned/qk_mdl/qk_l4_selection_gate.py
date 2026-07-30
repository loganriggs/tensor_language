"""LAYER-2 GATED SELECTION VOCABULARY (census-predicate simultaneous-substitution gate at layer 2).

Structural role mirrors qk_l1_selection_gate.py — a single-layer selection gate reporting held-out
predicate gain per head and a simultaneous-substitution gated cross-entropy with paired per-token SE
on FW[448:600]. But per the task it uses the CENSUS PREDICATE LIBRARY (MATCH_same, MATCH_prev,
KEY_punct/func/cap/newline/digit/subword, PREV1/2, MATCH_prev2, FIRST + positional TEMPLATE + causal),
not hand-coded per-head archetypes, because layer 2's programmatic heads fit standard predicates
(L2H5 = induction necessity core MATCH_same; L2H4 = KEY_punct). Tests whether the induction MATCH
predicate (meaning-verified held-out 98-111% at layer 1) survives the gate at layer 2, and whether
any OTHER layer-2 head is gated-nameable.

DERIVATION: copied VERBATIM from qk_selection_census_v2.py (its feats() predicate library, patterns(),
per-head normal equations, heldout_r2(), and forward_coded simultaneous-substitution gate are reused
unchanged). CHANGES, all layer-index / input only:
  - LAYER = 2 constant added; the GATE's PROGSET is restricted to programmatic heads AT LAYER 2 only
    (predicate gain >= 0.05), so forward_coded substitutes coded patterns solely at layer 2 (isolating
    the layer, exactly as qk_l1_selection_gate.py restricts to li == 1);
  - cooc batches reduced 8->4 for memory safety (same sequences, identical accumulated statistics);
  - summary + JSON report focus on layer 2 (full 162-head census still computed and saved);
  - output file qk_l2_selection_gate.json.
The heldout_r2 / gain / gate machinery, predicate definitions, and paired-SE computation are byte-for-
byte the census-v2 originals — nothing is paraphrased.
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
LAYER = 4
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
KP=torch.zeros(V,dtype=torch.bool);KF=torch.zeros(V,dtype=torch.bool);KC=torch.zeros(V,dtype=torch.bool)
KD=torch.zeros(V,dtype=torch.bool);KN=torch.zeros(V,dtype=torch.bool);KS=torch.zeros(V,dtype=torch.bool)
for i in range(50257):
    ss = tok.convert_ids_to_tokens(i)
    if ss is None: continue
    core = ss.replace('Ġ', ''); lead = ss.startswith('Ġ')
    if len(core) and all(c in _P for c in core): KP[i] = True
    if core.lower() in FUNC: KF[i] = True
    if lead and len(core) and core[0].isupper(): KC[i] = True
    if len(core) and all(c.isdigit() for c in core): KD[i]=True
    if 'Ċ' in ss: KN[i]=True
    if not lead and len(core) and core[0].isalpha() and core[0].islower(): KS[i]=True
KP,KF,KC,KD,KN,KS = KP.to(DEV),KF.to(DEV),KC.to(DEV),KD.to(DEV),KN.to(DEV),KS.to(DEV)
T0 = 128
FEATN = ['MATCH_prev', 'MATCH_same', 'KEY_punct', 'KEY_func', 'KEY_cap', 'FIRST', 'PREV1', 'PREV2', 'MATCH_prev2', 'KEY_digit', 'KEY_newline', 'KEY_subword']
NF = len(FEATN)

def feats(idx):
    """(B,NF,T,T) predicate features + causal mask."""
    B, T = idx.shape
    causal = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    prevtok = torch.roll(idx, 1, dims=1); prevtok[:, 0] = -1
    Fs = torch.zeros(B, NF, T, T, device=DEV)
    Fs[:, 0] = (prevtok.unsqueeze(1) == idx.unsqueeze(2)).float()        # tok_{j-1}==tok_i
    Fs[:, 1] = (idx.unsqueeze(1) == idx.unsqueeze(2)).float()            # tok_j==tok_i
    Fs[:, 2] = KP[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 3] = KF[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 4] = KC[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 5, :, 0] = 1.0
    Fs[:, 6] = torch.diag(torch.ones(T-1, device=DEV), -1)
    Fs[:, 7] = torch.diag(torch.ones(T-2, device=DEV), -2)
    prev2 = torch.roll(idx, 2, dims=1); prev2[:, :2] = -1
    Fs[:, 8] = (prev2.unsqueeze(1) == idx.unsqueeze(2)).float()
    Fs[:, 9] = KD[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 10] = KN[idx].float().unsqueeze(1).expand(B, T, T)
    Fs[:, 11] = KS[idx].float().unsqueeze(1).expand(B, T, T)
    return Fs * causal, causal


@torch.no_grad()
def patterns(idx):
    """per-layer (B,NH,T,T) patterns from the real forward."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = []
    for li in range(NL):
        b = m.transformer.h[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        out.append(pat)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + b.mlp(F.rms_norm(x, (D,)))
    return out

# accumulate per-head normal equations: features = [NF predicates, TEMPLATE, 1]
K = NF + 2
AtA = torch.zeros(NL, NH, K, K, device=DEV, dtype=torch.float64)
Aty = torch.zeros(NL, NH, K, device=DEV, dtype=torch.float64)
# first pass: templates (mean pattern per head)
TPL = torch.zeros(NL, NH, T0, T0, device=DEV); nb = 0
for i in range(0, 48, 4):
    pats = patterns(COOC[i:i+4].to(DEV)[:, :T0])
    for li in range(NL): TPL[li] += pats[li].mean(0)
    nb += 1
TPL /= nb
print("templates ready", flush=True)
# second pass: normal equations
for i in range(48, 168, 4):
    idx = COOC[i:i+4].to(DEV)[:, :T0]
    Fs, causal = feats(idx); B = idx.shape[0]
    pats = patterns(idx)
    mask_flat = causal.expand(B, T0, T0).reshape(-1)
    Xbase = torch.cat([Fs.reshape(B, NF, -1), torch.zeros(B, 2, T0*T0, device=DEV)], 1)
    for li in range(NL):
        tplf = (TPL[li].unsqueeze(0).expand(B, NH, T0, T0) * causal).reshape(B, NH, -1)
        for h in range(NH):
            X = Xbase.clone(); X[:, NF] = tplf[:, h]; X[:, NF+1] = causal.expand(B, T0, T0).reshape(B, -1).float()
            Xf = X.permute(0, 2, 1).reshape(-1, K)[mask_flat].double()
            yf = pats[li][:, h].reshape(-1)[mask_flat].double()
            AtA[li, h] += Xf.T @ Xf; Aty[li, h] += Xf.T @ yf
print("normal equations done", flush=True)
W = torch.linalg.solve(AtA + 1e-6*torch.eye(K, device=DEV, dtype=torch.float64), Aty.unsqueeze(-1)).squeeze(-1).float()

# held-out R^2: full model vs template-only
@torch.no_grad()
def heldout_r2():
    ss_res = torch.zeros(NL, NH, device=DEV); ss_tpl = torch.zeros(NL, NH, device=DEV); ss_tot = torch.zeros(NL, NH, device=DEV)
    for i in range(200, 240, 4):
        idx = COOC[i:i+4].to(DEV)[:, :T0]
        Fs, causal = feats(idx); B = idx.shape[0]
        pats = patterns(idx)
        for li in range(NL):
            for h in range(NH):
                tpl = (TPL[li, h].unsqueeze(0) * causal)
                y = pats[li][:, h]
                pred = (Fs * W[li, h, :NF].view(1, NF, 1, 1)).sum(1) + W[li, h, NF]*tpl + W[li, h, NF+1]*causal
                # template-only refit: use tpl scale from W but drop predicates
                pred_t = W[li, h, NF]*tpl + W[li, h, NF+1]*causal
                mu = y[:, causal].mean()
                ss_res[li, h] += ((pred - y)[:, causal]**2).sum(); ss_tpl[li, h] += ((pred_t - y)[:, causal]**2).sum()
                ss_tot[li, h] += ((y[:, causal] - mu)**2).sum()
    return 1 - ss_res/ss_tot, 1 - ss_tpl/ss_tot
R2full, R2tpl = heldout_r2()
gain = (R2full - R2tpl)
census = []
for li in range(NL):
    for h in range(NH):
        coef = W[li, h, :NF]
        top = int(coef.abs().argmax())
        census.append({'layer': li, 'head': h, 'r2_full': round(float(R2full[li, h]), 3),
                       'r2_template': round(float(R2tpl[li, h]), 3), 'predicate_gain': round(float(gain[li, h]), 3),
                       'top_predicate': FEATN[top], 'top_coef': round(float(coef[top]), 4)})
prog = [c for c in census if c['predicate_gain'] >= 0.05]
from collections import Counter
print(f"PROGRAMMATIC heads (predicate gain >=5% held-out): {len(prog)}/162", flush=True)
print("by top predicate:", Counter(c['top_predicate'] for c in prog).most_common(), flush=True)
print("by layer:", Counter(c['layer'] for c in prog), flush=True)
# layer-LAYER focus: report every head's gain + top predicate at this layer
print(f"--- LAYER {LAYER} per-head census ---", flush=True)
for c in [c for c in census if c['layer'] == LAYER]:
    flagp = ' PROG' if c['predicate_gain'] >= 0.05 else ''
    print(f"  L{LAYER}H{c['head']}: gain {c['predicate_gain']} top={c['top_predicate']} ({c['top_coef']}) "
          f"full {c['r2_full']} / tpl {c['r2_template']}{flagp}", flush=True)

# GATE: substitute coded patterns at LAYER's programmatic heads simultaneously; dCE on held slice
PROGSET = {(c['layer'], c['head']) for c in prog if c['layer'] == LAYER}
@torch.no_grad()
def forward_coded(idx, subst):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    Fs, causal = feats(idx) if subst else (None, mask)
    for li in range(NL):
        b = m.transformer.h[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        if subst:
            for h in range(NH):
                if (li, h) in PROGSET:
                    tplc = (TPL[li, h][:T, :T].unsqueeze(0) * mask)
                    coded = (Fs * W[li, h, :NF].view(1, NF, 1, 1)).sum(1) + W[li, h, NF]*tplc + W[li, h, NF+1]*mask
                    pat = pat.clone(); pat[:, h] = coded.masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + b.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def per_tok(subst):
    ces = []
    for i in range(0, len(HELD), 4):
        bb = HELD[i:i+4].to(DEV)[:, :T0+1]
        lg = forward_coded(bb[:, :-1], subst)
        ce = F.cross_entropy(lg.reshape(-1, V), bb[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.cpu())
    return torch.cat(ces)

ce_real = per_tok(False); ce_coded = per_tok(True)
d = ce_coded - ce_real
gate = {'layer': LAYER, 'n_heads_coded': len(PROGSET),
        'heads': sorted(h for (l, h) in PROGSET),
        'dCE': round(float(d.mean()), 5), 'SE': round(float(d.std()/np.sqrt(d.numel())), 6)}
lay_prog = [c for c in prog if c['layer'] == LAYER]
gains_here = [c['predicate_gain'] for c in census if c['layer'] == LAYER]
print(f"GATE: {len(PROGSET)} layer-{LAYER} programmatic heads coded simultaneously "
      f"(heads {gate['heads']}): dCE +{gate['dCE']} (SE {gate['SE']})", flush=True)
print(f"SUMMARY L{LAYER} SELECTION: {len(lay_prog)} programmatic heads at layer {LAYER}; "
      f"per-head predicate gain range {min(gains_here):.3f}..{max(gains_here):.3f}; "
      f"gate +{gate['dCE']} nats/tok (SE {gate['SE']})", flush=True)
json.dump({'layer': LAYER, 'census': census, 'programmatic': prog,
           'layer_programmatic': lay_prog, 'gate': gate},
          open(f'{QK}/qk_l{LAYER}_selection_gate.json', 'w'), indent=2)
print(f"QK L{LAYER} SELECTION GATE DONE", flush=True)
