"""T2: SELECTION-CHANNEL CENSUS across all 18 layers x 9 heads. For each head, fit its pattern
against a small library of explicit selection predicates (the program family the induction
predicate validated), held-out R^2 per head, then GATE the nameable heads by substitution.
Predicate features per (query i, key j):
  MATCH_prev  : 1[tok_{j-1} == tok_i]      (induction-style match)
  MATCH_same  : 1[tok_j == tok_i]          (same-token attention)
  KEY_punct / KEY_func / KEY_cap : key-token class indicators
  FIRST       : 1[j == 0]
  PREV1/PREV2 : 1[j == i-1] / 1[j == i-2]  (local recency spikes)
  TEMPLATE    : the head's positional mean pattern (offset structure)
Fit per head by least squares on cooc batches; held-out R^2 on fresh cooc rows. Census table =
per-head best structure + coefficient profile. GATE: substitute coded patterns at the top-K
programmatic heads (excluding template-only heads) simultaneously; dCE with SE on held-back
FW[448:600]. Notes: template captures pure-positional heads (named 'positional'); a head is called
PROGRAMMATIC if predicates beyond template explain >=5% additional pattern variance held-out.
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
m, cfg = load_elriggs('bilin12')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
KP = torch.zeros(V, dtype=torch.bool); KF = torch.zeros(V, dtype=torch.bool); KC = torch.zeros(V, dtype=torch.bool)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): KP[i] = True
    if core.lower() in FUNC: KF[i] = True
    if lead and len(core) and core[0].isupper(): KC[i] = True
KP, KF, KC = KP.to(DEV), KF.to(DEV), KC.to(DEV)
T0 = 128
FEATN = ['MATCH_prev', 'MATCH_same', 'KEY_punct', 'KEY_func', 'KEY_cap', 'FIRST', 'PREV1', 'PREV2']
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
    eye1 = torch.diag(torch.ones(T-1, device=DEV), -1); Fs[:, 6] = eye1
    eye2 = torch.diag(torch.ones(T-2, device=DEV), -2); Fs[:, 7] = eye2
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
        q, k = qk(a.c_q), qk(a.c_k)
        sc = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
        pat = sc.square().masked_fill(~mask, 0.0); pat = pat/pat.sum(-1, keepdim=True).clamp_min(1e-9)
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
for i in range(0, 48, 8):
    pats = patterns(COOC[i:i+8].to(DEV)[:, :T0])
    for li in range(NL): TPL[li] += pats[li].mean(0)
    nb += 1
TPL /= nb
print("templates ready", flush=True)
# second pass: normal equations
for i in range(48, 168, 8):
    idx = COOC[i:i+8].to(DEV)[:, :T0]
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
    for i in range(200, 240, 8):
        idx = COOC[i:i+8].to(DEV)[:, :T0]
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
for c in sorted(prog, key=lambda c: -c['predicate_gain'])[:15]:
    print(f"  L{c['layer']}H{c['head']}: gain {c['predicate_gain']} top={c['top_predicate']} ({c['top_coef']})", flush=True)

# GATE: substitute coded patterns at programmatic heads simultaneously; dCE on held slice
PROGSET = {(c['layer'], c['head']) for c in prog}
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

json.dump({'census': census, 'programmatic': prog}, open(f'{QK}/qk_census_bilin12.json', 'w'), indent=2)
print(f'bilin12 programmatic: {len(prog)}/{NH*NL}; KEY_cap heads:', [(c['layer'],c['head']) for c in prog if c['top_predicate']=='KEY_cap'], flush=True)
print('MATCH_same heads:', [(c['layer'],c['head']) for c in prog if c['top_predicate']=='MATCH_same'], flush=True)
print('QK CENSUS BILIN12 DONE', flush=True)
