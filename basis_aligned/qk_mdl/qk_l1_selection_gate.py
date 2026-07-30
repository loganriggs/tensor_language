"""LAYER-1 GATED SELECTION VOCABULARY: converts §6b's descriptive layer-1 archetype names into a
fully-gated selection vocabulary — the first fully-gated selection names ABOVE layer 0. The nine
validated layer-1 archetype clusters (one per head, read off qk_l1_stage23's top archetypes / §6b) are
coded as EXPLICIT key-token selection predicates:
  h0  term-punct sentence boundary   (KTERM: pure-punct token carrying '.','!','?')  archetypes ., ). ".
  h1  mid-word subword fragment      (KSUB : no-leading-space alphabetic-lowercase)  archetypes cknowled, theless
  h2  newline / document boundary    (KNL  : contains 'Ċ' or <|endoftext|>)          archetypes \n, <|endoftext|>
  h3  comma / clause-internal        (KCOMMA: core == ',')                            archetypes , ' ,'
  h4  term-punct sentence boundary   (KTERM)                                          archetypes ., ?"., !).
  h5  newline / rare                 (KNL)                                            archetypes \n, <|endoftext|>
  h6  newline / document boundary    (KNL)                                            archetypes \n, <|endoftext|>
  h7  term-punct sentence boundary   (KTERM)                                          archetypes !)., ?)., .
  h8  newline / quote / bracket      (KQB : KNL or quote or bracket)                  archetypes \n, "' , ['
Each head's layer-1 pattern is fit by least squares against [its archetype predicate, the head's
positional TEMPLATE, causal-bias]; held-out predicate gain per cluster = R^2(full) - R^2(template-only)
on fresh cooc rows. Then the GATE: substitute all nine coded layer-1 patterns SIMULTANEOUSLY; dCE with
paired per-token SE on the held-back slice FW[448:600].

DERIVATION: copied from qk_selection_census_v2.py (its templates / normal-equations / held-out-R^2 /
forward_coded simultaneous-substitution gate machinery is reused verbatim). CHANGES: (1) everything is
restricted to LAYER 1 only (pattern_l1 returns just the layer-1 pattern; the gate substitutes at
li == 1); (2) instead of the generic 12-predicate battery fit to every head, each of the nine heads is
coded with its OWN archetype key-class predicate (a per-head single selection feature, K=3 design
[archetype, template, causal]); (3) PROGSET = all nine layer-1 heads (the coded selection vocabulary).
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
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
EOT = 50256   # gpt2 <|endoftext|>
_TERM = {'.', '!', '?'}
# archetype key-token classes (built like census v2's KP/KF/... loop)
KTERM = torch.zeros(V, dtype=torch.bool); KSUB = torch.zeros(V, dtype=torch.bool)
KNL = torch.zeros(V, dtype=torch.bool); KCOMMA = torch.zeros(V, dtype=torch.bool)
KQUOTE = torch.zeros(V, dtype=torch.bool); KBRACK = torch.zeros(V, dtype=torch.bool)
for i in range(50257):
    ss = tok.convert_ids_to_tokens(i)
    if ss is None: continue
    core = ss.replace('Ġ', ''); lead = ss.startswith('Ġ')
    if len(core) and all(c in _P for c in core) and any(c in _TERM for c in core): KTERM[i] = True
    if not lead and len(core) and core[0].isalpha() and core[0].islower(): KSUB[i] = True
    if 'Ċ' in ss: KNL[i] = True
    if core == ',': KCOMMA[i] = True
    if core in ('"', "'", '"""', "''"): KQUOTE[i] = True
    if core in ('(', ')', '[', ']', '{', '}'): KBRACK[i] = True
KNL[EOT] = True
KQB = KNL | KQUOTE | KBRACK
KTERM, KSUB, KNL, KCOMMA, KQB = (t.to(DEV) for t in (KTERM, KSUB, KNL, KCOMMA, KQB))
# per-head archetype predicate assignment (the nine clusters)
ARCH = [KTERM, KSUB, KNL, KCOMMA, KTERM, KNL, KNL, KTERM, KQB]
ARCHNAME = ['term_punct_boundary', 'subword_fragment', 'newline_docbound', 'comma_clause',
            'term_punct_boundary', 'newline_rare', 'newline_docbound', 'term_punct_boundary',
            'newline_quote_bracket']
T0 = 128
K = 3   # design: [archetype predicate, TEMPLATE, causal-bias]


@torch.no_grad()
def pattern_l1(idx):
    """layer-1 (B,NH,T,T) pattern from the real forward (run through layer 1)."""
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(2):
        b = m.transformer.h[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        if li == 1:
            return pat
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + b.mlp(F.rms_norm(x, (D,)))


# first pass: per-head positional templates (mean layer-1 pattern)
TPL1 = torch.zeros(NH, T0, T0, device=DEV); nb = 0
for i in range(0, 48, 8):
    pat = pattern_l1(COOC[i:i+8].to(DEV)[:, :T0])
    TPL1 += pat.mean(0); nb += 1
TPL1 /= nb
print("layer-1 templates ready", flush=True)

# second pass: per-head normal equations, design [archetype, template, causal]
AtA = torch.zeros(NH, K, K, device=DEV, dtype=torch.float64)
Aty = torch.zeros(NH, K, device=DEV, dtype=torch.float64)
for i in range(48, 168, 8):
    idx = COOC[i:i+8].to(DEV)[:, :T0]; B = idx.shape[0]
    causal = torch.tril(torch.ones(T0, T0, device=DEV, dtype=torch.bool))
    pat = pattern_l1(idx)
    mask_flat = causal.expand(B, T0, T0).reshape(-1)
    cf = causal.expand(B, T0, T0).reshape(B, -1).float()
    for h in range(NH):
        af = (ARCH[h][idx].float().unsqueeze(1).expand(B, T0, T0) * causal).reshape(B, -1)
        tplf = (TPL1[h].unsqueeze(0).expand(B, T0, T0) * causal).reshape(B, -1)
        X = torch.stack([af, tplf, cf], 1)                      # (B,K,T0*T0)
        Xf = X.permute(0, 2, 1).reshape(-1, K)[mask_flat].double()
        yf = pat[:, h].reshape(-1)[mask_flat].double()
        AtA[h] += Xf.T @ Xf; Aty[h] += Xf.T @ yf
W = torch.linalg.solve(AtA + 1e-6*torch.eye(K, device=DEV, dtype=torch.float64), Aty.unsqueeze(-1)).squeeze(-1).float()
print("normal equations done", flush=True)


# held-out R^2: archetype+template vs template-only
@torch.no_grad()
def heldout_r2():
    ss_res = torch.zeros(NH, device=DEV); ss_tpl = torch.zeros(NH, device=DEV); ss_tot = torch.zeros(NH, device=DEV)
    for i in range(200, 240, 8):
        idx = COOC[i:i+8].to(DEV)[:, :T0]; B = idx.shape[0]
        causal = torch.tril(torch.ones(T0, T0, device=DEV, dtype=torch.bool))
        pat = pattern_l1(idx)
        for h in range(NH):
            af = ARCH[h][idx].float().unsqueeze(1).expand(B, T0, T0) * causal
            tpl = TPL1[h].unsqueeze(0) * causal
            y = pat[:, h]
            pred = W[h, 0]*af + W[h, 1]*tpl + W[h, 2]*causal
            pred_t = W[h, 1]*tpl + W[h, 2]*causal
            mu = y[:, causal].mean()
            ss_res[h] += ((pred - y)[:, causal]**2).sum(); ss_tpl[h] += ((pred_t - y)[:, causal]**2).sum()
            ss_tot[h] += ((y[:, causal] - mu)**2).sum()
    return 1 - ss_res/ss_tot, 1 - ss_tpl/ss_tot
R2full, R2tpl = heldout_r2()
gain = (R2full - R2tpl)
clusters = []
for h in range(NH):
    clusters.append({'head': h, 'cluster': ARCHNAME[h], 'r2_full': round(float(R2full[h]), 3),
                     'r2_template': round(float(R2tpl[h]), 3), 'predicate_gain': round(float(gain[h]), 3),
                     'arch_coef': round(float(W[h, 0]), 4)})
    print(f"  L1H{h} [{ARCHNAME[h]}]: gain {float(gain[h]):.3f} (full {float(R2full[h]):.3f} / tpl {float(R2tpl[h]):.3f}) coef {float(W[h,0]):.4f}", flush=True)


# GATE: substitute all nine coded layer-1 patterns simultaneously; dCE with SE on held slice
PROGSET = {(1, h) for h in range(NH)}
@torch.no_grad()
def forward_coded(idx, subst):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
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
        if subst and li == 1:
            pat = pat.clone()
            for h in range(NH):
                af = ARCH[h][idx].float().unsqueeze(1).expand(B, T, T) * mask
                tplc = (TPL1[h][:T, :T].unsqueeze(0) * mask)
                coded = W[h, 0]*af + W[h, 1]*tplc + W[h, 2]*mask
                pat[:, h] = coded.masked_fill(~mask, 0.0)
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
gate = {'n_heads_coded': len(PROGSET), 'dCE': round(float(d.mean()), 5), 'SE': round(float(d.std()/np.sqrt(d.numel())), 6)}
print(f"GATE: all {len(PROGSET)} layer-1 archetype patterns coded simultaneously: dCE +{gate['dCE']} (SE {gate['SE']})", flush=True)
print(f"SUMMARY L1 SELECTION: mean held-out predicate gain {float(gain.mean()):.3f} "
      f"(range {float(gain.min()):.3f}..{float(gain.max()):.3f}); all-9 gate +{gate['dCE']} nats/tok (SE {gate['SE']})", flush=True)
json.dump({'clusters': clusters, 'gate': gate}, open(f'{QK}/qk_l1_selection_gate.json', 'w'), indent=2)
print("QK L1 SELECTION GATE DONE", flush=True)
