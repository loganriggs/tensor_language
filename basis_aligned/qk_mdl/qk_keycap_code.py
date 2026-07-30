"""§35 MEANING-CODE PROTOCOL on the CAPITALIZATION predictor -- candidate 5th fully-gated MEANING site.

The KEY_cap cluster (L15H3/H4, L16H0/H1/H5) ATTENDS capitalized keys and, per the census / §T4 joint
knockout (capital-selective, +0.046), PREDICTS capitals -- the clean "selection = function" cluster.
CANDIDATE CODE (named explicitly): "attend capitalized-key positions -> boost capital-initial
next-token predictions" = a key-side predicate KEY_cap = 1[key token is capital-initial] + a
capital-token readout.

Protocol (name-as-code -> substitution gate -> dial -> extraction -> self-red-team), held-back FW[448:600],
paired per-sequence standard errors:
 1. NAME: the code above.
 2. SUBSTITUTION GATE: at the 5 cluster heads replace the FULL attention pattern with the coded predicate
    pat_coded = a * 1[key is capital-initial] + t * positional_template + c   (a,t,c fit per head on the
    held set by normal equations, then FROZEN). Held-back CE cost vs real patterns AND vs mean-ablation of
    the cluster, with paired SE, on natural text AND capital-target positions. Does the coded version keep
    the capital-vs-lowercase logit margin while staying cheap on natural text?
 3. DIAL: scale the coded capital-boost a -> s*a, s in {0,0.5,1,1.5,2}; monotone control of the capital
    margin with natural CE ~flat (induction-dial style).
 4. EXTRACTION + RED-TEAM: (a) type-blind? margin at contexts WITH vs WITHOUT capital keys; (b) positional
    confound -- is "capital" just sentence-initial/post-period? restrict to MID-SENTENCE capitals
    (proper-noun-like; prev token not a sentence-ender); (c) static-prior lesson -- does capital prediction
    SURVIVE cluster attention (mean-)ablation (a capital prior) rather than requiring the attention code?
 5. VERDICT: genuine gated nameable code (5th meaning site) or fails the gate / reduces to a positional prior?

Forward conventions copied verbatim from qk_conditional_redirect.py.
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
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)

# --- vocab masks (capital / lowercase-word readouts + sentence-ender context) ---
def build_masks():
    cap = torch.zeros(V, dtype=torch.bool); low = torch.zeros(V, dtype=torch.bool)
    endctx = torch.zeros(V, dtype=torch.bool)
    for i in range(50257):
        s = tok.convert_ids_to_tokens(i)
        if s is None: continue
        core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
        if lead and len(core) and core[0].isalpha() and core[0].isupper(): cap[i] = True   # capital-initial word
        if lead and len(core) and core[0].isalpha() and core[0].islower(): low[i] = True    # lowercase-initial word
        cc = s.replace('Ġ', '').replace('Ċ', '\n')
        if ('Ċ' in s) or (len(cc) and cc[-1] in '.!?:;'): endctx[i] = True                   # sentence-ender context token
    return cap.to(DEV), low.to(DEV), endctx.to(DEV)
CAPV, LOWV, ENDCTX = build_masks()
print(f"vocab: capital-word {int(CAPV.sum())}, lowercase-word {int(LOWV.sum())}, sentence-ender {int(ENDCTX.sum())}", flush=True)

CLUSTER = [(15, 3), (15, 4), (16, 0), (16, 1), (16, 5)]   # KEY_cap cluster (task spec)
CIDX = {lh: i for i, lh in enumerate(CLUSTER)}
NC = len(CLUSTER)
# frozen coded coefficients (a=capital-key, t=positional template, c=const) per cluster head; set by fit()
COEF_A = torch.zeros(NC, device=DEV); COEF_T = torch.zeros(NC, device=DEV); COEF_C = torch.zeros(NC, device=DEV)
TMPL = torch.zeros(NC, 127, 127, device=DEV)   # positional template per head (T=127 context length)
MEAN = {}                                       # mean head output for mean-ablation

# held-back evaluation slice
EVAL = FINEWEB[448:600][:, :128].to(DEV)        # (152,128)
NSEQ = EVAL.shape[0]; T = 127
BATCH = 8

def cap_key_mask(idx):
    """(B,T,T): entry[b,q,k] = 1 if key token idx[b,k] is capital-initial (causal), query-independent."""
    B, Tt = idx.shape
    ck = CAPV[idx].float()                                  # (B,T) is key capital-initial
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    return ck[:, None, :].expand(B, Tt, Tt) * mask.float()  # (B,T,T)

@torch.no_grad()
def forward(idx, mode='model', dial=1.0):
    """mode: 'model' (real) | 'coded' (substitute cluster patterns with the coded predicate) |
    'meanabl' (mean-ablate the cluster head outputs -- honest deletion). dial scales the coded a-coef."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    CK = cap_key_mask(idx) if mode == 'coded' else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, Tt, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        if mode == 'coded':
            pats = []
            for h in range(NH):
                if (li, h) in CIDX:
                    ci = CIDX[(li, h)]
                    coded = (dial*COEF_A[ci]) * CK + COEF_T[ci] * TMPL[ci][None] + COEF_C[ci]
                    pats.append(coded.masked_fill(~mask, 0.0))
                else: pats.append(pat[:, h])
            pat = torch.stack(pats, 1)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if mode == 'meanabl':
            for h in range(NH):
                if (li, h) in CIDX: yh4[:, :, h, :] = MEAN[(li, h)]
        x = x + a.c_proj(yh4.reshape(B, Tt, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

# ------------------------------------------------------------------ collect templates + head-output means
@torch.no_grad()
def collect():
    patsum = {lh: torch.zeros(T, T, device=DEV) for lh in CLUSTER}
    ysum = {lh: torch.zeros(HD, device=DEV) for lh in CLUSTER}; ncnt = 0
    for i in range(0, NSEQ, BATCH):
        idx = EVAL[i:i+BATCH, :-1]; B, Tt = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, Tt, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            for h in range(NH):
                if (li, h) in CIDX:
                    patsum[(li, h)] += pat[:, h].sum(0); ysum[(li, h)] += yh4[:, :, h, :].sum((0, 1))
            x = x + a.c_proj(yh4.reshape(B, Tt, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
        ncnt += B
    for lh in CLUSTER:
        TMPL[CIDX[lh]] = patsum[lh] / ncnt
        MEAN[lh] = ysum[lh] / (ncnt * T)
    print("collected positional templates + head-output means", flush=True)

# ------------------------------------------------------------------ fit frozen coded coefficients (normal eqs)
@torch.no_grad()
def fit():
    ATA = {lh: torch.zeros(3, 3, device=DEV) for lh in CLUSTER}
    ATy = {lh: torch.zeros(3, device=DEV) for lh in CLUSTER}
    for i in range(0, NSEQ, BATCH):
        idx = EVAL[i:i+BATCH, :-1]; B, Tt = idx.shape
        CK = cap_key_mask(idx)
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool)); mb = mask.expand(B, Tt, Tt)
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, Tt, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            for h in range(NH):
                if (li, h) in CIDX:
                    ci = CIDX[(li, h)]
                    Xa = CK[mb]; Xt = TMPL[ci][None].expand(B, Tt, Tt)[mb]; Xc = torch.ones_like(Xa)
                    Xf = torch.stack([Xa, Xt, Xc], 1)                       # (n,3)
                    y = pat[:, h][mb]                                        # (n,)
                    ATA[(li, h)] += Xf.T @ Xf; ATy[(li, h)] += Xf.T @ y
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh4.reshape(B, Tt, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    for lh in CLUSTER:
        sol = torch.linalg.solve(ATA[lh] + 1e-6*torch.eye(3, device=DEV), ATy[lh])
        ci = CIDX[lh]; COEF_A[ci], COEF_T[ci], COEF_C[ci] = sol[0], sol[1], sol[2]
    print("fitted coded coefficients (a=cap-key, t=template, c=const):", flush=True)
    for lh in CLUSTER:
        ci = CIDX[lh]
        print(f"  L{lh[0]}H{lh[1]}: a={COEF_A[ci]:+.4f}  t={COEF_T[ci]:+.4f}  c={COEF_C[ci]:+.4f}", flush=True)

# ------------------------------------------------------------------ per-sequence CE + capital margin arrays
@torch.no_grad()
def eval_arrays(mode='model', dial=1.0):
    """Returns per-seq arrays: CE (NSEQ,T), margin (NSEQ,T) = mean capital-logit - mean lowercase-logit."""
    CE = torch.zeros(NSEQ, T); MG = torch.zeros(NSEQ, T)
    for i in range(0, NSEQ, BATCH):
        idx = EVAL[i:i+BATCH, :-1]; tgt = EVAL[i:i+BATCH, 1:]
        lg = forward(idx, mode, dial).float()
        ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(idx.shape[0], T)
        mg = lg[:, :, CAPV].mean(-1) - lg[:, :, LOWV].mean(-1)
        CE[i:i+idx.shape[0]] = ce.cpu(); MG[i:i+idx.shape[0]] = mg.cpu()
    return CE, MG

# position masks (computed once, on cpu)
def pos_masks():
    tgt = EVAL[:, 1:].cpu(); ctx = EVAL[:, :-1].cpu()           # (NSEQ,T)
    cap_tgt = CAPV.cpu()[tgt]                                    # next token is capital-initial
    sent_init = ENDCTX.cpu()[ctx]                                # context token is a sentence-ender
    capkey_ctx = CAPV.cpu()[ctx]                                 # (NSEQ,T) context token itself is capital
    has_capkey = torch.zeros(NSEQ, T, dtype=torch.bool)
    for s in range(NSEQ):
        seen = False
        for q in range(T):
            if capkey_ctx[s, q]: seen = True
            has_capkey[s, q] = seen                              # any capital key at/before position q
    midcap = cap_tgt & (~sent_init)                             # mid-sentence (proper-noun-like) capital target
    sentcap = cap_tgt & sent_init                               # sentence-initial capital target
    return {'cap_tgt': cap_tgt, 'midcap': midcap, 'sentcap': sentcap, 'has_capkey': has_capkey}
PM = pos_masks()
print(f"probe positions -- cap targets {int(PM['cap_tgt'].sum())}, mid-sentence caps {int(PM['midcap'].sum())}, "
      f"sentence-initial caps {int(PM['sentcap'].sum())}", flush=True)

def paired_se(a, b, sel):
    """paired per-sequence SE of (a-b) restricted to positions in bool mask sel. a,b: (NSEQ,T)."""
    diff = (a - b); per = torch.stack([diff[s][sel[s]].mean() if sel[s].any() else torch.tensor(float('nan'))
                                       for s in range(NSEQ)])
    per = per[~torch.isnan(per)]
    return float(per.mean()), float(per.std() / (len(per)**0.5)), len(per)

def seq_mean(a, sel):
    per = torch.stack([a[s][sel[s]].mean() if sel[s].any() else torch.tensor(float('nan')) for s in range(NSEQ)])
    per = per[~torch.isnan(per)]
    return float(per.mean()), float(per.std()/(len(per)**0.5))

# ================================================================== RUN
collect(); fit()
res = {'cluster': [list(c) for c in CLUSTER],
       'coef': {f'L{lh[0]}H{lh[1]}': [round(float(COEF_A[CIDX[lh]]), 4), round(float(COEF_T[CIDX[lh]]), 4),
                                      round(float(COEF_C[CIDX[lh]]), 4)] for lh in CLUSTER},
       'n_probe': {'cap_tgt': int(PM['cap_tgt'].sum()), 'midcap': int(PM['midcap'].sum()),
                   'sentcap': int(PM['sentcap'].sum())}}

ALLPOS = torch.ones(NSEQ, T, dtype=torch.bool)
CE_m, MG_m = eval_arrays('model'); CE_c, MG_c = eval_arrays('coded', 1.0); CE_a, MG_a = eval_arrays('meanabl')

# ---- 2. SUBSTITUTION GATE: CE cost vs real, and vs mean-ablation, natural + capital-target, paired SE
gate = {}
for name, sel in [('natural', ALLPOS), ('cap_target', PM['cap_tgt']), ('midsentence_cap', PM['midcap'])]:
    dm, sm = seq_mean(CE_m, sel)
    dc, dc_se, _ = paired_se(CE_c, CE_m, sel)   # coded - model
    da, da_se, _ = paired_se(CE_a, CE_m, sel)   # meanabl - model
    # coded vs mean-ablation (does the code beat the honest deletion?)
    cva, cva_se, _ = paired_se(CE_c, CE_a, sel)
    gate[name] = {'model_CE': round(dm, 4), 'coded_dCE': round(dc, 5), 'coded_dCE_SE': round(dc_se, 5),
                  'meanabl_dCE': round(da, 5), 'meanabl_dCE_SE': round(da_se, 5),
                  'coded_minus_meanabl': round(cva, 5), 'coded_minus_meanabl_SE': round(cva_se, 5)}
res['substitution_gate'] = gate

# capital margin retained by the code? (higher margin = boosts capitals)
marg = {}
for name, sel in [('cap_target', PM['cap_tgt']), ('midsentence_cap', PM['midcap']), ('sentence_initial_cap', PM['sentcap'])]:
    mm, mm_se = seq_mean(MG_m, sel); mc, _ = seq_mean(MG_c, sel); ma, _ = seq_mean(MG_a, sel)
    marg[name] = {'model_margin': round(mm, 3), 'coded_margin': round(mc, 3), 'meanabl_margin': round(ma, 3),
                  'model_margin_SE': round(mm_se, 3)}
res['capital_margin'] = marg

# ---- 3. DIAL: scale coded capital-boost; monotone margin, natural CE flat
dial = {}
for s in (0.0, 0.5, 1.0, 1.5, 2.0):
    CEs, MGs = eval_arrays('coded', s)
    nat, _ = seq_mean(CEs, ALLPOS); mg_cap, _ = seq_mean(MGs, PM['cap_tgt']); mg_mid, _ = seq_mean(MGs, PM['midcap'])
    dial[f's={s}'] = {'natural_CE': round(nat, 4), 'margin_cap_target': round(mg_cap, 3), 'margin_midcap': round(mg_mid, 3)}
    print(f"dial s={s}: natural_CE {round(nat,4)}  margin(cap) {round(mg_cap,3)}  margin(midcap) {round(mg_mid,3)}", flush=True)
res['dial'] = dial

# ---- 4. RED-TEAM
red = {}
# (a) type-blindness: coded margin at contexts WITH vs WITHOUT capital keys (code is key-predicate -> should differ)
with_ck = PM['cap_tgt'] & PM['has_capkey']; no_ck = PM['cap_tgt'] & (~PM['has_capkey'])
mc_w, _ = seq_mean(MG_c, with_ck); mc_n, _ = seq_mean(MG_c, no_ck)
mm_w, _ = seq_mean(MG_m, with_ck); mm_n, _ = seq_mean(MG_m, no_ck)
red['type_blindness'] = {'n_with_capkey': int(with_ck.sum()), 'n_no_capkey': int(no_ck.sum()),
                         'coded_margin_with_capkey': round(mc_w, 3), 'coded_margin_no_capkey': round(mc_n, 3),
                         'model_margin_with_capkey': round(mm_w, 3), 'model_margin_no_capkey': round(mm_n, 3)}
# (b) positional confound: gate cost + margin at mid-sentence vs sentence-initial caps (already in gate/marg)
red['positional_confound'] = {
    'midsentence_cap_gate': gate['midsentence_cap'],
    'margin_midcap': marg['midsentence_cap'], 'margin_sentinit_cap': marg['sentence_initial_cap']}
# (c) static-prior: does capital prediction SURVIVE cluster attention (mean-)ablation?
sp = {}
for name, sel in [('cap_target', PM['cap_tgt']), ('midsentence_cap', PM['midcap'])]:
    mm, _ = seq_mean(MG_m, sel); ma, _ = seq_mean(MG_a, sel)
    drop, drop_se, _ = paired_se(MG_a, MG_m, sel)   # meanabl - model (negative = margin drops when ablated)
    sp[name] = {'model_margin': round(mm, 3), 'meanabl_margin': round(ma, 3),
                'margin_change_on_ablation': round(drop, 3), 'SE': round(drop_se, 3),
                'frac_retained': round(ma/mm, 3) if mm != 0 else None}
red['static_prior'] = sp
res['red_team'] = red

json.dump(res, open(f'{QK}/qk_keycap_code.json', 'w'), indent=2)
print("\n===== SUBSTITUTION GATE =====", flush=True)
print(json.dumps(gate, indent=2), flush=True)
print("\n===== CAPITAL MARGIN =====", flush=True)
print(json.dumps(marg, indent=2), flush=True)
print("\n===== RED-TEAM =====", flush=True)
print(json.dumps(red, indent=2), flush=True)
print("\nQK KEYCAP CODE DONE", flush=True)
