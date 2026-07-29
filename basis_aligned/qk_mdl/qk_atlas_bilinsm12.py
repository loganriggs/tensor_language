"""Generality test: run the functional atlas on a SECOND model, bilin12
(gpt2-bilinear-sqrd-attn-12l-6h-768embd) -- bilinear MLP like bilin18 but SINGLE-BRANCH squared,
NORMALIZED attention (different attention family). Does the three-family structure (category=early
MLP, induction=attention copy, layout=outlier) reproduce in a different bilinear-MLP transformer?
84 components (72 heads + 12 MLPs). Same 7-task battery + induction.
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
m, cfg = load_elriggs('bilinsm12')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
print(f"bilinsm12: NH{NH} HD{HD} D{D} NL{NL} V{V}", flush=True)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which','you','have','he','they','has'}

def build_masks():
    masks = {k: torch.zeros(V, dtype=torch.bool) for k in ['subword','punct','capital','digit','newline','funcword']}
    for i in range(50257):
        s = tok.convert_ids_to_tokens(i)
        if s is None: continue
        core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
        if not lead and len(core) and core[0].isalpha() and core[0].islower(): masks['subword'][i] = True
        if len(core) and all(c in _P for c in core): masks['punct'][i] = True
        if lead and len(core) and core[0].isupper(): masks['capital'][i] = True
        if len(core) and all(c.isdigit() for c in core): masks['digit'][i] = True
        if 'Ċ' in s or '\n' in s: masks['newline'][i] = True
        if core.lower() in FUNC: masks['funcword'][i] = True
    return {k: v.to(DEV) for k, v in masks.items()}
MASKS = build_masks(); VOCAB_TASKS = list(MASKS.keys())
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]

EVn = FINEWEB[:64, :128].to(DEV)
P = 64; NSEQ = 48
pref = FINEWEB[100:100+NSEQ, 1:1+P]; EVi = torch.cat([pref, pref], 1).to(DEV)
SECi = torch.arange(P, 2*P-1, device=DEV); FIRi = torch.arange(1, P-1, device=DEV)


@torch.no_grad()
def run(EV, keep, MEAN, collect=False):
    idx = EV[:, :-1]; B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); means = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        qh, kh = qk(a.c_q), qk(a.c_k)                              # (B,T,NH,HD) rms+rotary
        y = F.scaled_dot_product_attention(qh.transpose(1,2), kh.transpose(1,2), v.transpose(1,2), is_causal=True)
        yh4 = y.transpose(1, 2)                                     # (B,T,NH,HD) STANDARD SOFTMAX attention
        if collect: means[('h', li)] = yh4.mean((0, 1))
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect: means[('m', li)] = mo.mean((0, 1))
        if keep is not None and ('m', li) not in keep: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
    return lg, means

def ce_per_task(lg, EV):
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return {t: (ce[MASKS[t][tgt]].mean().item() if MASKS[t][tgt].any() else float('nan')) for t in VOCAB_TASKS}, ce
def induction_adv(lg):
    tgt = EVi[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EVi.shape[0], -1)
    return ce[:, FIRi].mean().item() - ce[:, SECi].mean().item()

_, MEANn = run(EVn, None, None, True); _, MEANi = run(EVi, None, None, True)
lgn, _ = run(EVn, None, None); base_ce, _ = ce_per_task(lgn, EVn)
lgf, _ = run(EVn, set(), MEANn); floor_ce, _ = ce_per_task(lgf, EVn)
lgi, _ = run(EVi, None, None); adv_full = induction_adv(lgi)
lgif, _ = run(EVi, set(), MEANi); adv_floor = induction_adv(lgif)
# base CE all positions (competence)
allce = F.cross_entropy(lgn.reshape(-1, V), EVn[:, 1:].reshape(-1)).item()
print(f"bilinsm12 base CE(128) {allce:.3f} | induction adv {adv_full:.3f}", {t: round(base_ce[t],2) for t in VOCAB_TASKS}, flush=True)

TASKS = VOCAB_TASKS + ['induction']
IMP = {t: {} for t in TASKS}
for c in ALL:
    keep = set(ALL) - {c}
    l1, _ = run(EVn, keep, MEANn); ct, _ = ce_per_task(l1, EVn)
    for t in VOCAB_TASKS: IMP[t][c] = ct[t] - base_ce[t]
    l2, _ = run(EVi, keep, MEANi); IMP['induction'][c] = adv_full - induction_adv(l2)
print("importance matrix computed", flush=True)

score = {t: (floor_ce[t] - base_ce[t]) for t in VOCAB_TASKS}; score['induction'] = adv_full - adv_floor
def vec(t): return np.array([IMP[t][c] / (score[t] + 1e-9) for c in ALL])
corr = {}
for a in TASKS:
    for b in TASKS:
        if a < b: corr[f"{a}~{b}"] = round(float(np.corrcoef(vec(a), vec(b))[0, 1]), 3)
mean_imp = {c: float(np.mean([IMP[t][c]/(score[t]+1e-9) for t in TASKS])) for c in ALL}
universal = sorted(ALL, key=lambda c: -mean_imp[c])[:8]
def mass(t):
    hm = sum(max(0, IMP[t][c]) for c in ALL if c[0] == 'h'); mm = sum(max(0, IMP[t][c]) for c in ALL if c[0] == 'm')
    tot = hm+mm+1e-9; return round(hm/tot, 3), round(mm/tot, 3)
masses = {t: mass(t) for t in TASKS}
top5 = {t: [str(c) for c in sorted(ALL, key=lambda c: -IMP[t][c])[:5]] for t in TASKS}
print("\nUNIVERSAL:", [str(c) for c in universal], flush=True)
print("task correlations:", flush=True)
for k, v in sorted(corr.items(), key=lambda x: -x[1]): print(f"  {k}: {v}", flush=True)
print("head/MLP mass:", flush=True)
for t in TASKS: print(f"  {t}: heads {masses[t][0]:.2f}/MLP {masses[t][1]:.2f}  top5 {top5[t]}", flush=True)

res = {'base_ce': round(allce, 4), 'induction_adv': round(adv_full, 4),
       'full_ce': {t: round(base_ce[t], 4) for t in VOCAB_TASKS}, 'task_score': {t: round(score[t], 4) for t in TASKS},
       'universal': [str(c) for c in universal], 'task_correlation': corr,
       'head_mlp_mass': masses, 'top5_per_task': top5,
       'importance_matrix': {t: {str(c): round(IMP[t][c], 4) for c in ALL} for t in TASKS}}
json.dump(res, open(f'{QK}/qk_atlas_bilinsm12.json', 'w'), indent=2)
print("\nQK ATLAS BILINSM12 DONE", flush=True)
