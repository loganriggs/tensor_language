"""FUNCTIONAL ATLAS: per-component importance across a task battery, in one sweep.
For each of the model's 180 components (162 heads + 18 MLPs), mean-ablate it and score EVERY task
from the same forward. Tasks (target = next token): subword-continuation, punctuation, capitalized-
word, digit, newline, function-word (vocab masks, shared forward) + induction (separate repeated
eval). Importance[task, comp] = task-score drop when comp is removed. The tasks x components matrix
exposes modular structure: universal vs task-specific components, and whether tasks cluster.
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
MASKS = build_masks()
VOCAB_TASKS = list(MASKS.keys())
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
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
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
    out = {}
    for t in VOCAB_TASKS:
        cm = MASKS[t][tgt]
        out[t] = ce[cm].mean().item() if cm.any() else float('nan')
    return out, ce

def induction_adv(lg):
    tgt = EVi[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EVi.shape[0], -1)
    return ce[:, FIRi].mean().item() - ce[:, SECi].mean().item()

# means (ablation targets) collected per eval
_, MEANn = run(EVn, None, None, True)
_, MEANi = run(EVi, None, None, True)
lg_full_n, _ = run(EVn, None, None); base_ce, _ = ce_per_task(lg_full_n, EVn)
lg_floor_n, _ = run(EVn, set(), MEANn); floor_ce, _ = ce_per_task(lg_floor_n, EVn)
lg_full_i, _ = run(EVi, None, None); adv_full = induction_adv(lg_full_i)
lg_floor_i, _ = run(EVi, set(), MEANi); adv_floor = induction_adv(lg_floor_i)
counts = {t: int(MASKS[t][EVn[:, 1:]].sum()) for t in VOCAB_TASKS}
print("target counts:", counts, flush=True)
print("full CE:", {t: round(base_ce[t], 3) for t in VOCAB_TASKS}, "| induction adv", round(adv_full, 3), flush=True)

TASKS = VOCAB_TASKS + ['induction']
# importance matrix: TASKS x components
IMP = {t: {} for t in TASKS}
for c in ALL:
    keep = set(ALL) - {c}
    lgn, _ = run(EVn, keep, MEANn); ce_t, _ = ce_per_task(lgn, EVn)
    for t in VOCAB_TASKS:
        IMP[t][c] = ce_t[t] - base_ce[t]        # CE rise when removed (higher=more important)
    lgi, _ = run(EVi, keep, MEANi)
    IMP['induction'][c] = adv_full - induction_adv(lgi)
print("importance matrix computed", flush=True)

# normalize each task by its total task score for comparability
score = {t: (floor_ce[t] - base_ce[t]) for t in VOCAB_TASKS}; score['induction'] = adv_full - adv_floor
def vec(t): return np.array([IMP[t][c] / (score[t] + 1e-9) for c in ALL])
# task-task correlation of importance profiles
corr = {}
for a in TASKS:
    for b in TASKS:
        if a < b:
            va, vb = vec(a), vec(b); corr[f"{a}~{b}"] = round(float(np.corrcoef(va, vb)[0, 1]), 3)
# universal components: mean normalized importance across tasks
mean_imp = {c: float(np.mean([IMP[t][c] / (score[t] + 1e-9) for t in TASKS])) for c in ALL}
universal = sorted(ALL, key=lambda c: -mean_imp[c])[:10]
# per task top-5 and head/MLP mass
def head_mlp_mass(t):
    hm = sum(max(0, IMP[t][c]) for c in ALL if c[0] == 'h'); mm = sum(max(0, IMP[t][c]) for c in ALL if c[0] == 'm')
    tot = hm + mm + 1e-9; return round(hm/tot, 3), round(mm/tot, 3)
top5 = {t: [str(c) for c in sorted(ALL, key=lambda c: -IMP[t][c])[:5]] for t in TASKS}
masses = {t: head_mlp_mass(t) for t in TASKS}
print("\nUNIVERSAL components (mean importance):", [str(c) for c in universal], flush=True)
print("\ntask-task importance correlations:", flush=True)
for k, v in sorted(corr.items(), key=lambda x: -x[1]): print(f"  {k}: {v}", flush=True)
print("\nhead/MLP importance mass per task:", flush=True)
for t in TASKS: print(f"  {t}: heads {masses[t][0]:.2f} / MLP {masses[t][1]:.2f}   top5 {top5[t]}", flush=True)

res = {'target_counts': counts, 'full_ce': {t: round(base_ce[t], 4) for t in VOCAB_TASKS},
       'task_score': {t: round(score[t], 4) for t in TASKS}, 'induction_adv': round(adv_full, 4),
       'universal_components': [str(c) for c in universal],
       'task_correlation': corr, 'head_mlp_mass': {t: masses[t] for t in TASKS}, 'top5_per_task': top5,
       'importance_matrix': {t: {str(c): round(IMP[t][c], 4) for c in ALL} for t in TASKS}}
json.dump(res, open(f'{QK}/qk_circuit_atlas.json', 'w'), indent=2)
print("\nQK CIRCUIT ATLAS DONE", flush=True)
