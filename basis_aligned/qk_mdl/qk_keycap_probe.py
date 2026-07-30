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

CAP_HEADS = [(15, 3), (15, 4), (16, 0), (16, 1), (16, 5)]
EVn = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[448:600][:, :128].to(DEV)
P = 64; NSEQ = 48
prefN = FINEWEB[100:100+NSEQ, 1:1+P]; EVi = torch.cat([prefN, prefN], 1).to(DEV)
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


_, MEANn = run(EVn, None, None, True)
_, MEANi = run(EVi, None, None, True)
def ce_tasks(lg, EV):
    tgt = EV[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return {t: (float(ce[MASKS[t][tgt]].mean()) if MASKS[t][tgt].any() else None) for t in MASKS} | {'natural': float(ce.mean())}
def ind_adv(lg):
    tgt = EVi[:, 1:]; ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(NSEQ, -1)
    return float(ce[:, FIRi].mean() - ce[:, SECi].mean())
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]
base = ce_tasks(run(EVn, None, None)[0], EVn); base_adv = ind_adv(run(EVi, None, None)[0])
res = {'cluster': [list(c) for c in CAP_HEADS], 'base': base, 'base_induction': base_adv}
keep = set(ALL) - {('h', li, h) for (li, h) in CAP_HEADS}
d = ce_tasks(run(EVn, keep, MEANn)[0], EVn)
res['joint_knockout'] = {t: (round(d[t]-base[t], 5) if d[t] is not None else None) for t in d}
res['joint_induction_drop'] = round(base_adv - ind_adv(run(EVi, keep, MEANi)[0]), 4)
print('JOINT KEY_cap cluster knockout, task dCE:', sorted([(t, v) for t, v in res['joint_knockout'].items() if v], key=lambda x: -x[1])[:6], flush=True)
print('induction drop:', res['joint_induction_drop'], flush=True)
for (li, h) in CAP_HEADS:
    k1 = set(ALL) - {('h', li, h)}
    d1 = ce_tasks(run(EVn, k1, MEANn)[0], EVn)
    res[f'single_L{li}H{h}'] = {t: (round(d1[t]-base[t], 5) if d1[t] is not None else None) for t in d1}
    top = sorted([(t, v) for t, v in res[f'single_L{li}H{h}'].items() if v], key=lambda x: -x[1])[:3]
    print(f'  L{li}H{h}: {top}', flush=True)
json.dump(res, open(f'{QK}/qk_keycap_probe.json', 'w'), indent=2)
print('QK KEYCAP PROBE DONE', flush=True)
