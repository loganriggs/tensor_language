"""Control for the minimal circuits: is the ~90% retention selective, or would ANY same-size set of
components retain the task? For each circuit, sample random component-sets of the SAME size (keep
those, mean-ablate the rest) and measure task retention. If random sets retain far less than the
found minimal circuit, the circuit is genuinely selective (positive control for the pruning).
Runs all three tasks: induction (repeated-prefix advantage), subword & punctuation (CE reduction).
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
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]

tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
def vocab_mask(kind):
    msk = torch.zeros(V, dtype=torch.bool)
    for i in range(50257):
        s = tok.convert_ids_to_tokens(i)
        if s is None: continue
        if kind == 'sub':
            if not s.startswith('Ġ') and len(s) and s[0].isalpha() and s[0].islower(): msk[i] = True
        else:
            core = s.replace('Ġ', '')
            if len(core) and all(c in _P for c in core): msk[i] = True
    return msk.to(DEV)
CONT_SUB = vocab_mask('sub'); CONT_PUN = vocab_mask('pun')


@torch.no_grad()
def run_forward(EV, keep, MEAN, collect_mean=False):
    idx = EV[:, :-1]; B, T = idx.shape
    dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
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
        if collect_mean: means[('h', li)] = yh4.mean((0, 1))
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mean: means[('m', li)] = mo.mean((0, 1))
        if keep is not None and ('m', li) not in keep: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
    return lg


def eval_induction(keep, MEAN, EV, FIR, SEC):
    lg = run_forward(EV, keep, MEAN); tgt = EV[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item()

def eval_ce_on(keep, MEAN, EV, cmask):
    lg = run_forward(EV, keep, MEAN); tgt = EV[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(EV.shape[0], -1)
    return ce[cmask].mean().item()

gen = torch.Generator().manual_seed(7)
def rand_sets(size, n):
    out = []
    for _ in range(n):
        perm = torch.randperm(len(ALL), generator=gen)[:size]
        out.append(set(ALL[i] for i in perm.tolist()))
    return out

res = {}

# ---- induction ----
P = 64; NSEQ = 48
pref = FINEWEB[:NSEQ, 1:1+P]; EVi = torch.cat([pref, pref], 1).to(DEV)
SEC = torch.arange(P, 2*P-1, device=DEV); FIR = torch.arange(1, P-1, device=DEV)
MEANi = run_forward(EVi, None, None, collect_mean=True)[1] if False else None
# collect means via a clean pass
def collect_means(EV):
    idx = EV[:, :-1]; B, T = idx.shape
    dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
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
        means[('h', li)] = yh4.mean((0, 1))
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        means[('m', li)] = mo.mean((0, 1)); x = x + mo
    return means
MEANi = collect_means(EVi)
adv_full = eval_induction(None, MEANi, EVi, FIR, SEC); adv_none = eval_induction(set(), MEANi, EVi, FIR, SEC)
KEEP = set(eval(c) for c in json.load(open(f'{QK}/qk_induction_minimal.json'))['minimal_components'])
adv_min = eval_induction(KEEP, MEANi, EVi, FIR, SEC)
rnd = [(eval_induction(s, MEANi, EVi, FIR, SEC) - adv_none) / (adv_full - adv_none) for s in rand_sets(len(KEEP), 20)]
res['induction'] = {'size': len(KEEP), 'minimal_ret': round((adv_min-adv_none)/(adv_full-adv_none), 4),
                    'random_mean': round(float(np.mean(rnd)), 4), 'random_max': round(float(np.max(rnd)), 4),
                    'random_std': round(float(np.std(rnd)), 4)}
print("induction:", res['induction'], flush=True)

# ---- subword & punctuation ----
EVn = FINEWEB[:48, :128].to(DEV); MEANn = collect_means(EVn)
for name, cmask, jf in [('subword', CONT_SUB, 'qk_subword_circuit.json'), ('punct', CONT_PUN, 'qk_punct_circuit.json')]:
    cm = cmask[EVn[:, 1:]]
    ce_full = eval_ce_on(None, MEANn, EVn, cm); ce_floor = eval_ce_on(set(), MEANn, EVn, cm)
    score = ce_floor - ce_full
    KEEP2 = set(eval(c) for c in json.load(open(f'{QK}/{jf}'))['minimal_components'])
    ce_min = eval_ce_on(KEEP2, MEANn, EVn, cm)
    rnd = [(ce_floor - eval_ce_on(s, MEANn, EVn, cm)) / score for s in rand_sets(len(KEEP2), 20)]
    res[name] = {'size': len(KEEP2), 'minimal_ret': round((ce_floor-ce_min)/score, 4),
                 'random_mean': round(float(np.mean(rnd)), 4), 'random_max': round(float(np.max(rnd)), 4),
                 'random_std': round(float(np.std(rnd)), 4)}
    print(f"{name}:", res[name], flush=True)

json.dump(res, open(f'{QK}/qk_circuit_null.json', 'w'), indent=2)
print("QK CIRCUIT NULL DONE", flush=True)
