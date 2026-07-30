"""STEP 2 of verify->patch->minimal-circuit->red-team for the "greater-of-two-digits"
behavior in bilin18 (verified acc2=0.986 few-shot, see qk_algoverify_greater_of_two.py).

Mean-ablation knockout harness. For the 72-prompt greater-of-two set we mean-ablate
(over the prompt set, an in-distribution zero point -- NOT zero-ablation) each:
  - whole attention layer (c_proj output),  li in 0..17
  - each attention head (per-head value slice y before c_proj),  (li, h)
  - each MLP,  li in 0..17
and measure the drop in behavior:
  - MARGIN = mean over prompts of logit[ larger digit ] - logit[ smaller digit ] at
    the final position (the comparison signal proper; baseline ~ positive & large).
  - acc2 = argmax restricted to {a,b} == max(a,b).
Importance of a component = baseline_margin - ablated_margin (drop). We also report acc2.

Forward is copied verbatim from tier2_model.reference_forward / qk_conditional_redirect.py
(bilinear unnormalized (q1.k1)(q2.k2)/D^2, per-head QK rms_norm then RoPE, v mixed with
block-0 v via lamb, soft-capped logits). Mean-ablation substitutes a component's output
at every position with its per-position mean over the prompt set (early positions are
identical across prompts by causality, so this only bites at the query tail).
"""
import json, sys, os
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
tok = AutoTokenizer.from_pretrained('gpt2')

def tid(s):
    t = tok(s)['input_ids']; assert len(t) == 1, s; return t[0]
D_TID = {d: tid(' ' + str(d)) for d in range(1, 10)}
FS = "7 3 -> 7\n2 8 -> 8\n"

# --- build the 72-prompt batch (all identical token length) ---
prompts, meta = [], []
for a in range(1, 10):
    for b in range(1, 10):
        if a == b: continue
        prompts.append(FS + f"{a} {b} ->")
        meta.append((a, b, max(a, b), min(a, b)))
IDS = torch.stack([tok(p, return_tensors='pt')['input_ids'][0] for p in prompts]).to(DEV)
NP, T = IDS.shape
assert len({len(tok(p)['input_ids']) for p in prompts}) == 1, "prompts differ in length"
A = torch.tensor([D_TID[a] for a, b, g, s in meta], device=DEV)   # token id of a
Bt = torch.tensor([D_TID[b] for a, b, g, s in meta], device=DEV)  # token id of b
GT = torch.tensor([D_TID[g] for a, b, g, s in meta], device=DEV)  # token id of larger
SM = torch.tensor([D_TID[s] for a, b, g, s in meta], device=DEV)  # token id of smaller
print(f"batch {NP} prompts, T={T}", flush=True)

CHUNK = 8  # respect shared-GPU footprint rule


@torch.no_grad()
def forward(idx, means=None, ablate=None, collect=None):
    """means: dict of per-position component means (T,...) for ablation.
    ablate: None | a single target | a list/set of targets, each
            ('attn', li) | ('mlp', li) | ('head', li, h).
    collect: None or a dict to accumulate per-position sums.
    Returns logits (B,T,V)."""
    if ablate is not None and isinstance(ablate, tuple):
        ablate = [ablate]
    ab = set(ablate) if ablate is not None else set()
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, Tt, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)         # (B,T,NH,HD) per-head value
        # ---- head-level mean-ablation ----
        heads_ab = [t[2] for t in ab if t[0] == 'head' and t[1] == li]
        if heads_ab:
            yh = yh.clone()
            for h in heads_ab:
                yh[:, :, h, :] = means['head'][li][:, h, :].unsqueeze(0)
        if collect is not None:
            collect['head'][li] += yh.sum(0)
        attn_out = a.c_proj(yh.reshape(B, Tt, -1))
        # ---- whole-attn mean-ablation ----
        if ('attn', li) in ab:
            attn_out = means['attn'][li].unsqueeze(0).expand(B, -1, -1)
        if collect is not None:
            collect['attn'][li] += attn_out.sum(0)
        x = x + attn_out
        mlp_out = blk.mlp(F.rms_norm(x, (D,)))
        # ---- MLP mean-ablation ----
        if ('mlp', li) in ab:
            mlp_out = means['mlp'][li].unsqueeze(0).expand(B, -1, -1)
        if collect is not None:
            collect['mlp'][li] += mlp_out.sum(0)
        x = x + mlp_out
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)


# ---- pass 1: collect per-position means over the full prompt set (chunked) ----
means_sum = {'head': [torch.zeros(T, NH, HD, device=DEV) for _ in range(NL)],
             'attn': [torch.zeros(T, D, device=DEV) for _ in range(NL)],
             'mlp':  [torch.zeros(T, D, device=DEV) for _ in range(NL)]}
for i in range(0, NP, CHUNK):
    forward(IDS[i:i+CHUNK], collect=means_sum)
means = {k: [t / NP for t in v] for k, v in means_sum.items()}


@torch.no_grad()
def run_behavior(ablate=None):
    """Full-batch (chunked) behavior under an ablation."""
    lg_all, ls_all = [], []
    for i in range(0, NP, CHUNK):
        lp = forward(IDS[i:i+CHUNK], means=means, ablate=ablate)[:, -1].float()
        lg_all.append(lp.gather(-1, GT[i:i+CHUNK].unsqueeze(-1)).squeeze(-1))
        ls_all.append(lp.gather(-1, SM[i:i+CHUNK].unsqueeze(-1)).squeeze(-1))
    lg = torch.cat(lg_all); ls = torch.cat(ls_all)
    margin = float((lg - ls).mean())
    acc2 = float((lg > ls).float().mean())
    return margin, acc2


base_margin, base_acc2 = run_behavior(None)
print(f"BASELINE margin={base_margin:.4f} acc2={base_acc2:.4f}", flush=True)

results = []
# attention layers
for li in range(NL):
    mg, ac = run_behavior(('attn', li))
    results.append({'comp': f'attn.L{li}', 'kind': 'attn', 'li': li, 'h': None,
                    'margin': round(mg, 4), 'acc2': round(ac, 4),
                    'margin_drop': round(base_margin - mg, 4), 'acc2_drop': round(base_acc2 - ac, 4)})
# MLPs
for li in range(NL):
    mg, ac = run_behavior(('mlp', li))
    results.append({'comp': f'mlp.L{li}', 'kind': 'mlp', 'li': li, 'h': None,
                    'margin': round(mg, 4), 'acc2': round(ac, 4),
                    'margin_drop': round(base_margin - mg, 4), 'acc2_drop': round(base_acc2 - ac, 4)})
# heads
for li in range(NL):
    for h in range(NH):
        mg, ac = run_behavior(('head', li, h))
        results.append({'comp': f'head.L{li}.H{h}', 'kind': 'head', 'li': li, 'h': h,
                        'margin': round(mg, 4), 'acc2': round(ac, 4),
                        'margin_drop': round(base_margin - mg, 4), 'acc2_drop': round(base_acc2 - ac, 4)})

results.sort(key=lambda r: r['margin_drop'], reverse=True)

# ---- GROUP / cumulative knockouts: diffuse vs redundant-but-localized ----
def targets_of(kind):
    if kind == 'all_attn': return [('attn', li) for li in range(NL)]
    if kind == 'all_mlp':  return [('mlp', li) for li in range(NL)]
    if kind == 'all_heads': return [('head', li, h) for li in range(NL) for h in range(NH)]
group = {}
for nm in ['all_attn', 'all_mlp', 'all_heads']:
    mg, ac = run_behavior(targets_of(nm))
    group[nm] = {'margin': round(mg, 4), 'acc2': round(ac, 4),
                 'margin_drop': round(base_margin - mg, 4), 'acc2_drop': round(base_acc2 - ac, 4)}
    print(f"GROUP {nm:10s} margin={mg:.3f} (drop {base_margin-mg:.3f}) acc2={ac:.3f}", flush=True)

# greedy cumulative: add components one at a time (by marginal margin_drop) up to 12
def target_tuple(r):
    return (r['kind'], r['li']) if r['kind'] != 'head' else ('head', r['li'], r['h'])
pool = [target_tuple(r) for r in results]
chosen, greedy = [], []
remaining = list(pool)
for step in range(12):
    best, best_mg = None, 1e9
    for cand in remaining:
        mg, ac = run_behavior(chosen + [cand])
        if mg < best_mg:
            best_mg, best_ac, best = mg, ac, cand
    chosen.append(best); remaining.remove(best)
    greedy.append({'added': str(best), 'set_size': len(chosen), 'margin': round(best_mg, 4),
                   'acc2': round(best_ac, 4), 'margin_drop': round(base_margin - best_mg, 4)})
    print(f"GREEDY +{str(best):18s} |set|={len(chosen):2d} margin={best_mg:.3f} acc2={best_ac:.3f}", flush=True)

out = {'baseline': {'margin': round(base_margin, 4), 'acc2': round(base_acc2, 4), 'n_prompts': NP, 'T': T},
       'ranking': results, 'group': group, 'greedy_cumulative': greedy}
json.dump(out, open(f'{QK}/qk_gtwo_patch.json', 'w'), indent=2)

print("\n=== TOP 25 by margin_drop (mean-ablation knockout) ===", flush=True)
print(f"{'component':16s} {'margin':>8s} {'m_drop':>8s} {'acc2':>6s} {'a_drop':>7s}")
for r in results[:25]:
    print(f"{r['comp']:16s} {r['margin']:8.3f} {r['margin_drop']:8.3f} {r['acc2']:6.3f} {r['acc2_drop']:7.3f}", flush=True)
tot = sum(max(r['margin_drop'], 0) for r in results)
top5 = sum(max(r['margin_drop'], 0) for r in results[:5])
top10 = sum(max(r['margin_drop'], 0) for r in results[:10])
print(f"\nsum(pos margin_drop)={tot:.3f}  top5 share={top5/tot:.3f}  top10 share={top10/tot:.3f}", flush=True)
print("DONE", flush=True)
