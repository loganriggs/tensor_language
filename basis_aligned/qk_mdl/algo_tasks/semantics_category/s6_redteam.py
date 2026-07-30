"""Step 6: RED-TEAM.
(a) Concentration: is the category-k mass gain under +1.0*r_med steering spread across the
    category or concentrated in its top-5 tokens? (share of total positive within-category
    prob gain captured by top-5 gaining tokens; plus which tokens).
(b) Effective dimensionality: Jacobian J[k,j] = d mass_j / d alpha from the +-0.25 s2 runs;
    singular values -> effective rank of the steering map. Plus direction cosines (s1).
(c) Breaking alphas: dCE at alpha in {2,4,8,16} x r_med (both signs, funcword dir + worst dir)
    on a 36-row held subset.
"""
import json, sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_category')
from common import (FINEWEB, HELD_ROWS, CATNAMES, CATM, tok, forward, batches, oom_retry,
                    cat_mass, per_token_ce, paired_stats, load_probe, OUT, DEV, D, V)
import torch.nn.functional as F

P = load_probe(); d_unit = P['d_unit'].to(DEV); r_med = P['r_med']
ROWS = list(HELD_ROWS)
SUB = ROWS[:36]

# ---------- (a) concentration of mass gain ----------
def mean_probs(dvec, alpha, rows):
    acc = torch.zeros(V, dtype=torch.float64); n = 0
    edit = None if dvec is None else (lambda x: x + alpha*dvec)
    for idx in batches(FINEWEB, rows):
        lg, _ = oom_retry(forward, idx[:, :-1], edit_fn=edit)
        p = F.softmax(lg, -1).double().reshape(-1, V)
        acc += p.sum(0).cpu(); n += p.shape[0]
    return acc / n

print("(a) concentration...", flush=True)
p0 = mean_probs(None, 0.0, SUB)
conc = {}
for k in range(6):
    pk = mean_probs(d_unit[:, k], 1.0*r_med, SUB)
    dp = (pk - p0)
    within = dp.clone(); within[~CATM[k].cpu()] = 0
    pos = within.clamp(min=0)
    total_gain = float(pos.sum())
    top5v, top5i = pos.topk(5)
    conc[CATNAMES[k]] = {
        'd_mass_k_total': round(float(within.sum()), 4),
        'positive_within_gain': round(total_gain, 4),
        'top5_share_of_positive_gain': round(float(top5v.sum()) / max(total_gain, 1e-9), 4),
        'top5_tokens': [tok.decode([int(i)]) for i in top5i],
        'top5_gains': [round(float(v), 4) for v in top5v],
        'n_tokens_for_half_gain': int((pos.sort(descending=True).values.cumsum(0) < 0.5*total_gain).sum()) + 1}
    print(CATNAMES[k], conc[CATNAMES[k]], flush=True)

# ---------- (b) effective rank ----------
s2 = json.load(open(f'{OUT}/s2_steering.json'))
J = torch.tensor([[(s2['named'][CATNAMES[k]]['+0.25']['d_mass'][j]
                    - s2['named'][CATNAMES[k]]['-0.25']['d_mass'][j]) / 0.5
                   for j in range(6)] for k in range(6)], dtype=torch.float64)
sv = torch.linalg.svdvals(J)
eff_rank = float((sv.sum())**2 / (sv**2).sum())   # participation-ratio effective rank
rank_b = {'jacobian_dmass_per_relalpha': [[round(J[k, j].item(), 4) for j in range(6)] for k in range(6)],
          'singular_values': [round(x, 4) for x in sv.tolist()],
          'effective_rank_participation': round(eff_rank, 3)}
print("(b) Jacobian svals:", rank_b['singular_values'], "eff rank", rank_b['effective_rank_participation'], flush=True)

# ---------- (c) breaking alphas ----------
print("(c) breaking alphas...", flush=True)
base_ce = []
for idx in batches(FINEWEB, SUB):
    lg, _ = oom_retry(forward, idx[:, :-1])
    base_ce.append(per_token_ce(lg, idx).reshape(-1).cpu())
base_ce = torch.cat(base_ce)
breaking = {}
for kname in ['funcword', 'punct']:
    k = CATNAMES.index(kname); breaking[kname] = {}
    for rel in [2.0, 4.0, 8.0, 16.0, -2.0, -4.0, -8.0]:
        ce_l = []
        edit = lambda x: x + rel*r_med*d_unit[:, k]
        for idx in batches(FINEWEB, SUB):
            lg, _ = oom_retry(forward, idx[:, :-1], edit_fn=edit)
            ce_l.append(per_token_ce(lg, idx).reshape(-1).cpu())
        dce = torch.cat(ce_l) - base_ce
        mn, _, se = paired_stats(dce, len(SUB))
        breaking[kname][f'{rel:+.1f}'] = {'dCE': round(mn, 4), 'se_row': round(se, 4)}
        print(f"  {kname} a={rel:+.1f}: dCE {mn:+.4f}±{se:.4f}", flush=True)

json.dump({'concentration_alpha1.0': conc, 'effective_rank': rank_b,
           'breaking_alphas_36rows': breaking,
           'probe_dir_cosines_see_s1': 'see s1_probe.json cos_centered_dirs'},
          open(f'{OUT}/s6_redteam.json', 'w'), indent=2)
print("S6 DONE", flush=True)
