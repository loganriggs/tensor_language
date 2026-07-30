"""Step 2: STEERING GATE. Add alpha*d_k to the block-3 residual at ALL positions on the
HELD-BACK audit set (FineWeb rows 448:600); measure the full 6x6 dose-response matrix
d(mass_j) for steering direction k, alpha swept both signs, plus paired per-token dCE+SE
(collateral cost). Control: 6 random unit directions, same alpha grid.

PASS = strong diagonal, weak off-diagonal, monotone in alpha.
"""
import json, sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_category')
from common import (FINEWEB, HELD_ROWS, CATNAMES, forward, batches, oom_retry, cat_mass,
                    per_token_ce, paired_stats, load_probe, OUT, DEV, D)

P = load_probe()
d_unit = P['d_unit'].to(DEV)          # (D,6) centered unit decision axes
r_med = P['r_med']
g = torch.Generator(device='cpu').manual_seed(123)
rand_unit = torch.randn(D, 6, generator=g)
rand_unit = (rand_unit / rand_unit.norm(dim=0, keepdim=True)).to(DEV)

REL_ALPHAS = [0.25, 0.5, 1.0, 2.0]
ALPHAS = [s*a*r_med for a in REL_ALPHAS for s in (+1, -1)]
ROWS = list(HELD_ROWS)
NR = len(ROWS)

# baseline
base_mass, base_ce = [], []
for idx in batches(FINEWEB, ROWS):
    lg, _ = oom_retry(forward, idx[:, :-1])
    base_mass.append(cat_mass(lg).reshape(-1, 6).cpu())
    base_ce.append(per_token_ce(lg, idx).reshape(-1).cpu())
base_mass = torch.cat(base_mass); base_ce = torch.cat(base_ce)
print(f"baseline: {base_ce.numel()} tokens, CE {base_ce.mean():.4f}, mean mass "
      f"{[round(x,4) for x in base_mass.mean(0).tolist()]}", flush=True)

def steer_run(dvec, alpha):
    dm, dce = [], []
    edit = lambda x: x + alpha*dvec
    for idx in batches(FINEWEB, ROWS):
        lg, _ = oom_retry(forward, idx[:, :-1], edit_fn=edit)
        dm.append(cat_mass(lg).reshape(-1, 6).cpu())
        dce.append(per_token_ce(lg, idx).reshape(-1).cpu())
    dm = torch.cat(dm) - base_mass
    dce = torch.cat(dce) - base_ce
    mean_dm = dm.mean(0)
    ce_m, ce_se_tok, ce_se_row = paired_stats(dce, NR)
    return mean_dm, (ce_m, ce_se_tok, ce_se_row)

res = {'r_med': r_med, 'rel_alphas': REL_ALPHAS, 'catnames': CATNAMES,
       'baseline_ce': round(base_ce.mean().item(), 4),
       'baseline_mass': [round(x, 4) for x in base_mass.mean(0).tolist()],
       'named': {}, 'random': {}}
for tag, dirs in [('named', d_unit), ('random', rand_unit)]:
    for k in range(6):
        kname = CATNAMES[k] if tag == 'named' else f'rand{k}'
        res[tag][kname] = {}
        for alpha in ALPHAS:
            rel = alpha / r_med
            mdm, (ce_m, se_t, se_r) = steer_run(dirs[:, k], alpha)
            res[tag][kname][f'{rel:+.2f}'] = {
                'd_mass': [round(x, 4) for x in mdm.tolist()],
                'dCE': round(ce_m, 4), 'dCE_se_tok': round(se_t, 5), 'dCE_se_row': round(se_r, 5)}
            print(f"{tag} {kname:9s} a={rel:+.2f}: dmass {[f'{x:+.4f}' for x in mdm.tolist()]} "
                  f"dCE {ce_m:+.4f}±{se_r:.4f}", flush=True)

json.dump(res, open(f'{OUT}/s2_steering.json', 'w'), indent=2)
print("S2 DONE", flush=True)
