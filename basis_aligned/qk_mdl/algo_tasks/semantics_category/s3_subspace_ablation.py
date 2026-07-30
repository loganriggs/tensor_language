"""Step 3: SUBSPACE ABLATION on the held-back audit set. Project the probe subspace out of
the block-3 residual at all positions; decompose damage with the exact split
CE = categoryCE + withinCE. Direction-level prediction: categoryCE takes the hit
disproportionately (unlike the FALSIFIED layer-level MLP0-3 claim, ratio 0.27 vs control 0.36).
Controls: 5 random 6-dim subspaces + a 'mean-preserving' variant (project out but restore
per-token mean component? No: we project the DEVIATION only is not defined -- we use plain
orthogonal projection of the residual; random subspaces give the size-matched control).

Subspace = span of the 6 RAW probe columns (rank ~5.x; also report 5-dim centered version).
"""
import json, sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_category')
from common import (FINEWEB, HELD_ROWS, CATNAMES, forward, batches, oom_retry,
                    per_token_ce, cat_within_ce, paired_stats, load_probe, OUT, DEV, D)

P = load_probe()
ROWS = list(HELD_ROWS); NR = len(ROWS)

def orth(cols):
    Q, _ = torch.linalg.qr(cols.double())
    return Q.float().to(DEV)  # (D, r)

U_raw = orth(P['Wd'])                # 6-dim (numerically; 6th sv tiny)
U_cent = orth(P['d_cent'])[:, :5]    # centered axes are exactly rank 5
gcpu = torch.Generator().manual_seed(777)
rand_subs = [orth(torch.randn(D, 6, generator=gcpu)) for _ in range(5)]

def eval_proj(U):
    """project span(U) out of blk3 residual at all positions; return paired deltas."""
    edit = (lambda x: x - (x @ U) @ U.T) if U is not None else None
    ce_l, cat_l, win_l = [], [], []
    for idx in batches(FINEWEB, ROWS):
        lg, _ = oom_retry(forward, idx[:, :-1], edit_fn=edit)
        ce_l.append(per_token_ce(lg, idx).reshape(-1).cpu())
        c, w = cat_within_ce(lg, idx)
        cat_l.append(c.reshape(-1).cpu()); win_l.append(w.reshape(-1).cpu())
    return torch.cat(ce_l), torch.cat(cat_l), torch.cat(win_l)

print("baseline...", flush=True)
ce0, cat0, win0 = eval_proj(None)
res = {'baseline': {'CE': round(ce0.mean().item(), 4), 'categoryCE': round(cat0.mean().item(), 4),
                    'withinCE': round(win0.mean().item(), 4)}, 'ablations': {}}
print(res['baseline'], flush=True)

def record(name, ce, cat, win):
    dce, dcat, dwin = ce - ce0, cat - cat0, win - win0
    m_ce, _, se_ce = paired_stats(dce, NR)
    m_ca, _, se_ca = paired_stats(dcat, NR)
    m_wi, _, se_wi = paired_stats(dwin, NR)
    r = {'dCE': round(m_ce, 4), 'dCE_se_row': round(se_ce, 5),
         'd_categoryCE': round(m_ca, 4), 'd_categoryCE_se_row': round(se_ca, 5),
         'd_withinCE': round(m_wi, 4), 'd_withinCE_se_row': round(se_wi, 5),
         'ratio_cat_over_within': round(m_ca / max(m_wi, 1e-6), 3)}
    res['ablations'][name] = r
    print(f"{name}: dCE {m_ce:+.4f}±{se_ce:.4f} | d_catCE {m_ca:+.4f}±{se_ca:.4f} | "
          f"d_withinCE {m_wi:+.4f}±{se_wi:.4f} | ratio {r['ratio_cat_over_within']}", flush=True)

record('probe_subspace_raw6', *eval_proj(U_raw))
record('probe_subspace_cent5', *eval_proj(U_cent))
for i, U in enumerate(rand_subs):
    record(f'random6_seed{i}', *eval_proj(U))

rs = [res['ablations'][f'random6_seed{i}'] for i in range(5)]
res['random6_mean'] = {k: round(sum(r[k] for r in rs)/5, 4)
                       for k in ['dCE', 'd_categoryCE', 'd_withinCE', 'ratio_cat_over_within']}
json.dump(res, open(f'{OUT}/s3_subspace.json', 'w'), indent=2)
print("S3 DONE", flush=True)
