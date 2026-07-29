"""Step 2: activation-patching importance over all 180 components (162 heads + 18 MLPs).

Patch the CLEAN activation of one component into the CORRUPTED run (all positions),
metric = recovered fraction of the correct-digit logit margin at the final position:
    rf = (M_patch - M_corr) / (M_clean - M_corr),  M = logit[clean_ans] - logit[corr_ans].
Averaged over the 30 analysis pairs. Then: top-10, cumulative top-k, positional
analysis for the top heads (and top MLPs), and comparison with the qk_circuit_atlas
'digit' importance ranking.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/increment')
from common import get_model, forward, retry_oom, POSITIONS, OUT, QK

torch.manual_seed(0)
m, cfg = get_model()
NH, NL = cfg['n_head'], cfg['n_layer']
S = torch.load(f'{OUT}/stimuli.pt')
NA = 30  # analysis pairs
clean, corr = S['clean'][:NA].cuda(), S['corr'][:NA].cuda()
ca, xa = S['clean_ans'][:NA].cuda(), S['corr_ans'][:NA].cuda()
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]
BS = 8


def chunks():
    for i in range(0, NA, BS):
        yield i, clean[i:i + BS], corr[i:i + BS], ca[i:i + BS], xa[i:i + BS]


# --- cache clean activations per chunk, plus baseline margins -------------
clean_caches = {}
M_clean = torch.zeros(NA); M_corr = torch.zeros(NA)
with torch.no_grad():
    for i, bc, bx, bca, bxa in chunks():
        cch = {}
        lgc = retry_oom(forward, m, bc, cache=cch)[:, -1].float()
        # keep only component entries (drop resid) to save memory
        clean_caches[i] = {k: v for k, v in cch.items() if k[0] in ('h', 'm')}
        lgx = retry_oom(forward, m, bx)[:, -1].float()
        n = torch.arange(len(bca), device='cuda')
        M_clean[i:i + len(bca)] = (lgc[n, bca] - lgc[n, bxa]).cpu()
        M_corr[i:i + len(bca)] = (lgx[n, bca] - lgx[n, bxa]).cpu()
GAP = (M_clean - M_corr)
print(f"baseline: M_clean {M_clean.mean():.3f}  M_corr {M_corr.mean():.3f}", flush=True)


def margins_with_patch(patch_keys, positions=None):
    """Patch clean acts of patch_keys (at `positions`, None=all) into corrupted run.
    Returns per-pair patched margin [NA]."""
    Mp = torch.zeros(NA)
    with torch.no_grad():
        for i, bc, bx, bca, bxa in chunks():
            cch = clean_caches[i]
            patch = {}
            for key in patch_keys:
                if key[0] == 'h':
                    _, li, h = key
                    patch[key] = (cch[('h', li)][:, :, h, :], positions)
                else:
                    patch[key] = (cch[key], positions)
            lg = retry_oom(forward, m, bx, patch=patch)[:, -1].float()
            n = torch.arange(len(bca), device='cuda')
            Mp[i:i + len(bca)] = (lg[n, bca] - lg[n, bxa]).cpu()
    return Mp


def rf(Mp):
    return ((Mp - M_corr) / GAP).mean().item()


# --- single-component sweep ------------------------------------------------
imp = {}
for j, c in enumerate(ALL):
    imp[c] = rf(margins_with_patch([c]))
    if (j + 1) % 30 == 0:
        print(f"  {j+1}/180 components done", flush=True)

ranked = sorted(ALL, key=lambda c: -imp[c])
print("\nTOP-10 components (recovered margin fraction, patched at all positions):")
for c in ranked[:10]:
    print(f"  {str(c):12s} {imp[c]:+.4f}")

# --- cumulative top-k ------------------------------------------------------
cum = {}
for k in [1, 2, 3, 5, 8, 12, 20, 30]:
    cum[k] = rf(margins_with_patch(ranked[:k]))
print("\ncumulative top-k recovered fraction:", {k: round(v, 4) for k, v in cum.items()})

# --- positional analysis for top heads and top MLPs ------------------------
top_heads = [c for c in ranked if c[0] == 'h'][:5]
top_mlps = [c for c in ranked if c[0] == 'm'][:3]
posres = {}
for c in top_heads + top_mlps:
    posres[str(c)] = {}
    for pname, pos in POSITIONS.items():
        posres[str(c)][pname] = round(rf(margins_with_patch([c], positions=pos)), 4)
    print(f"positional {str(c)}: {posres[str(c)]}", flush=True)

# --- comparison with qk_circuit_atlas 'digit' importance -------------------
atlas = json.load(open(f'{QK}/qk_circuit_atlas.json'))['importance_matrix']['digit']
mine = np.array([imp[c] for c in ALL])
theirs = np.array([atlas[str(c)] for c in ALL])
from scipy.stats import spearmanr
rho_all = spearmanr(mine, theirs).statistic
# also on top-30 union (rank agreement among components that matter to either)
idx_union = np.argsort(-mine)[:30].tolist() + np.argsort(-theirs)[:30].tolist()
idx_union = sorted(set(idx_union))
rho_top = spearmanr(mine[idx_union], theirs[idx_union]).statistic
mytop10 = [str(c) for c in ranked[:10]]
attop10 = [k for k, _ in sorted(atlas.items(), key=lambda x: -x[1])[:10]]
overlap = [c for c in mytop10 if c in attop10]
print(f"\nSpearman(all 180) = {rho_all:.3f};  Spearman(top-30 union) = {rho_top:.3f}")
print(f"my top10:    {mytop10}")
print(f"atlas top10: {attop10}")
print(f"overlap: {overlap}")

res = {
    'baseline': {'M_clean_mean': M_clean.mean().item(), 'M_corr_mean': M_corr.mean().item()},
    'importance': {str(c): round(imp[c], 5) for c in ALL},
    'top10': [(str(c), round(imp[c], 4)) for c in ranked[:10]],
    'cumulative_topk': {str(k): round(v, 4) for k, v in cum.items()},
    'positional': posres,
    'atlas_comparison': {'spearman_all': round(float(rho_all), 4),
                         'spearman_top30_union': round(float(rho_top), 4),
                         'my_top10': mytop10, 'atlas_top10': attop10,
                         'top10_overlap': overlap},
}
json.dump(res, open(f'{OUT}/s2_patching.json', 'w'), indent=2)
print('\nsaved s2_patching.json')
