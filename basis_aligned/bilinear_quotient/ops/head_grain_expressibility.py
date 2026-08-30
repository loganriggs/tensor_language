# CAN A COMPONENT-GRAIN PROBE BE EXPRESSED IN HEAD-GRAIN MOTIF LANGUAGE?
#
# BENCHMARK_BACKLOG rung 8, design prerequisite. §332 sent this rung to the backlog with a design note:
# "the deeper census is attention-probed, so its mechanism rung requires pattern-side constructions
# (motif conditions composed with value reads), not deeper MLP folds ... the ladder's next rung is
# architectural, and goes to the backlog with a design note rather than an overnight run."
#
# TWO THINGS MEASURED BEFORE DESIGNING (LESSON B), both from census_state_diverse.pt, no GPU:
#   1. The rung's framing is CONFIRMED, not refuted: 234 of 311 census leaves (75.2%) are attention-probed
#      by their top probe, and 265 (85.2%) have an attention probe in their top three. "The ladder rung for
#      the census majority" is accurate.
#   2. But the breakdown exposes a granularity mismatch the design note does not mention. Of those 234,
#      only 24 are HEAD-probed; 208 are PCA bands of an attention COMPONENT's output. The proposed
#      mechanism language -- motifs -- is a HEAD-level object (prev/self/induction heads, 162 of them).
#      **The rung proposes a head-grain language for a population that is component-grain probed.**
#
# That mismatch is the rung's real precondition, and it is testable without building any mechanism: if a
# component's leading output directions are carried by one or two heads, head-grain motif conditions can
# express component-grain probes; if the variance is spread across all nine heads, they cannot, and the
# rung needs either head-level re-probing of the census or a component-level mechanism language.
#
# METHOD. c_proj's input is the concatenated per-head outputs (9 heads x 128), so head h's contribution to
# the component output is exactly W[:, h*128:(h+1)*128] @ x[h*128:(h+1)*128] -- separable with no
# attribution heuristic. For each attention component: PCA the output over the census grid, then for each
# leading direction measure each head's share of the variance along it.
#
# REGISTERED PREDICTIONS:
#   (a) HEAD-GRAIN IS EXPRESSIVE: averaged over the top-8 output PCA directions, the top-2 heads carry
#       >= 50% of the variance, for the median attention component. The uniform null is 2/9 = 22.2%, so
#       this asks for better than double chance. If FALSE the component's leading directions are spread
#       across heads and a motif-composition language cannot address what the census actually probes.
#   (b) IT IS NOT ONE ODD LAYER: (a)'s condition holds for >= 12 of the 18 attention components.
#   (c) CONTROL, and (a) may not be read without it: the same top-2 share measured along RANDOM unit
#       directions is < 35%. PCA directions are chosen to concentrate variance and could look
#       head-concentrated for that reason alone; the random-direction share is what head concentration
#       looks like when nothing selected for it.
#
# Writes head_grain_expressibility_results.json. DISCOVERY ONLY; designs nothing and builds no mechanism.
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

if os.environ.get('BQLIB_DRYRUN') == '1':
    if not os.path.exists(os.path.join(BQ, 'census_state_diverse.pt')):
        print('DRYRUN FAIL: census state absent'); raise SystemExit(1)
    print('DRYRUN OK: census state present; 18 attention components x 9 heads, '
          'top-8 PCA directions, random-direction control')
    raise SystemExit(0)

import torch                                                              # noqa: E402

import census_lib as C                                                    # noqa: E402

NH = 9
HD = 1152 // NH
TOPK = 8
NRAND = 64

C.use_state('census_state_diverse.pt')
R = C.rows()
t0 = time.time()


@torch.no_grad()
def head_contribs(li):
    """per-head contributions to attention component li's output, over the grid."""
    cap = []
    h = C.MODS[f'a{li}'].c_proj.register_forward_pre_hook(
        lambda mo, args: cap.append(args[0].detach().float().reshape(-1, C.D).cpu()))
    for i in range(0, R.shape[0], 8):
        bb = R[i:i + 8, :257].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
    h.remove()
    X = torch.cat(cap)
    W = C.MODS[f'a{li}'].c_proj.weight.detach().float().cpu()
    return X, W


def shares(li):
    X, W = head_contribs(li)
    per = []
    for hh in range(NH):
        sl = slice(hh * HD, (hh + 1) * HD)
        per.append(X[:, sl] @ W[:, sl].T)          # head hh's contribution, (N, D)
    tot = sum(per)
    tot = tot - tot.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(tot, full_matrices=False)
    pcs = Vh[:TOPK]
    g = torch.Generator().manual_seed(20260830 + li)
    rnd = torch.randn(NRAND, C.D, generator=g)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)

    def top2(dirs):
        out = []
        for d in dirs:
            v = torch.stack([(p @ d).var() for p in per])
            v = v / v.sum().clamp_min(1e-12)
            out.append(float(v.sort(descending=True).values[:2].sum()))
        return sum(out) / len(out)
    return round(top2(pcs), 4), round(top2(rnd), 4)


RES = {}
for li in range(18):
    p, r = shares(li)
    RES[f'a{li}'] = {'top2_share_pca': p, 'top2_share_random': r}
    print(f'  a{li:<2d} top-2 head share: PCA {p:.4f} | random {r:.4f}  '
          f'({time.time()-t0:.0f}s)', flush=True)

pv = sorted(v['top2_share_pca'] for v in RES.values())
med = pv[len(pv) // 2]
nok = sum(1 for v in RES.values() if v['top2_share_pca'] >= 0.50)
rmax = max(v['top2_share_random'] for v in RES.values())
rmean = sum(v['top2_share_random'] for v in RES.values()) / len(RES)
pa = med >= 0.50
pb = nok >= 12
pc = rmean < 0.35
out = {'n_heads': NH, 'top_k_directions': TOPK, 'uniform_null_top2': round(2 / NH, 4),
       'per_component': RES, 'median_top2_share_pca': med,
       'components_at_50pct': nok, 'mean_random_top2': round(rmean, 4),
       'max_random_top2': round(rmax, 4),
       'census_attention_probed_top': '234/311 = 75.2%',
       'census_head_probed_top': '24/311 = 7.7%',
       'pred_a_median_top2_at_50': bool(pa),
       'pred_b_twelve_of_eighteen': bool(pb),
       'pred_c_random_control_low': bool(pc)}
json.dump(out, open('head_grain_expressibility_results.json', 'w'), indent=1)
print(f'\nuniform null (2 of 9 heads) = {2/NH:.4f}')
print(f'(c) CONTROL mean random-direction top-2 share {rmean:.4f} < 0.35: '
      f"{'HELD' if pc else 'FAILED'}")
if not pc:
    print('    CONTROL FAILED -- top-2 concentration is what ANY direction shows '
          'here, so (a) and (b) say nothing about PCA directions specifically.')
else:
    print(f"(a) median top-2 PCA share {med:.4f} >= 0.50: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=12 of 18 components at 0.50: {nok} : {'HELD' if pb else 'FAILED'}")
    if not pa:
        print('    READING: the census majority is component-probed and its '
              'leading directions are NOT head-concentrated, so rung 8 needs a '
              'component-level mechanism language or head-level re-probing.')
print(f'wrote head_grain_expressibility_results.json ({time.time()-t0:.0f}s)')
