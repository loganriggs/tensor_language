# ARE THE CENSUS'S ACTUAL PROBE BANDS HEAD-CONCENTRATED? §2091's open question, with the bar MEASURED.
#
# BENCHMARK_BACKLOG rung 8. §2091 established that the rung is mis-grained -- 208 of the 234
# attention-probed census leaves are probed by PCA BANDS of an attention COMPONENT, while the proposed
# mechanism language (motifs) is head-level -- and then failed its own control by setting the bar by eye.
# It registered "random-direction top-2 head share < 0.35" reasoning from the 0.2222 uniform null; the
# real random baseline is 0.5414 on average and ranges 0.387 to 0.761 ACROSS COMPONENTS. That failure
# gated out its two substantive predicates, and §2091 recorded it as LESSON 111 repeated by its author.
#
# This is the run §2091 said was the honest next step, done the way LESSON 111 actually prescribes:
#   * the null is MEASURED, not assumed -- §2091's per-component random-direction baselines are the
#     comparison, and they differ by a factor of two between components, which is exactly why a single
#     global bar was the wrong instrument;
#   * the comparison is PAIRED -- each leaf's band is scored against ITS OWN component's baseline;
#   * and it measures the census's REAL probe bands via census_lib.pca_block(key, stag, span), not each
#     component's own leading directions, which §2091 flagged as "not a number about what the census
#     probes".
#
# COST NOTE: §2091 captured the full grid per component and took 1833s. This subsamples rows 4x, which
# pred_c exists to police.
#
# REGISTERED PREDICTIONS:
#   (a) THE BANDS ARE MORE HEAD-CONCENTRATED THAN CHANCE AT THEIR OWN COMPONENT: the median over leaves of
#       (band top-2 head share - that component's measured random baseline) is >= +0.10. If FALSE, the
#       directions the census actually probes are no more head-concentrated than arbitrary directions at
#       the same component, and a head-grain motif language cannot address what the census probes --
#       rung 8 would then need a component-level mechanism language, which is a different rung.
#   (b) AND IT IS THE COMMON CASE, not a tail: the band share exceeds its component's baseline for >= 60%
#       of the component-probed leaves.
#   (c) INSTRUMENT CONTROL, and (a)/(b) may not be read without it: the per-component random baselines
#       recomputed here, on 4x-subsampled rows, reproduce §2091's full-grid values within 0.02. §2091's
#       numbers are the bar this run is scored against, so if subsampling moves the baseline the
#       comparison is meaningless.
#
# Writes band_head_concentration_results.json. DISCOVERY ONLY.
import json
import os
import re
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'head_grain_expressibility_results.json']
    miss = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    b = json.load(open(os.path.join(BQ, 'head_grain_expressibility_results.json')))
    print(f"DRYRUN OK: S2091 baselines present (mean random {b['mean_random_top2']}, "
          f"range over components); pairing each leaf's real probe band against its "
          f"own component's baseline")
    raise SystemExit(0)

import torch                                                              # noqa: E402

import census_lib as C                                                    # noqa: E402

NH = 9
HD = 1152 // NH
STRIDE = 4
BASE = json.load(open('head_grain_expressibility_results.json'))['per_component']

C.use_state('census_state_diverse.pt')
R = C.rows()
t0 = time.time()

LEAVES = []
for lf in C.state()['leaves']:
    tp = lf.get('top_probes') or []
    if not tp:
        continue
    p = tp[0] if isinstance(tp[0], str) else str(tp[0])
    mt = re.match(r"\('pca', '(a\d+)', '([^']+)', \((\d+), (\d+)\)\)", p)
    if mt:
        LEAVES.append((lf['tag'], mt.group(1), mt.group(2),
                       (int(mt.group(3)), int(mt.group(4)))))
COMPS = sorted({c for _t, c, _s, _b in LEAVES}, key=lambda k: int(k[1:]))
print(f'{len(LEAVES)} component-probed leaves over {len(COMPS)} components', flush=True)


@torch.no_grad()
def per_head(key):
    cap = []
    h = C.MODS[key].c_proj.register_forward_pre_hook(
        lambda mo, args: cap.append(args[0].detach().float().reshape(-1, C.D).cpu()))
    for i in range(0, R.shape[0], 8 * STRIDE):
        bb = R[i:i + 8, :257].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
    h.remove()
    X = torch.cat(cap)
    W = C.MODS[key].c_proj.weight.detach().float().cpu()
    return [X[:, hh * HD:(hh + 1) * HD] @ W[:, hh * HD:(hh + 1) * HD].T for hh in range(NH)]


def top2_along(per, dirs):
    out = []
    for d in dirs:
        v = torch.stack([(p @ d).var() for p in per])
        v = v / v.sum().clamp_min(1e-12)
        out.append(float(v.sort(descending=True).values[:2].sum()))
    return sum(out) / len(out)


ROWS = []
REPRO = {}
for key in COMPS:
    per = per_head(key)
    g = torch.Generator().manual_seed(20260830 + int(key[1:]))
    rnd = torch.randn(64, C.D, generator=g)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)
    REPRO[key] = round(top2_along(per, rnd), 4)
    for tag, k, stag, blk in [x for x in LEAVES if x[1] == key]:
        try:
            V = C.pca_block(k, stag, blk).float().cpu()
        except Exception as exc:
            print(f'    {tag}: band unavailable ({type(exc).__name__})', flush=True)
            continue
        ROWS.append({'tag': tag, 'comp': key, 'span': list(blk),
                     'band_top2': round(top2_along(per, V), 4),
                     'baseline_S2091': BASE[key]['top2_share_random'],
                     'baseline_here': REPRO[key]})
    print(f'  {key}: baseline S2091 {BASE[key]["top2_share_random"]:.4f} | here '
          f'{REPRO[key]:.4f} | leaves {sum(1 for r in ROWS if r["comp"]==key)} '
          f'({time.time()-t0:.0f}s)', flush=True)

dev = max(abs(REPRO[k] - BASE[k]['top2_share_random']) for k in COMPS)
delt = sorted(r['band_top2'] - r['baseline_S2091'] for r in ROWS)
med = delt[len(delt) // 2] if delt else float('nan')
above = sum(1 for r in ROWS if r['band_top2'] > r['baseline_S2091'])
frac = above / max(len(ROWS), 1)
pc = dev <= 0.02
pa = med >= 0.10
pb = frac >= 0.60
out = {'n_leaves': len(ROWS), 'n_components': len(COMPS), 'stride': STRIDE,
       'median_delta_vs_own_baseline': round(med, 4),
       'frac_above_own_baseline': round(frac, 4),
       'max_baseline_deviation_from_S2091': round(dev, 4),
       'baselines_here': REPRO, 'leaves': ROWS,
       'pred_a_median_delta_10pt': bool(pa),
       'pred_b_common_case': bool(pb),
       'pred_c_baseline_reproduces': bool(pc)}
json.dump(out, open('band_head_concentration_results.json', 'w'), indent=1)
print(f'\n(c) baselines reproduce S2091 within 0.02 (max dev {dev:.4f}): '
      f"{'HELD' if pc else 'FAILED'}")
if not pc:
    print('    CONTROL FAILED -- 4x row subsampling moved the baseline, so the '
          'paired comparison against S2091 is void.')
else:
    print(f"(a) median band-minus-baseline {med:+.4f} >= +0.10: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) above own baseline for {frac:.1%} of leaves (bar 60%): "
          f"{'HELD' if pb else 'FAILED'}")
    if not pa:
        print('    READING: the census\'s real probe bands are no more '
              'head-concentrated than arbitrary directions at the same '
              'component -- rung 8 needs a COMPONENT-level mechanism language.')
print(f'wrote band_head_concentration_results.json ({time.time()-t0:.0f}s)')
