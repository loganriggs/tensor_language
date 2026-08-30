# IS "MOTIF" THE RIGHT HEAD-GRAIN VOCABULARY? Rung 8's first content test.
#
# S2092 established rung 8's expressibility precondition: the census's real probe bands are more
# head-concentrated than arbitrary directions at the same component, for 208 of 208 component-probed
# leaves (median +0.1590 above each leaf's own baseline). It also stated exactly what it did NOT show:
# "does NOT say motif classes (prev/self/induction) are the right head-grain vocabulary".
#
# That is this run. S332's proposal is "motif conditions composed with value reads", and motifs are a
# specific vocabulary: the 162 heads are classed diffuse / self / prev / ind / first. If the heads the
# census actually loads on are DIFFUSE, the motif vocabulary cannot name them however concentrated the
# bands are, and rung 8 needs a different head-grain language.
#
# THE NULL IS MEASURED FIRST, which is the whole point after S2091 failed exactly here. Among the 99 heads
# of the 11 leaf-carrying components, **61 are named-motif and 38 diffuse -- a base rate of 0.6162**. So
# "most leaf heads are named" is NOT evidence of anything; the bar has to be set above the base rate, and
# an instinct like "fewer than 40% of leaves have both heads diffuse" would have been satisfied by chance
# (0.384^2 = 0.147). Bars below are set from 0.6162, not from intuition.
#
# REGISTERED PREDICTIONS:
#   (a) MOTIF HEADS ARE OVER-REPRESENTED: the named-motif fraction among leaves' top-2 band heads exceeds
#       the component-matched base rate by >= 1.20x. If FALSE, the census loads on motif and diffuse heads
#       in proportion to their availability, the motif vocabulary has no special claim on what the census
#       probes, and rung 8 needs a head-grain language that can name diffuse heads.
#   (b) NULL CONTROL, and (a) may not be read without it: drawing 2 of 9 heads uniformly at random within
#       each leaf's own component reproduces the component-matched base rate within +-0.03. This checks
#       that the base rate is computed over the right population -- the failure mode of S2091 was a null
#       computed for the wrong ensemble, not a bar that was merely too low.
#   (c) IT IS NOT ONE COMPONENT: the named fraction exceeds that component's own base rate in >= 7 of the
#       11 leaf-carrying components. a7 alone holds 56 of 208 leaves, so a single component could carry a
#       pooled result.
#
# Writes motif_vocabulary_results.json. DISCOVERY ONLY.
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
    need = ['census_state_diverse.pt', 'attn_motifs3_results.json']
    miss = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    if not os.path.exists(os.path.join(BQ, 'attn_motifs3_results.json')):
        print('DRYRUN FAIL: motif table absent'); raise SystemExit(1)
    _mt = json.load(open(os.path.join(BQ, 'attn_motifs3_results.json')))['motif_table']
    print(f"DRYRUN OK: motif table has {len(_mt)} heads; classing leaves' top-2 "
          f"band heads against a measured component-matched base rate")
    raise SystemExit(0)

import torch                                                              # noqa: E402

import census_lib as C                                                    # noqa: E402

NH = 9
HD = 1152 // NH
STRIDE = 4
C.use_state('census_state_diverse.pt')
R = C.rows()
t0 = time.time()

LEAVES = []
for lf in C.state()['leaves']:
    tp = lf.get('top_probes') or []
    if not tp:
        continue
    pstr = tp[0] if isinstance(tp[0], str) else str(tp[0])
    mt = re.match(r"\('pca', '(a\d+)', '([^']+)', \((\d+), (\d+)\)\)", pstr)
    if mt:
        LEAVES.append((lf['tag'], mt.group(1), mt.group(2),
                       (int(mt.group(3)), int(mt.group(4)))))
COMPS = sorted({c for _t, c, _s, _b in LEAVES}, key=lambda k: int(k[1:]))
print(f'{len(LEAVES)} component-probed leaves over {len(COMPS)} components', flush=True)

MOTIF = {(e[0], e[1]): e[2] for e in
         json.load(open('attn_motifs3_results.json'))['motif_table']}

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


def top2_heads(per, dirs):
    """the two heads carrying most variance along the band, and their share."""
    acc = torch.zeros(NH)
    for d in dirs:
        v = torch.stack([(p @ d).var() for p in per])
        acc += v / v.sum().clamp_min(1e-12)
    acc = acc / len(dirs)
    o = acc.argsort(descending=True)
    return [int(o[0]), int(o[1])], float(acc[o[:2]].sum())


ROWS = []
for key in COMPS:
    li = int(key[1:])
    per = per_head(key)
    for tag, k, stag, blk in [x for x in LEAVES if x[1] == key]:
        try:
            V = C.pca_block(k, stag, blk).float().cpu()
        except Exception as exc:
            print(f'    {tag}: band unavailable ({type(exc).__name__})', flush=True)
            continue
        hs2, sh = top2_heads(per, V)
        cls = [MOTIF.get((li, h), '?') for h in hs2]
        ROWS.append({'tag': tag, 'comp': key, 'heads': hs2, 'classes': cls,
                     'top2_share': round(sh, 4),
                     'n_named': sum(1 for c_ in cls if c_ not in ('diffuse', '?'))})
    print(f'  {key}: {sum(1 for r in ROWS if r["comp"]==key)} leaves '
          f'({time.time()-t0:.0f}s)', flush=True)

# component-matched base rate, weighted by each component's leaf count
import collections
cnt = collections.Counter(r['comp'] for r in ROWS)
base_by_comp = {}
for key in COMPS:
    li = int(key[1:])
    cl = [MOTIF.get((li, h), '?') for h in range(NH)]
    base_by_comp[key] = sum(1 for c_ in cl if c_ not in ('diffuse', '?')) / NH
tot = sum(cnt.values())
base_matched = sum(base_by_comp[k] * cnt[k] for k in cnt) / max(tot, 1)
named = sum(r['n_named'] for r in ROWS)
obs = named / max(2 * len(ROWS), 1)
ratio = obs / max(base_matched, 1e-9)

g = torch.Generator().manual_seed(20260830)
draws = []
for _ in range(2000):
    tot_n = 0
    for r in ROWS:
        li = int(r['comp'][1:])
        pick = torch.randperm(NH, generator=g)[:2].tolist()
        tot_n += sum(1 for h in pick if MOTIF.get((li, h), '?') not in ('diffuse', '?'))
    draws.append(tot_n / (2 * len(ROWS)))
null_mean = sum(draws) / len(draws)

per_comp = {}
for key in COMPS:
    sub = [r for r in ROWS if r['comp'] == key]
    if not sub:
        continue
    o = sum(r['n_named'] for r in sub) / (2 * len(sub))
    per_comp[key] = {'observed': round(o, 4), 'base': round(base_by_comp[key], 4),
                     'n_leaves': len(sub), 'above': bool(o > base_by_comp[key])}
nabove = sum(1 for v in per_comp.values() if v['above'])

pb = abs(null_mean - base_matched) <= 0.03
pa = ratio >= 1.20
pc = nabove >= 7
out = {'n_leaves': len(ROWS), 'observed_named_fraction': round(obs, 4),
       'component_matched_base_rate': round(base_matched, 4),
       'permutation_null_mean': round(null_mean, 4),
       'over_representation_ratio': round(ratio, 4),
       'components_above_own_base': nabove, 'n_components': len(per_comp),
       'per_component': per_comp, 'leaves': ROWS,
       'pred_a_motifs_over_represented': bool(pa),
       'pred_b_null_matches_base_rate': bool(pb),
       'pred_c_not_one_component': bool(pc)}
json.dump(out, open('motif_vocabulary_results.json', 'w'), indent=1)
print(f"\nobserved named fraction {obs:.4f} | base rate {base_matched:.4f} | "
      f"permutation null {null_mean:.4f} | ratio {ratio:.4f}")
print(f"(b) NULL CONTROL |null - base| {abs(null_mean-base_matched):.4f} <= 0.03: "
      f"{'HELD' if pb else 'FAILED'}")
if not pb:
    print('    CONTROL FAILED -- the base rate is computed over the wrong '
          'population; (a) and (c) may not be read.')
else:
    print(f"(a) over-representation {ratio:.4f} >= 1.20: {'HELD' if pa else 'FAILED'}")
    print(f"(c) above own base in {nabove}/{len(per_comp)} components (bar 7): "
          f"{'HELD' if pc else 'FAILED'}")
    if not pa:
        print("    READING: the census loads on motif and diffuse heads in "
              "proportion to availability -- the MOTIF vocabulary has no "
              "special claim on what the census probes.")
print(f'wrote motif_vocabulary_results.json ({time.time()-t0:.0f}s)')
