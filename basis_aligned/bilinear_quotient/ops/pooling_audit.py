# BQGATE: LIBRARY  -- an audit over artifacts, not an experiment; no GPU, no model.
#
# §1973's open question: two claims have been overturned by changing the INSTRUMENT rather than the data,
# and the remaining 2-of-3 conclusions have never been re-read under pooling. Every result JSON with a
# `paired` block stores per-role mean, se and n, and a paired difference pools exactly from those:
#
#   mean_pool = sum(n_i * mean_i) / N
#   sd_i      = se_i * sqrt(n_i)
#   var_pool  = [ sum((n_i - 1) * sd_i^2) + sum(n_i * (mean_i - mean_pool)^2) ] / (N - 1)
#   t_pool    = mean_pool / sqrt(var_pool / N)
#
# So the whole arc can be re-read from disk. This reports every comparison where the per-role VOTE and
# the POOLED evidence disagree, and flags those where skip1200 -- the half-sized role (§1971) -- was a
# supporting rather than a dissenting vote, which §1973 named as the ones at risk.
import glob
import json
import math
import os

PT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
HALF = 'skip1200'


def pool(stats):
    """stats: {role: {'mean':.., 'se':.., 'n':..}} -> pooled mean, t, N"""
    rows = [(v['mean'], v['se'], v['n']) for v in stats.values() if v.get('n')]
    if len(rows) < 2:
        return None
    N = sum(n for _m, _s, n in rows)
    mp = sum(m * n for m, _s, n in rows) / N
    ss = sum((n - 1) * (s * math.sqrt(n)) ** 2 + n * (m - mp) ** 2 for m, s, n in rows)
    var = ss / (N - 1)
    se = math.sqrt(var / N) if var > 0 else 0.0
    return mp, (mp / se if se else math.inf * (1 if mp > 0 else -1)), N


def main():
    rows = []
    for p in sorted(glob.glob(PT + 'ops/*_results.json')):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        pr = d.get('paired')
        if not isinstance(pr, dict):
            continue
        # two shapes in the tree: {cov: {role: {pair: stats}}} and {role: {pair: stats}}
        covs = pr if all(isinstance(v, dict) and any(k.startswith('skip') for k in v) for v in pr.values()) \
            else {'-': pr}
        for cov, byrole in covs.items():
            if not all(isinstance(v, dict) for v in byrole.values()):
                continue
            pairs = set()
            for role, byp in byrole.items():
                if isinstance(byp, dict):
                    pairs |= {k for k, v in byp.items() if isinstance(v, dict) and 'se' in v}
            for pair in sorted(pairs):
                stats = {r: byrole[r][pair] for r in byrole
                         if isinstance(byrole.get(r), dict) and pair in byrole[r]}
                if len(stats) < 3:
                    continue
                got = pool(stats)
                if not got:
                    continue
                mp, tp, N = got
                sign_votes = sum(1 for v in stats.values() if v['mean'] < 0)
                pooled_neg = mp < 0
                disagree = (sign_votes >= 2) != pooled_neg
                half_supports = (HALF in stats
                                 and (stats[HALF]['mean'] < 0) == (sign_votes >= 2))
                rows.append((os.path.basename(p)[:-13], cov, pair, sign_votes, mp * 1000, tp, N,
                             disagree, half_supports))

    print(f'  comparisons with three roles of paired stats: {len(rows)}')
    dis = [r for r in rows if r[7]]
    print(f'  where the 2-of-3 SIGN VOTE and the pooled sign DISAGREE: {len(dis)}')
    for r in dis:
        print(f'    {r[0]:34s} {r[1]:7s} {r[2]:26s} votes {r[3]}/3  pooled {r[4]:+.3f}m t {r[5]:+.2f}')
    risk = [r for r in rows if r[8] and abs(r[5]) < 2.0]
    print(f'\n  AT RISK -- {HALF} supported the majority AND the pooled |t| < 2: {len(risk)}')
    for r in risk[:12]:
        print(f'    {r[0]:34s} {r[1]:7s} {r[2]:26s} votes {r[3]}/3  pooled {r[4]:+.3f}m t {r[5]:+.2f}')
    strong = [r for r in rows if abs(r[5]) >= 10]
    print(f'\n  unambiguous (pooled |t| >= 10): {len(strong)} of {len(rows)}')


main()
