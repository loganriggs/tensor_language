# candidate_confirm_65_78: THE TWO-SKIP RULE (§1389 reviewer) applied to the wave-2
# candidates — heads 6.5 (structured-format delimiters?) and 7.8 (payload-recurrer,
# category unknown). Scope each on TWO disjoint fresh skips; mechanical category proxies.
#
# Registered predictions:
#   pred_a 6.5's top-15 scope targets are >= 50% delimiter-type tokens (containing one of
#          | , : ; \n - tab) in BOTH skips.
#   pred_b 7.8's two skips share >= 5 target token TYPES between their top-15s (some
#          consistent category exists).
#   pred_c per-head mean_dce within +-50% across the two skips (magnitude stability).
import json, sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import swarm_lib as sw

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'candidate_confirm_65_78_results.json'
DELIMS = set('|,:;-\n\t')


def is_delim(t):
    ts = t.strip()
    return (ts != '' and all(c in DELIMS for c in ts)) or t == '\n' or '\n' in t


def run(spec, skips=(1300, 1500)):
    outs = []
    for sk in skips:
        sc = sw.specificity_and_scope(spec, nrows=96, topk=15, skip=sk)
        outs.append(sc)
        print(f"{spec} skip {sk}: mean {sc['mean_dce']} | "
              f"targets {[e['target'] for e in sc['top_examples'][:8]]}", flush=True)
    return outs


res = {}
a65 = run(('head_ov', 6, 5))
a78 = run(('head_ov', 7, 8))
res['6.5'] = a65
res['7.8'] = a78

frac_delim = []
for sc in a65:
    tg = [e['target'] for e in sc['top_examples']]
    frac_delim.append(sum(1 for t in tg if is_delim(t)) / max(len(tg), 1))
t1 = set(e['target'] for e in a78[0]['top_examples'])
t2 = set(e['target'] for e in a78[1]['top_examples'])
shared = len(t1 & t2)
m65 = [sc['mean_dce'] for sc in a65]
m78 = [sc['mean_dce'] for sc in a78]
def stable(ms):
    lo, hi = min(ms), max(ms)
    return hi <= 1.5 * max(abs(lo), 1e-6) if lo > 0 else abs(hi - lo) <= 0.5 * max(abs(hi), abs(lo), 1e-6)
pa = all(f >= 0.5 for f in frac_delim)
pb = shared >= 5
pc = stable([abs(x) for x in m65]) and stable([abs(x) for x in m78])
out = {'6.5_frac_delim': [round(f, 3) for f in frac_delim],
       '7.8_shared_types': shared, '6.5_means': m65, '7.8_means': m78,
       'scopes': res,
       'pred_a_65_delims_both': bool(pa), 'pred_b_78_consistent': bool(pb),
       'pred_c_magnitudes_stable': bool(pc)}
json.dump(out, open(OUT, 'w'), indent=1)
print(f"6.5 delim fracs {frac_delim} | 7.8 shared {shared} | "
      f"pred_a {pa} pred_b {pb} pred_c {pc}")
print(f"wrote {OUT}")
