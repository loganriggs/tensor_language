# CLASS-BOTTLENECK SPECTRUM AT THE TAIL (rung 53; math review 2026-08-30 22:10).
#
# CONVENTION (S2135): damage numbers elsewhere are CE above the real model, lower is better. This rung is a
# CPU-light capture: does the S2145 per-layer dictionary cost track the WITHIN-CLASS RESIDUAL ENERGY of the real
# attention outputs? The aXL dictionary forces each tail attention through a rank-<=10 class-conditional map
# (means for the 6 CONSTN classes, linear maps for the 4 LINK classes); the mean-only residual fraction
# e_li = E||y - mu_class||^2 / E||y||^2 is the natural energy proxy for what the dictionary cannot carry.
# Real-model forward on the FW eval rows (R0:R1, all 256 positions), oracle classes.
#
# S2145 marginals (damage added by each replacement): a10L..a17L =
#   [0.0180, 0.0180, 0.0057, 0.0302, 0.0729, 0.0350, 0.1572, 0.0158]
#
# REGISTERED PREDICTIONS:
#   (a) ENERGY TRACKS PRICE: Spearman rho(e_li, marginal_li) >= 0.7 over the 8 layers.
#   (b) THE MAX IS a16: argmax_li e_li = 16.
#   (c) THE TWO SMALLEST ARE {a12, a17}.
# NULL: damage is CE-weighted, not energy-weighted (S2117's rho 0.81 was good but imperfect); rho < 0.7 with (b)
# still held would partially save the frame.
#
# Writes tail_class_spectrum_results.json. Self-reviewed.
import json, os, sys, time
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')

if os.environ.get('BQLIB_DRYRUN') == '1':
    _bq = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    if not os.path.exists(_bq + 'frontier_tail_prefix_results.json'):
        print('DRYRUN FAIL: S2145 artifact absent'); raise SystemExit(1)
    _p = json.load(open(_bq + 'frontier_tail_prefix_results.json'))
    print(f"DRYRUN OK: S2145 marginals {_p['marginals_a10L_to_a17L']}; one real-model capture pass")
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402
from bilin18_joint_removal import m, FW, DEV                              # noqa: E402
from circuit_dictionary import classify                                   # noqa: E402

T0 = time.time()
D = 1152; R0, R1 = 120, 300
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
MARG = [0.0180, 0.0180, 0.0057, 0.0302, 0.0729, 0.0350, 0.1572, 0.0158]
cls = classify(R0, R1).to(DEV).reshape(-1)
caps = {li: [] for li in range(10, 18)}
hs = []
for li in range(10, 18):
    def mk(li=li):
        def h(mo, i_, o_):
            caps[li].append(o_[0].detach().reshape(-1, D).float())
        return h
    hs.append(m.transformer.h[li].attn.register_forward_hook(mk()))
with torch.no_grad():
    for i in range(R0, R1, 4):
        idx = FW[i:i + 4, :257].to(DEV)[:, :-1]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
for h in hs: h.remove()
e = {}
for li in range(10, 18):
    Y = torch.cat(caps[li]); caps[li] = None
    tot = float((Y ** 2).sum()); res = 0.0
    for k in range(10):
        sel = cls == k
        if int(sel.sum()) == 0: continue
        mu = Y[sel].mean(0)
        res += float(((Y[sel] - mu) ** 2).sum())
    e[li] = res / max(tot, 1e-12)
ev = [e[li] for li in range(10, 18)]
def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
    for pos, i in enumerate(order): r[i] = pos
    return r
re_, rm = ranks(ev), ranks(MARG)
n = 8
rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(re_, rm)) / (n * (n * n - 1))
pa = rho >= 0.7
pb = max(range(10, 18), key=lambda li: e[li]) == 16
small2 = set(sorted(range(10, 18), key=lambda li: e[li])[:2])
pc = small2 == {12, 17}
out = {'within_class_residual_fraction': {f'a{li}': round(e[li], 4) for li in range(10, 18)},
       's2145_marginals': MARG, 'spearman_rho': round(rho, 3),
       'pred_a_energy_tracks_price': bool(pa), 'pred_b_max_is_a16': bool(pb),
       'pred_c_smallest_are_a12_a17': bool(pc), 'self_reviewed': True,
       'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open(PT + 'tail_class_spectrum_results.json', 'w'), indent=1)
print('within-class residual fraction: ' + ' '.join(f'a{li}:{e[li]:.3f}' for li in range(10, 18)))
print(f"(a) Spearman rho {rho:+.3f} >= 0.7: {'HELD' if pa else 'FAILED'}")
print(f"(b) argmax = a16: {'HELD' if pb else 'FAILED'}")
print(f"(c) smallest two = a12,a17 (got {sorted('a%d' % x for x in small2)}): {'HELD' if pc else 'FAILED'}")
print(f'wrote tail_class_spectrum_results.json ({time.time() - T0:.0f}s)')
