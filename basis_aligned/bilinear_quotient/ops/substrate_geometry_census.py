# THE SUBSTRATE GEOMETRY OF EVERY CIRCUIT-DENSE COMPONENT, ON ONE INSTRUMENT.
#
# RUNG 3: §2064's open question, and the fix for a caveat I wrote into §2064 myself -- "three components
# is enough to refute a universal claim and not enough to establish a typical one."
#
# §2056/§2058 (a8), §2062 (a16) and §2064 (m16) each ran the same test at one component and found three
# different arrangements: a8 has a shared substrate WITH separable circuits underneath; m16 has a shared
# substrate with NOTHING separable underneath; a16 has no shared substrate at all. Those were three
# separate runs written at three different times, and the population question was left open.
#
# This runs the identical instrument over EVERY component that wins at least 4 circuits in §2059's census
# -- a8 (16), a16 (13), m16 (6), a3 (5), m14 (5), m13 (4), covering 49 of the 62 curated circuits -- and
# classifies each into the three arrangements. a8, a16 and m16 are re-measured rather than quoted, so
# every number in the table comes from one run of one instrument.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  The shared substrate is the MAJORITY arrangement: at least 4 of the 6 components have a shared
#           direction explaining >= 0.80 of their circuits' directional variance. §2064 found 2 of 3 and
#           explicitly declined to call that a population rate; this is that claim, registered and
#           testable. If FALSE, a16's already-separate geometry is at least as common and §2064's
#           "shared substrate generalises" is itself too broad.
#   pred_b  §2058's TWO-LEVEL structure stays UNIQUE TO a8: exactly one component shows both a shared
#           substrate (>= 0.80) AND a genuine reversal on removing it -- residual |cos| < 0.50 together
#           with strictly more circuits selective at a >= 10% margin in the residual than in the full
#           directions. §2064 found the reversal fails at both other components tested. If FALSE and a
#           second component reverses, §2058 is not a single-component result and §2064's surviving
#           narrow claim needs revision in turn.
#   pred_c  INSTRUMENT CONSISTENCY, not a fact about the model: shared-variance-explained and mean
#           pairwise |cos| are two summaries of the same geometry, so across the 6 components they must
#           correlate at |Pearson r| >= 0.90. If FALSE, the two summaries are measuring different things
#           and every three-way classification above -- including §2062's and §2064's -- rests on a
#           distinction the instrument does not actually make.
#
# Writes circuits/SUBSTRATE_CENSUS.json. DISCOVERY ONLY. No circuit file is modified.
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

# PLAN PRE-FLIGHT (LESSON 109): census_lib builds MODS from the live model at import, so enqueue's
# BQLIB_DRYRUN gate must be answered BEFORE that import or the gate runs the experiment for real.
MIN_CIRCUITS = 4                                     # components winning at least this many circuits

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/BATTERY.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    b = json.load(open(os.path.join(BQ, 'circuits/BATTERY.json')))
    import collections
    c = collections.Counter(v['best_mean'] for v in b['by_tag'].values())
    comps = sorted([k for k, n in c.items() if n >= MIN_CIRCUITS], key=lambda k: -c[k])
    if len(comps) < 3:
        print(f'DRYRUN FAIL: only {len(comps)} components win >= {MIN_CIRCUITS} circuits')
        raise SystemExit(1)
    print(f'DRYRUN OK: {len(comps)} components win >= {MIN_CIRCUITS} circuits: '
          f'{[(k, c[k]) for k in comps]}, covering {sum(c[k] for k in comps)} of '
          f'{len(b["by_tag"])} circuits')
    raise SystemExit(0)

import collections                                                        # noqa: E402

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import census_lib as C                                                    # noqa: E402

BAT = json.load(open('circuits/BATTERY.json'))
CNT = collections.Counter(v['best_mean'] for v in BAT['by_tag'].values())
COMPS = sorted([k for k, n in CNT.items() if n >= MIN_CIRCUITS], key=lambda k: -CNT[k])

C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()
print(f'grid {nflat}; components with >= {MIN_CIRCUITS} circuits: '
      f'{[(k, CNT[k]) for k in COMPS]}', flush=True)


def masks_for(tags):
    out = {}
    for t in tags:
        lf = C.leaf(t)
        mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
        sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
        if mm.sum() == 0 or (~sl).sum() == 0:
            continue                                 # a root node has no off-slice; §2057's NaN trap
        out[t] = (mm, sl)
    return out


@torch.no_grad()
def capture(key):
    R = C.rows(); cap = []
    h = C.MODS[key].register_forward_hook(
        lambda mo, i_, o_: cap.append(((o_[0] if isinstance(o_, tuple) else o_)
                                       .detach().float().reshape(-1, C.D).cpu())))
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
    h.remove()
    return torch.cat(cap)


@torch.no_grad()
def project_out_dce(key, u):
    R = C.rows(); u = u.to(C.DEV); ces = []
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)

        def fh(mo, i_, o_):
            y = o_[0] if isinstance(o_, tuple) else o_
            f = y.float().reshape(-1, C.D)
            f = f - (f @ u).unsqueeze(1) * u.unsqueeze(0)
            f = f.view_as(y).to(y.dtype)
            return (f, o_[1]) if isinstance(o_, tuple) else f

        hh = C.MODS[key].register_forward_hook(fh)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
        x = F.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
        for blkm in C.m.transformer.h:
            x, v1 = blkm(x, v1, x0)
        lg = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
        ces.append(F.cross_entropy(lg.view(-1, lg.size(-1)), tg, reduction='none').cpu())
        hh.remove()
    return torch.cat(ces).float() - base


def n_sel(tab, tags, margin):
    """circuits whose own concentration beats the mean of the others by at least `margin`.

    §2062 reported raw selectivity and had to retrofit a 10% bar when 4 of its 11 'selective' circuits
    turned out to be within 5% of a tie. The margin is explicit here.
    """
    n = 0
    for src in tags:
        oth = [v for k, v in tab[src].items() if k != src and v is not None]
        if tab[src][src] is not None and oth and tab[src][src] > margin * (sum(oth) / len(oth)):
            n += 1
    return n


def study(key):
    tags_all = [t for t, v in BAT['by_tag'].items() if v['best_mean'] == key]
    mk = masks_for(tags_all)
    tags = [t for t in tags_all if t in mk]
    if len(tags) < 3:
        return None
    acts = capture(key)
    dirs = {}
    for t in tags:
        mm, sl = mk[t]
        u = acts[mm].mean(0) - acts[~sl].mean(0)
        if not torch.isfinite(u).all() or u.norm() == 0:
            return None
        dirs[t] = u / u.norm()
    M = torch.stack([dirs[t] for t in tags])
    pairs = [float(abs(dirs[a] @ dirs[b])) for i, a in enumerate(tags) for b in tags[i + 1:]]
    S = torch.linalg.svd(M, full_matrices=False)[1]
    share = float((S[0] ** 2) / (S ** 2).sum())
    shared = torch.linalg.svd(M, full_matrices=False)[2][0]
    shared = shared / shared.norm()
    resid = {}
    for t in tags:
        r = dirs[t] - (dirs[t] @ shared) * shared
        resid[t] = r / r.norm()
    rpairs = [float(abs(resid[a] @ resid[b])) for i, a in enumerate(tags) for b in tags[i + 1:]]

    def table(vecs):
        out = {}
        for s_ in tags:
            d = project_out_dce(key, vecs[s_])
            row = {}
            for t in tags:
                mm, sl = mk[t]
                am = float(d[mm].abs().mean()); ag = float(d[~sl].abs().mean())
                row[t] = round(am / ag, 3) if ag > 0 else None
            out[s_] = row
        return out

    ft, rt = table(dirs), table(resid)
    nf, nr = n_sel(ft, tags, 1.10), n_sel(rt, tags, 1.10)
    mean_full = sum(pairs) / len(pairs); mean_res = sum(rpairs) / len(rpairs)
    substrate = share >= 0.80
    reversal = substrate and mean_res < 0.50 and nr > nf
    arrangement = ('a8-like: substrate WITH separable circuits under it' if reversal else
                   'm16-like: substrate with NOTHING separable under it' if substrate else
                   'a16-like: no shared substrate, circuits already separate')
    return {'circuits': tags, 'n': len(tags),
            'shared_variance_explained': round(share, 4),
            'mean_pairwise_abs_cos_full': round(mean_full, 4),
            'mean_pairwise_abs_cos_residual': round(mean_res, 4),
            'full_selective_10pct': f'{nf}/{len(tags)}', 'residual_selective_10pct': f'{nr}/{len(tags)}',
            'has_shared_substrate': bool(substrate), 'shows_two_level_reversal': bool(reversal),
            'arrangement': arrangement,
            'concentration_full': ft, 'concentration_residual': rt}


t0 = time.time()
RES = {}
for key in COMPS:
    r = study(key)
    if r is None:
        print(f'  {key}: skipped (fewer than 3 usable circuits)', flush=True)
        continue
    RES[key] = r
    print(f'  {key:4s} n={r["n"]:2d}  shared {r["shared_variance_explained"]:.4f}  '
          f'|cos| {r["mean_pairwise_abs_cos_full"]:.4f}  resid |cos| '
          f'{r["mean_pairwise_abs_cos_residual"]:.4f}  sel full {r["full_selective_10pct"]} -> resid '
          f'{r["residual_selective_10pct"]}  | {r["arrangement"]} ({time.time()-t0:.0f}s)', flush=True)

subs = [k for k, v in RES.items() if v['has_shared_substrate']]
revs = [k for k, v in RES.items() if v['shows_two_level_reversal']]
xs = [v['shared_variance_explained'] for v in RES.values()]
ys = [v['mean_pairwise_abs_cos_full'] for v in RES.values()]
n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
sx = sum((a - mx) ** 2 for a in xs) ** 0.5; sy = sum((b - my) ** 2 for b in ys) ** 0.5
r_pearson = cov / (sx * sy) if sx > 0 and sy > 0 else 0.0

rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude',
       'method': 'per-circuit direction = unit(mean over members - mean off slice) in a component output '
                 'space; shared direction = top right-singular vector; residual = direction with the '
                 'shared component removed; selectivity by rank-1 projection ablation at a >=10% margin. '
                 'Identical instrument at every component, all in one run.',
       'min_circuits': MIN_CIRCUITS, 'components': COMPS,
       'pred_a_substrate_is_majority': bool(len(subs) >= 4),
       'pred_b_reversal_unique_to_a8': bool(revs == ['a8']),
       'pred_c_summaries_agree': bool(abs(r_pearson) >= 0.90),
       'components_with_substrate': subs, 'components_with_reversal': revs,
       'pearson_shared_vs_cos': round(r_pearson, 4),
       'note': 'read-only artifact; no circuit file was modified', 'by_component': RES}
json.dump(rep, open('circuits/SUBSTRATE_CENSUS.json', 'w'), indent=1)

print(f'\nwrote circuits/SUBSTRATE_CENSUS.json ({time.time()-t0:.0f}s)')
print(f'pred_a  shared substrate in {len(subs)}/{len(RES)} components {subs} (bar >=4) : '
      f'{rep["pred_a_substrate_is_majority"]}')
print(f'pred_b  two-level reversal unique to a8: found in {revs} (bar == [a8]) : '
      f'{rep["pred_b_reversal_unique_to_a8"]}')
print(f'pred_c  Pearson(shared_variance, mean|cos|) = {r_pearson:.4f} (bar |r|>=0.90) : '
      f'{rep["pred_c_summaries_agree"]}')
