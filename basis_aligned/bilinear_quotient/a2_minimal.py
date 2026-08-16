"""Fifth attempt, and a constructive one: WHICH part of the residual does the damage?

Four rounds of guessing summary statistics went 0-for-4 on the main question
(reader coherence, coupling concentration, reader alignment, reader profile), and
the transplant control in `a2_profile.py` showed why: the damage is intrinsic to
the residual, not a relationship between residual and signal, so relational
statistics could never have been it.

So stop guessing and search. Decompose the residual by its own reader-mode SVD
into 23 rank-one pieces, and ask directly which pieces carry the damage:

  * remove each piece singly and measure recovery;
  * greedily accumulate the pieces whose removal helps, until recovery is restored;
  * compare against removing the same number of pieces chosen by singular value
    (top-down and bottom-up) and at random.

Everything is scored at the SAME off-block mass — the remainder is rescaled after
removal — so the comparison is not about how much was deleted. The unrescaled
version is reported too, so the effect cannot be an artefact of amplification.

If a small-mass set restores recovery, the damage is localised and describable. If
no set short of most of the residual works, it is genuinely diffuse and the honest
answer to the open question is that there is no compact culprit.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
import a2_calibrate as cal
from a2_coherence import reader_unfold, off_block_part, scale_to_off_block, FREQS
from a2_profile import unvec
from fix_blind_eps import blind_and_oracle
from bq_common import interaction

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)   # JADE is CPU-bound; grabbing every thread thrashes badly
P = m.P
GOAL = 9          # frequencies recovered that count as "restored"


def reader_pieces(R):
    """R = sum_j sigma_j u_j (x) V_j, as m separate (m,d,d) rank-one-in-reader parts."""
    d = R.shape[-1]
    V = reader_unfold(R)
    U, s, Vh = torch.linalg.svd(V, full_matrices=False)
    pieces = []
    for j in range(len(s)):
        pieces.append(unvec(s[j] * torch.outer(U[:, j], Vh[j]), d))
    return pieces, s


def score(base, E, target, tag):
    if float(E.norm()) < 1e-12:
        return None
    Qp = scale_to_off_block(base, off_block_part(E), target)
    r = blind_and_oracle(Qp, tag, verbose=False)
    return r['blind']['n_freq_full'], r['oracle']['n_freq_full']


def main():
    t0 = time.time()
    x, y, a, b = m.all_pairs()
    basis = m.identifiable_basis(x)
    cache = torch.load('a2_cache.pt', weights_only=False)
    allw = list(range(0, P // 2 + 1))
    out = {'goal': GOAL}

    Qid, _ = m.canonicalise(interaction(cache[0]['p']), basis)
    Qb = m.block_project(Qid, allw)
    Res = off_block_part(Qid - Qb)
    target = m.fourier_power(Qid)['off_block']
    pieces, sv = reader_pieces(Res)
    mass = (sv ** 2) / (sv ** 2).sum()
    print(f'residual decomposed into {len(pieces)} reader-mode pieces; '
          f'top piece holds {float(mass[0]):.3f} of the mass')
    base = score(Qb, Res, target, 'full residual')
    print(f'baseline, full residual: blind {base[0]}/11  oracle {base[1]}/11')
    out['baseline'] = {'blind': base[0], 'oracle': base[1]}

    print('\n== 1. remove each reader piece singly (remainder rescaled to equal mass) ==')
    singles = []
    for j in range(len(pieces)):
        E = Res - pieces[j]
        r = score(Qb, E, target, f'drop {j}')
        singles.append({'piece': j, 'mass_share': float(mass[j]),
                        'blind': r[0], 'oracle': r[1]})
    out['singles'] = singles
    best = sorted(singles, key=lambda v: -(v['oracle'] + v['blind']))[:6]
    for v in best:
        print(f"  drop piece {v['piece']:2d} (mass {v['mass_share']:.3f}) -> "
              f"blind {v['blind']:2d}/11  oracle {v['oracle']:2d}/11")
    print(f"  ... worst: " + ', '.join(
        f"{v['piece']}({v['oracle']})" for v in sorted(singles, key=lambda v: v['oracle'])[:5]))

    print('\n== 2. remove pieces cumulatively, ranked by their single-removal benefit ==')
    # ranking once and removing in that order costs 23 scorings instead of the ~240 a
    # re-ranking greedy needs, and JADE here is CPU-bound. Slightly less optimal, but
    # the question is whether a SMALL set suffices, which this answers either way.
    order = [v['piece'] for v in sorted(singles, key=lambda v: -(v['oracle'] + v['blind']))]
    chosen, cur = [], Res.clone()
    greedy = []
    for step, j in enumerate(order[:14]):
        chosen.append(j)
        cur = cur - pieces[j]
        r = score(Qb, cur, target, f'cum {step}')
        if r is None:
            break
        removed = float(sum(mass[k] for k in chosen))
        greedy.append({'step': step + 1, 'piece': j, 'cumulative_mass_removed': removed,
                       'oracle': r[1], 'blind': r[0], 'chosen': list(chosen)})
        print(f"  after {step+1:2d} pieces (last: {j:2d}) | cumulative mass removed "
              f"{removed:.3f} | blind {r[0]:2d}/11  oracle {r[1]:2d}/11", flush=True)
        if r[1] >= GOAL:
            break
    out['greedy'] = greedy

    n_removed = len(chosen)
    print(f'\n== 3. baselines at the same count ({n_removed} pieces removed) ==')
    out['baselines'] = []
    orders = {'greedy (found above)': chosen,
              'top by singular value': list(range(n_removed)),
              'bottom by singular value': list(range(len(pieces) - n_removed, len(pieces)))}
    g = torch.Generator().manual_seed(0)
    orders['random'] = torch.randperm(len(pieces), generator=g)[:n_removed].tolist()
    for tag, idx in orders.items():
        E = Res.clone()
        for j in idx:
            E = E - pieces[j]
        r = score(Qb, E, target, tag)
        removed = float(sum(mass[j] for j in idx))
        row = {'strategy': tag, 'pieces': list(idx), 'mass_removed': removed,
               'blind': r[0], 'oracle': r[1]}
        out['baselines'].append(row)
        print(f"  {tag:24s} mass removed {removed:.3f} -> blind {r[0]:2d}/11  "
              f"oracle {r[1]:2d}/11")

    print('\n== 4. control: is the effect just amplification of the remainder? ==')
    E = Res.clone()
    for j in chosen:
        E = E - pieces[j]
    Qp_unscaled = Qb + E
    r_un = blind_and_oracle(Qp_unscaled, 'unrescaled', verbose=False)
    out['unrescaled'] = {'off_block': r_un['off_block'],
                         'blind': r_un['blind']['n_freq_full'],
                         'oracle': r_un['oracle']['n_freq_full']}
    print(f"  greedy removal WITHOUT rescaling: off-block "
          f"{out['unrescaled']['off_block']:.4f} (vs {target:.4f}) -> "
          f"blind {out['unrescaled']['blind']}/11  oracle {out['unrescaled']['oracle']}/11")

    print('\n== 5. does the damaging part transplant? ==')
    Qstar = cal.planted_family(FREQS)
    dam = sum(pieces[j] for j in chosen)
    r = score(Qstar, dam, target, 'damaging part alone, on the planted signal')
    out['damaging_transplant'] = {'blind': r[0], 'oracle': r[1],
                                  'mass_share': float(sum(mass[j] for j in chosen))}
    print(f"  the removed pieces alone, on the PLANTED signal: blind {r[0]}/11  "
          f"oracle {r[1]}/11   (they are {out['damaging_transplant']['mass_share']:.3f} "
          f"of the residual's mass)")

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_minimal_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
