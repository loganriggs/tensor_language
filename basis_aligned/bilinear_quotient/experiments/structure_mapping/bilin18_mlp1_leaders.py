"""What ARE the ten directions that run MLP1?

§13 replaced "irreducibly distributed" with a concrete answer: MLP1's causal content is
carried by ~10 effective directions, the leader holding 27% of the layer. §74 (repo)
says none of these is nameable -- but its battery scored candidates by SOLO ablation
z-scores, the instrument §13 showed misranks, and it never had the Shapley ranking to
know which directions to even look at. §8.3 built a naming instrument that survived
verification on layer 17 (unembedding alignment checked against measured excitation over
a large corpus with a permutation null, after a first attempt that had no power). This
points that instrument at the six leading Shapley directions of MLP1.

Each direction q is an OUTPUT direction of MLP1: the layer writes c(x) = mo(x).q onto
the residual stream along q. Three descriptions per direction, two of them testable
against each other:

  writes    which tokens the write promotes/demotes if it reached the unembedding
            unchanged (wte . q, signed) -- the standard shortcut
  fires-on  which CURRENT tokens sit at the positions where |c| is largest -- measured,
            not assumed
  verdict   Spearman rho between |wte . q| and per-token mean c^2 across all tokens
            with >= MIN_COUNT occurrences, against a 200-draw permutation null; plus
            the same for the NEXT token, since a write can serve the prediction rather
            than describe the position

The honest expectation from §74 is that these are NOT crisply nameable (its z-battery
found nothing, and depth-0-1 layers feed 16 more layers before the unembedding, so the
'writes' column is a long-range extrapolation). The value of running it anyway: §13 says
these ten directions are where the causal mass IS, so even a partial name is a name for
something that matters, and a verified failure to name is the honest version of §74's
boundary -- measured with a working instrument on the right candidates.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_joint_removal import fwd, collect_out, orth, m, LAYER, DEV
from bilin18_joint_removal import FW

MIN_COUNT = 30
N_PERM = 200
N_DIR = 6
enc = tiktoken.get_encoding('gpt2')


def spearman(a, b):
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-30))


@torch.no_grad()
def collect_with_tokens(seqs, li):
    accs, curs, nxts = [], [], []
    for i in range(0, seqs.shape[0], 6):
        b = seqs[i:i + 6].to(DEV)
        acc = []
        fwd(b, collect=li, acc=acc)
        accs.append(acc[0])
        curs.append(b.reshape(-1))
        nxts.append(torch.cat([b[:, 1:], b[:, :1] * 0 - 1], 1).reshape(-1))
    return torch.cat(accs, 0), torch.cat(curs, 0), torch.cat(nxts, 0)


def main():
    t0 = time.time()
    R = json.load(open('bilin18_shapley_results.json'))
    phi = torch.tensor(R['shapley'])
    # The basis MUST come from the same data the Shapley run used, or "direction i"
    # here is not the vector that earned Shapley value phi_i. (The first two versions
    # of this script recomputed the SVD from their own corpora, and the tail
    # directions rotated -- caught because the 'writes toward' lists changed between
    # runs that should have differed only in excitation power.)
    from bilin18_joint_removal import TRAIN
    Ysh = collect_out(TRAIN, LAYER)
    _, Sv, Vh = torch.linalg.svd(Ysh - Ysh.mean(0), full_matrices=False)
    Q = orth(Vh[:32].T)
    # excitation, by contrast, wants all the corpus it can get
    seqs = FW[:512, :513]
    MO, cur, nxt = collect_with_tokens(seqs, LAYER)
    Yc = MO - MO.mean(0)
    order = phi.argsort(descending=True)
    wte = m.transformer.wte.weight.detach().float()
    uniq, counts = cur.unique(return_counts=True)
    keep = uniq[(counts >= MIN_COUNT) & (uniq >= 0)]
    print(f'{MO.shape[0]:,} positions | {keep.numel()} tokens with >= {MIN_COUNT} '
          f'occurrences\n')
    g = torch.Generator().manual_seed(0)
    out = {'n_positions': int(MO.shape[0]), 'n_tokens': int(keep.numel()),
           'directions': []}

    for r in range(N_DIR):
        i = int(order[r])
        q = Q[:, i].float()
        share = float(phi[i]) / float(phi.sum())
        c = (Yc.float() @ q)
        a = c ** 2
        gmean = a.mean()
        name = (wte @ q)
        promotes = [enc.decode([t]) for t in name.argsort(descending=True)[:8].tolist()]
        demotes = [enc.decode([t]) for t in name.argsort()[:8].tolist()]
        exc_c = torch.stack([a[cur == t].mean() for t in keep])
        exc_n = torch.stack([a[nxt == t].mean() if (nxt == t).any() else gmean
                             for t in keep])
        nm = name[keep].abs()
        rc, rn = spearman(nm, exc_c), spearman(nm, exc_n)
        null = sorted(abs(spearman(nm[torch.randperm(keep.numel(), generator=g)],
                                   exc_c)) for _ in range(N_PERM))
        p95 = null[int(0.95 * N_PERM)]
        top = keep[exc_c.argsort(descending=True)[:8]]
        fires = [enc.decode([t]) for t in top.tolist()]
        ratios = [round(float(v / gmean), 1)
                  for v in exc_c.sort(descending=True).values[:8]]
        ok = max(abs(rc), abs(rn)) > p95
        rec = {'rank': r + 1, 'svd_index': i, 'shapley': float(phi[i]),
               'share': share, 'promotes': promotes, 'demotes': demotes,
               'fires_on': fires, 'fire_ratios': ratios,
               'rho_current': rc, 'rho_next': rn, 'null_p95': p95,
               'named': bool(ok)}
        out['directions'].append(rec)
        print(f'#{r+1}  direction {i}  Shapley {phi[i]:+.4f} ({100*share:.0f}% of the '
              f'layer)')
        print(f'    writes toward:  {promotes}')
        print(f'    writes against: {demotes}')
        print(f'    fires on (measured): {fires}  (x mean: {ratios})')
        print(f'    rho current {rc:+.3f} | next {rn:+.3f} | null p95 {p95:.3f}  '
              f'-> {"NAMED" if ok else "not nameable"}', flush=True)
        print()

    n_named = sum(d['named'] for d in out['directions'])
    out['n_named'] = n_named
    print(f'{n_named}/{N_DIR} of the causally leading directions clear their '
          f'permutation null')
    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_mlp1_leaders_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
