"""A5 follow-ups: the null that actually tests sharing, and the arm where a
dictionary can beat PCA.

Two things the first A5 run showed the plan gets wrong, both fixed here:

1. The reader shuffle CANNOT test sharing. A component that is literally
   identical across readers contributes identical coordinates to every reader,
   so permuting readers leaves it untouched — measured ratio 4.00 before the
   shuffle and 3.5-4.0 after. The null that does the job is a matched family with
   NO sharing: same number of readers, same number of forms, same norms, each
   form used once. That is what `no_sharing_null` builds.

2. A dictionary cannot beat PCA on mutually ORTHOGONAL atoms — PCA finds the
   right subspace and there is nothing left to gain, which is what the first run
   measured (identical errors at every budget). The doc's PCA-orthogonality claim
   is about OVERLAPPING structure, so this adds the hierarchical arm it names:
   a coarse form plus refinements that share it.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from a5_shared import orthonormal_forms, match_atoms, D
from bq_common import reader_moment, spectrum, fit_dictionary, lam_relerr, form_shuffle

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


def moment_ratio(Qu):
    ev = spectrum(reader_moment(Qu))
    return float(ev[0] / ev[1].clamp_min(1e-30)), [float(v) for v in ev[:6]]


def pca_frontier(Qu, ks):
    iu, ju = torch.triu_indices(D, D)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Qu)
    V = Qu[:, iu, ju] * sc
    U_, s_, Vh_ = torch.linalg.svd(V, full_matrices=False)
    out = {}
    for k in ks:
        rec = (U_[:, :k] * s_[:k]) @ Vh_[:k]
        Qh = torch.zeros_like(Qu)
        Qh[:, iu, ju] = rec / sc
        Qh[:, ju, iu] = rec / sc
        out[k] = lam_relerr(Qu, Qh)
    return out, [float(v) for v in s_[:8]]


def sharing_vs_no_sharing(R=3, seed=0):
    """The correct control for 'this form is shared': a matched family in which
    it is not."""
    shared = orthonormal_forms(R + 1, seed)
    Q_share = torch.stack([shared[0] + shared[u + 1] for u in range(R)])
    ind = orthonormal_forms(2 * R, seed + 1)
    Q_indep = torch.stack([ind[2 * u] + ind[2 * u + 1] for u in range(R)])
    res = {'R': R}
    for tag, Q in (('shared', Q_share), ('no_sharing_null', Q_indep)):
        r, ev = moment_ratio(Q)
        sh = [moment_ratio(form_shuffle(Q, seed=s))[0] for s in range(3)]
        res[tag] = {'ratio': r, 'spectrum': ev, 'ratio_after_reader_shuffle': sh}
    res['analytic_ratio_if_shared'] = R + 1
    res['analytic_top_eigvec_overlap_with_shared'] = math.sqrt(R / (R + 1))
    print(f"  shared family:      top/next {res['shared']['ratio']:.3f} "
          f"(analytic R+1 = {R+1}) | after reader shuffle "
          f"{[round(v,2) for v in res['shared']['ratio_after_reader_shuffle']]}")
    print(f"  no-sharing control: top/next {res['no_sharing_null']['ratio']:.3f} "
          f"| after reader shuffle "
          f"{[round(v,2) for v in res['no_sharing_null']['ratio_after_reader_shuffle']]}")
    print(f"  => the reader shuffle separates nothing ({res['shared']['ratio']:.2f} vs "
          f"{res['shared']['ratio_after_reader_shuffle'][0]:.2f}); the no-sharing control "
          f"separates cleanly ({res['shared']['ratio']:.2f} vs "
          f"{res['no_sharing_null']['ratio']:.2f})")
    return res


def hierarchical(n_readers=9, seed=0, overlap=0.8):
    """A coarse form plus refinements that CONTAIN it — the overlapping structure
    the doc says PCA should smear. Reader u reads coarse + its own refinement,
    where each refinement is itself mostly the coarse form."""
    base = orthonormal_forms(1 + 3, seed + 11)
    coarse, spokes = base[0], base[1:]
    g = torch.Generator().manual_seed(seed)
    atoms, Qu = [], []
    for j in range(3):                       # three refinements of one coarse form
        A = overlap * coarse + math.sqrt(1 - overlap ** 2) * spokes[j]
        atoms.append(A / A.norm())
    for u in range(n_readers):
        w = torch.zeros(3)
        w[u % 3] = 1.0
        w = w + 0.15 * torch.randn(3, generator=g)
        Qu.append(sum(float(w[j]) * atoms[j] for j in range(3)))
    Qu = torch.stack(Qu)
    ks = [1, 2, 3, 4, 5]
    pca, sv = pca_frontier(Qu, ks)
    res = {'overlap': overlap, 'n_readers': n_readers, 'pca_singular_values': sv,
           'atom_mutual_cos': float((atoms[0] * atoms[1]).sum()), 'rows': {}}
    for k in ks:
        A, C, derr = fit_dictionary(Qu, k, steps=4000, lr=3e-2, l1=1e-2, seed=seed)
        m = match_atoms(A, atoms) if k >= 3 else None
        res['rows'][k] = {'pca_err': pca[k], 'dict_err': derr,
                          'match_cos': [x['cos'] for x in m] if m else None}
        mm = f" | dictionary atoms vs planted {[round(x['cos'], 3) for x in m]}" if m else ''
        print(f"    budget {k}: PCA err {pca[k]:.4f} | dictionary err {derr:.4f}{mm}")
    # what PCA's own components look like against the planted atoms
    iu, ju = torch.triu_indices(D, D)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Qu)
    V = Qu[:, iu, ju] * sc
    Vh = torch.linalg.svd(V, full_matrices=False)[2][:3]
    P = torch.zeros(3, D, D, device=Qu.device, dtype=Qu.dtype)
    P[:, iu, ju] = Vh / sc
    P[:, ju, iu] = Vh / sc
    res['pca_component_match'] = [x['cos'] for x in match_atoms(P, atoms)]
    print(f"    PCA's own 3 components vs the planted atoms: "
          f"{[round(v, 3) for v in res['pca_component_match']]} "
          f"(planted atoms overlap each other at cos {res['atom_mutual_cos']:.2f})")
    return res


def main():
    t0 = time.time()
    out = {}
    print('== the null that actually tests sharing ==')
    out['sharing_control'] = {str(R): sharing_vs_no_sharing(R) for R in (3, 5, 8)}

    print('\n== hierarchical / overlapping arm: can a dictionary beat PCA here? ==')
    out['hierarchical'] = {}
    for ov in (0.0, 0.5, 0.8, 0.95):
        print(f"  overlap {ov}:")
        out['hierarchical'][str(ov)] = hierarchical(overlap=ov)

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a5_extra_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
