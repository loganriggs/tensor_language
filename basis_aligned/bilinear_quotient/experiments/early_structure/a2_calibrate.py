"""A2 calibration — certify the block-recovery instrument on known truth BEFORE
believing (or disbelieving) what it says about a trained model.

Builds the exact planted modular-addition family
    Q_c = [[0, ½N_c],[½N_cᵀ, 0]],  N_c[a,b] = Σ_ω α_ω cos(2πω(a+b-c)/p)
whose ∗-algebra blocks are known (one 4-dim Fourier block per used frequency),
perturbs it by a controlled amount of noise, and measures where exact SBD and
approximate SBD stop recovering the planted decomposition. The output is the
instrument's operating range, on the same axis (off-block mass fraction) as the
number measured on the trained model.
"""

import json
import math
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
from bq_common import (sbd, sbd_robust, block_mass, support_basis, restrict,
                       isotypic_groups)

torch.set_default_dtype(torch.float64)
DEV = m.DEV
P = m.P


def planted_family(freqs, alphas=None, device=DEV):
    """Exact ideal solution using the given frequency set."""
    alphas = alphas or [1.0] * len(freqs)
    t = torch.arange(P, device=device, dtype=torch.get_default_dtype())
    A, B = torch.meshgrid(t, t, indexing='ij')
    Q = torch.zeros(P, 2 * P, 2 * P, device=device, dtype=torch.get_default_dtype())
    for c in range(P):
        N = torch.zeros(P, P, device=device, dtype=Q.dtype)
        for w, al in zip(freqs, alphas):
            N += al * torch.cos(2 * math.pi * w * (A + B - c) / P)
        Q[c, :P, P:] = 0.5 * N
        Q[c, P:, :P] = 0.5 * N.T
    return Q


def identify_groups(V, sizes, groups):
    """Merge the fine blocks of each isotypic component and match the component
    to a planted Fourier frequency subspace. The isotypic component is the
    canonical object: when an irrep has multiplicity > 1 (here 2, from the a↔b
    exchange symmetry of a+b) the split into individual blocks is NOT unique."""
    offs = [0]
    for s in sizes:
        offs.append(offs[-1] + s)
    out = []
    for gidx in groups:
        cols = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in gidx])
        Vb = V[:, cols]
        w = max(m.FBLOCKS, key=lambda k: float(((Vb.T @ m.FBLOCKS[k]) ** 2).sum()))
        out.append({'dim': int(Vb.shape[1]), 'n_fine': len(gidx), 'freq': int(w),
                    'overlap': float(((Vb.T @ m.FBLOCKS[w]) ** 2).sum() / Vb.shape[1])})
    return out


def score(V, sizes, groups, in_block_mass, extra=None):
    ids = identify_groups(V, sizes, groups)
    good = [g for g in ids if g['dim'] == 4 and g['overlap'] > 0.9]
    r = {'fine_sizes': sizes[:24], 'n_fine_blocks': len(sizes),
         'n_components': len(groups), 'component_dims': sorted(g['dim'] for g in ids),
         'in_block_mass': in_block_mass,
         'n_freq_recovered': len(good), 'freqs': sorted(g['freq'] for g in good),
         'mean_overlap_components': float(sum(g['overlap'] for g in ids) / max(1, len(ids)))}
    if extra:
        r.update(extra)
    return r


def analyse(Q, n_blocks_hint, energy=0.999, label=''):
    U, sv = support_basis(Q, thresh=1e-6)
    cum = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    k = int((cum < energy).sum()) + 1
    Ur = U[:, :k]
    Qr = restrict(Q, Ur)
    fp = m.fourier_power(Q)
    res = {'label': label, 'support_dim_kept': k, 'off_block_mass': fp['off_block']}

    Pe, sizes_e, info_e = sbd(Qr, gap_rel=1e-5)
    me, _ = block_mass(Qr, Pe, sizes_e)
    ge = isotypic_groups(Qr, Pe, sizes_e)
    res['exact'] = score(Ur @ Pe, sizes_e, ge, me,
                         {'commutant_dim': info_e['commutant_dim']})

    for tag, nb in (('auto', None), ('hinted', n_blocks_hint)):
        Pr, sizes_r, info_r = sbd_robust(Qr, n_blocks=nb)
        gr = isotypic_groups(Qr, Pr, sizes_r, commutant_dim=info_r['commutant_dim'])
        res[f'robust_{tag}'] = score(Ur @ Pr, sizes_r, gr, info_r['in_block_mass'],
                                     {'commutant_dim': info_r['commutant_dim'],
                                      'defect': info_r['defect']})
    return res


def main():
    out = {'p': P}
    freqs = list(range(1, P // 2 + 1))              # all 11 frequencies
    Qstar = planted_family(freqs)

    print('== the planted family itself ==')
    r0 = analyse(Qstar, 2 * len(freqs), label='planted')
    for tag in ('exact', 'robust_auto', 'robust_hinted'):
        e = r0[tag]
        print(f"  {tag:14s} fine {e['n_fine_blocks']:2d} blocks -> {e['n_components']:2d} components "
              f"dims {e['component_dims'][:12]} | freqs recovered {e['n_freq_recovered']}/11 "
              f"in-block {e['in_block_mass']:.6f} commutant {e['commutant_dim']}")
    out['planted'] = r0

    print('\n== sparse-frequency planted family (5 frequencies) ==')
    Qs5 = planted_family([1, 3, 4, 7, 9])
    r5 = analyse(Qs5, 10, label='planted5')
    for tag in ('exact', 'robust_auto'):
        e = r5[tag]
        print(f"  {tag:14s} fine {e['n_fine_blocks']:2d} -> {e['n_components']:2d} components "
              f"freqs {e['freqs']} in-block {e['in_block_mass']:.6f}")
    out['planted5'] = r5

    print('\n== noise sweep: isotropic symmetric noise ==')
    out['noise_iso'] = []
    g = torch.Generator().manual_seed(0)
    E = torch.randn(P, 2 * P, 2 * P, generator=g).to(Qstar)
    E = 0.5 * (E + E.transpose(1, 2))
    E = E / E.norm() * Qstar.norm()
    for eta in (0.0, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.5):
        r = analyse(Qstar + eta * E, 2 * len(freqs), label=f'iso{eta}')
        r['eta'] = eta
        print(f"  eta {eta:5.3f} offblk {r['off_block_mass']:.4f} | exact freqs "
              f"{r['exact']['n_freq_recovered']:2d}/11 | robust auto: fine {r['robust_auto']['n_fine_blocks']:2d} "
              f"comp {r['robust_auto']['n_components']:2d} freqs {r['robust_auto']['n_freq_recovered']:2d}/11 "
              f"inblk {r['robust_auto']['in_block_mass']:.4f} | hinted freqs "
              f"{r['robust_hinted']['n_freq_recovered']:2d}/11 inblk "
              f"{r['robust_hinted']['in_block_mass']:.4f} ovl "
              f"{r['robust_hinted']['mean_overlap_components']:.3f} defect "
              f"{r['robust_hinted']['defect']:.2e}")
        out['noise_iso'].append(r)

    print('\n== noise sweep: structured off-block Fourier coupling (what training leaves) ==')
    out['noise_struct'] = []
    # couple frequency pairs w_a != w_b, i.e. exactly the mass the trained model has
    Es = torch.zeros_like(Qstar)
    g2 = torch.Generator().manual_seed(1)
    for wa in freqs:
        for wb in freqs:
            if wa == wb:
                continue
            Ba, Bb = m.FBLOCKS[wa], m.FBLOCKS[wb]
            C = torch.randn(P, 4, 4, generator=g2).to(Qstar)
            T = torch.einsum('ip,mpq,jq->mij', Ba, C, Bb)
            Es += T + T.transpose(1, 2)
    Es = Es / Es.norm() * Qstar.norm()
    for eta in (0.0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.5, 0.7):
        r = analyse(Qstar + eta * Es, 2 * len(freqs), label=f'struct{eta}')
        r['eta'] = eta
        print(f"  eta {eta:5.3f} offblk {r['off_block_mass']:.4f} | exact freqs "
              f"{r['exact']['n_freq_recovered']:2d}/11 | robust auto: fine {r['robust_auto']['n_fine_blocks']:2d} "
              f"comp {r['robust_auto']['n_components']:2d} freqs {r['robust_auto']['n_freq_recovered']:2d}/11 "
              f"inblk {r['robust_auto']['in_block_mass']:.4f} | hinted freqs "
              f"{r['robust_hinted']['n_freq_recovered']:2d}/11 inblk "
              f"{r['robust_hinted']['in_block_mass']:.4f} ovl "
              f"{r['robust_hinted']['mean_overlap_components']:.3f} defect "
              f"{r['robust_hinted']['defect']:.2e}")
        out['noise_struct'].append(r)

    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_calibration.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
