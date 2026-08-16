"""A2 addendum: is the trained model's residual harder for block recovery than
isotropic noise of the same size? Calibrate JADE against STRUCTURED noise
(off-block frequency-pair coupling, the kind training actually leaves) and
against a memorisation-shaped residual, on the same off-block axis."""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
import a2_calibrate as cal
from a2_jade import frontier
from bq_common import interaction

torch.set_default_dtype(torch.float64)
P = m.P
freqs = list(range(1, P // 2 + 1))


def structured_noise(seed=1):
    Qstar = cal.planted_family(freqs)
    E = torch.zeros_like(Qstar)
    g = torch.Generator().manual_seed(seed)
    for wa in freqs:
        for wb in freqs:
            if wa == wb:
                continue
            Ba, Bb = m.FBLOCKS[wa], m.FBLOCKS[wb]
            C = torch.randn(P, 4, 4, generator=g).to(Qstar)
            T = torch.einsum('ip,mpq,jq->mij', Ba, C, Bb)
            E += T + T.transpose(1, 2)
    return Qstar, E / E.norm() * Qstar.norm()


def memorisation_noise(x, y, seed=2):
    """The residual a memorising model carries: a low-rank, data-aligned term
    that raises the train-set logits, expressed in Sym²."""
    Qstar = cal.planted_family(freqs)
    g = torch.Generator().manual_seed(seed)
    tr, te = m.split(0)
    E = torch.zeros_like(Qstar)
    xt = x[tr]
    yt = y[tr]
    for i, (xi, yi) in enumerate(zip(xt, yt)):
        E[int(yi)] += torch.outer(xi, xi)
    E = E - E.mean(0, keepdim=True)
    return Qstar, E / E.norm() * Qstar.norm()


def main():
    x, y, a, b = m.all_pairs()
    out = {}
    for tag, (Qstar, E) in (('structured', structured_noise()),
                            ('memorisation', memorisation_noise(x, y))):
        print(f'== JADE calibration vs {tag} noise ==')
        out[tag] = []
        for eta in (0.1, 0.2, 0.35, 0.5, 0.7):
            Qn = Qstar + eta * E
            ob = m.fourier_power(Qn)['off_block']
            print(f'  eta {eta:.2f} (off-block {ob:.4f}):')
            r = frontier(Qn, f'{tag}{eta}', verbose=False)
            r['eta'] = eta
            out[tag].append(r)
            print(f"    --> best {r['best_n_freq_full']}/11 at eps {r['best_tol']} "
                  f"(in-block {r['best_in_block_mass']:.4f})")
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_jade_struct.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
