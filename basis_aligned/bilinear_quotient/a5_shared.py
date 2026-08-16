"""A5 — shared vs private subcomputation: the reader-weighted global view.

DGP. One shared quadratic form feeds three readers; each reader also has one
private form of its own. Reader u's total form is
    Q_u = g_u · S_shared + h_u · S_private[u]
and the readers are explicit rows of a downstream linear map, so the layer has to
route a single piece of work to three different consumers.

Predictions (doc A5), registered before running:
 (i)   the top eigenvector of the reader-weighted second moment is the shared
       form, with an eigenvalue about 3x the private ones (the reader multiplicity);
 (ii)  a sparse dictionary over the reader forms recovers shared and private as
       SEPARATE atoms, with the right sparse mixing per reader;
 (iii) decomposing each reader's form on its own finds the shared form three
       times over, in three mutually incompatible bases — the concrete reason to
       fit globally and explain paths afterwards, rather than the reverse;
 (iv)  under the reader shuffle the shared/private gap collapses.

Also the PCA-orthogonality test: plant a coarse form plus overlapping refinements
and compare PCA, dictionary and block fits on functional error against description
length.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import (init_params, forward, interaction, train, reader_moment,
                       spectrum, participation_ratio, fit_dictionary, fit_cp,
                       form_shuffle, reader_shuffle, gauge_refactor, lam_relerr,
                       lam_cos, lam_norm2_each, reader_forms)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

D = 12          # input dim
MH = 8          # intermediate width the readers read from
H = 96
R = 3           # readers
STEPS = 12000
BATCH = 512

REGISTERED = {
    'i': 'top eigenvalue of the reader-weighted moment belongs to the shared form, '
         'ratio to the private ones about 3 (the number of readers sharing it)',
    'ii': 'a 4-atom sparse dictionary separates shared from private, each reader '
          'using exactly 2 atoms',
    'iii': 'per-reader decomposition finds the shared form once per reader, in '
           'bases that do not agree with each other',
    'iv': 'the reader shuffle collapses the shared/private eigenvalue gap',
}


def orthonormal_forms(n, seed):
    """n mutually orthogonal unit symmetric forms.

    NOT random positive-semidefinite matrices: those all have positive trace and
    therefore large positive inner products with each other, which plants a
    spurious common direction at the top of every spectrum and confounds the
    whole shared-versus-private question. Orthogonalising in Sym²(V) removes it.
    """
    g = torch.Generator().manual_seed(seed)
    iu, ju = torch.triu_indices(D, D)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0))
    V = torch.linalg.qr(torch.randn(D * (D + 1) // 2, n, generator=g))[0]
    out = []
    for k in range(n):
        S = torch.zeros(D, D)
        S[iu, ju] = V[:, k] / sc
        S[ju, iu] = V[:, k] / sc
        out.append((S / S.norm()).to(DEV))
    return out


def planted(seed=0, gains=(1.0, 1.0, 1.0), priv=(1.0, 1.0, 1.0)):
    forms = orthonormal_forms(R + 1, seed)
    S0, Sp = forms[0], forms[1:]
    U = torch.zeros(R, MH, device=DEV)          # reader u reads intermediate slot u
    for u in range(R):
        U[u, u] = 1.0
    Q = torch.stack([gains[u] * S0 + priv[u] * Sp[u] for u in range(R)])
    return {'S0': S0, 'Sp': Sp, 'U': U, 'Q': Q, 'gains': gains, 'priv': priv}


def data_fn(P, batch=BATCH, seed=[0]):
    g = torch.Generator(device=DEV).manual_seed(seed[0])
    seed[0] += 1

    def gen():
        x = torch.randn(batch, D, generator=g, device=DEV, dtype=torch.get_default_dtype())
        y = torch.einsum('ni,uij,nj->nu', x, P['Q'], x)
        return x, y

    return gen


def match_atoms(A, targets):
    """Greedy |cosine| matching of dictionary atoms to planted forms."""
    Af = A.flatten(1)
    Af = Af / Af.norm(dim=1, keepdim=True).clamp_min(1e-30)
    Tf = torch.stack([t.flatten() for t in targets])
    Tf = Tf / Tf.norm(dim=1, keepdim=True).clamp_min(1e-30)
    C = (Af @ Tf.T).abs()
    out = []
    used = set()
    for j in range(Tf.shape[0]):
        best, bi = -1, None
        for i in range(Af.shape[0]):
            if i in used:
                continue
            if float(C[i, j]) > best:
                best, bi = float(C[i, j]), i
        used.add(bi)
        out.append({'atom': bi, 'cos': best})
    return out


def run(seed=0, gains=(1.0, 1.0, 1.0), priv=(1.0, 1.0, 1.0), verbose=True, tag=''):
    P = planted(seed, gains, priv)
    p = init_params(D, H, MH, seed=seed + 20, device=DEV)
    p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
    gen = data_fn(P)

    def dfn():
        x, y = gen()
        return x, y @ P['U']            # train the 3 reader outputs, MH-wide head

    # the model outputs MH channels; the loss only sees the R reader projections
    def loss_data():
        x, y = gen()
        return x, y

    for k in p:
        p[k].requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for s in range(STEPS):
        x, y = loss_data()
        pred = forward(p, x) @ P['U'].T
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    for k in p:
        p[k].requires_grad_(False)

    x, y = loss_data()
    fvu = float((((forward(p, x) @ P['U'].T) - y) ** 2).mean() / y.var())

    Qi = interaction(p)                       # (MH, d, d) per intermediate channel
    Qu = reader_forms(Qi, P['U'])             # (R, d, d) per reader
    res = {'tag': tag, 'seed': seed, 'gains': list(gains), 'priv': list(priv), 'fvu': fvu,
           'reader_form_error': lam_relerr(P['Q'], Qu),
           'registered': REGISTERED}

    # (i) reader-weighted second moment
    M = reader_moment(Qu)
    ev = spectrum(M)
    res['moment_spectrum'] = [float(v) for v in ev[:8]]
    res['participation_ratio'] = participation_ratio(ev)
    d2 = D * (D + 1) // 2
    iu, ju = torch.triu_indices(D, D)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Qu)
    evec = torch.linalg.eigh(M)[1][:, -1]
    top = torch.zeros(D, D, device=DEV, dtype=Qu.dtype)
    top[iu, ju] = evec / sc
    top[ju, iu] = evec / sc
    res['top_eigvec_cos_shared'] = abs(lam_cos(top[None], P['S0'][None]))
    res['top_eigvec_cos_private'] = [abs(lam_cos(top[None], s[None])) for s in P['Sp']]
    res['eig_ratio_top_to_next'] = float(ev[0] / ev[1].clamp_min(1e-30))

    # (iv) reader-shuffle null on the same statistic
    sh = []
    for s_ in range(3):
        Ms = reader_moment(form_shuffle(Qu, seed=s_))
        evs = spectrum(Ms)
        sh.append(float(evs[0] / evs[1].clamp_min(1e-30)))
    res['eig_ratio_reader_shuffled'] = sh

    # (ii) sparse dictionary over the reader forms
    dic = {}
    for k in (2, 3, 4, 5, 6):
        A, C, err = fit_dictionary(Qu, k, steps=3000, lr=3e-2, l1=2e-2, seed=seed)
        codes = C.abs() / C.abs().max().clamp_min(1e-30)
        dic[k] = {'err': err, 'mean_active_atoms_per_reader':
                  float((codes > 0.15).to(codes.dtype).sum(1).mean()),
                  'match_shared_private': match_atoms(A, [P['S0']] + P['Sp']) if k >= 4 else None}
    res['dictionary'] = dic

    # (iii) per-reader decomposition: is the shared form found once per reader?
    per = []
    for u in range(R):
        ph, e = fit_cp(Qu[u:u + 1], 4, steps=2500, lr=3e-2, seed=seed)
        S = 0.5 * (torch.einsum('ka,kb->kab', ph['L'], ph['R']) +
                   torch.einsum('kb,ka->kab', ph['L'], ph['R']))
        S = S / S.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None]
        per.append({'reader': u, 'cp_err': e,
                    'best_cos_to_shared': max(abs(lam_cos(S[j:j + 1], P['S0'][None]))
                                              for j in range(S.shape[0]))})
    # do the three readers' recovered shared components agree with each other?
    res['per_reader'] = per

    # PCA of the reader forms, for the frontier comparison
    V = (Qu[:, iu, ju] * sc)
    U_, s_, Vh_ = torch.linalg.svd(V, full_matrices=False)
    res['pca_singular_values'] = [float(v) for v in s_]
    pca_err = {}
    for k in (1, 2, 3):
        rec = (U_[:, :k] * s_[:k]) @ Vh_[:k]
        Qh = torch.zeros_like(Qu)
        Qh[:, iu, ju] = rec / sc
        Qh[:, ju, iu] = rec / sc
        pca_err[k] = lam_relerr(Qu, Qh)
    res['pca_error'] = pca_err

    if verbose:
        print(f"  [{tag or 'main'} s{seed} gains={gains} priv={priv}] fvu {fvu:.2e} "
              f"form-err {res['reader_form_error']:.2e}")
        print(f"    (i)  moment spectrum {[round(v, 4) for v in res['moment_spectrum'][:5]]} "
              f"| top/next {res['eig_ratio_top_to_next']:.2f} | top eigvec vs shared "
              f"{res['top_eigvec_cos_shared']:.4f} vs private "
              f"{[round(v, 3) for v in res['top_eigvec_cos_private']]}")
        print(f"    (iv) same ratio after reader shuffle: "
              f"{[round(v, 2) for v in sh]}")
        print(f"    (ii) dictionary: " + ' '.join(
            f"k={k} err {v['err']:.3f} atoms/reader {v['mean_active_atoms_per_reader']:.2f}"
            for k, v in dic.items()))
        if dic[4]['match_shared_private']:
            print(f"         k=4 atom->planted cos: shared "
                  f"{dic[4]['match_shared_private'][0]['cos']:.3f}, private "
                  f"{[round(m['cos'], 3) for m in dic[4]['match_shared_private'][1:]]}")
        print(f"    (iii) per-reader CP best cos to the shared form: "
              f"{[round(v['best_cos_to_shared'], 3) for v in per]}")
        print(f"    PCA error at rank 1/2/3: "
              f"{[round(pca_err[k], 4) for k in (1, 2, 3)]}")
    return res


def run_bank(n_readers=8, n_atoms=4, per_reader=2, seed=0, verbose=True):
    """The doc's A5 gives every reader its own private form — which makes the
    shared/private split UNIDENTIFIABLE for a dictionary. With 3 readers, "one
    atom per reader" costs 3 atoms and the true answer costs 4, so the shortest
    description is the one that hides the sharing, and a sparse fit is right to
    prefer it. Sharing is only worth representing when readers outnumber atoms.

    Here 8 readers each combine 2 of 4 shared forms, so the true dictionary (4
    atoms) genuinely beats per-reader (8) and prediction (ii) becomes testable."""
    forms = orthonormal_forms(n_atoms, seed + 77)
    g = torch.Generator().manual_seed(seed + 5)
    codes = torch.zeros(n_readers, n_atoms)
    for u in range(n_readers):
        pick = torch.randperm(n_atoms, generator=g)[:per_reader]
        codes[u, pick] = (torch.randn(per_reader, generator=g).sign()
                          * (0.6 + 0.8 * torch.rand(per_reader, generator=g)))
    Qu = torch.einsum('ua,aij->uij', codes.to(DEV), torch.stack(forms))

    res = {'n_readers': n_readers, 'n_atoms': n_atoms, 'per_reader': per_reader,
           'codes': codes.tolist()}
    iu, ju = torch.triu_indices(D, D)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Qu)
    V = Qu[:, iu, ju] * sc
    U_, s_, Vh_ = torch.linalg.svd(V, full_matrices=False)
    res['pca_singular_values'] = [float(v) for v in s_[:8]]
    frontier = {}
    for k in range(1, min(n_readers, 8) + 1):
        rec = (U_[:, :k] * s_[:k]) @ Vh_[:k]
        Qh = torch.zeros_like(Qu)
        Qh[:, iu, ju] = rec / sc
        Qh[:, ju, iu] = rec / sc
        A, C, derr = fit_dictionary(Qu, k, steps=3000, lr=3e-2, l1=2e-2, seed=seed)
        cn = C.abs() / C.abs().max().clamp_min(1e-30)
        frontier[k] = {'pca_err': lam_relerr(Qu, Qh), 'dict_err': derr,
                       'atoms_per_reader': float((cn > 0.15).to(cn.dtype).sum(1).mean()),
                       'match': match_atoms(A, forms) if k >= n_atoms else None}
    res['frontier'] = frontier
    if verbose:
        print(f"  planted: {n_readers} readers, {n_atoms} shared forms, "
              f"{per_reader} of them per reader")
        for k, v in frontier.items():
            mm = (' | atom->planted cos '
                  + str([round(m['cos'], 3) for m in v['match']])) if v['match'] else ''
            print(f"    budget {k} atoms: PCA err {v['pca_err']:.4f} | dictionary err "
                  f"{v['dict_err']:.4f} | active atoms/reader {v['atoms_per_reader']:.2f}{mm}")
    return res


def main():
    t0 = time.time()
    out = {'config': {'d': D, 'mh': MH, 'h': H, 'readers': R, 'steps': STEPS,
                      'device': DEV}, 'registered': REGISTERED, 'runs': []}
    print('== ARM B: a structure where sharing is actually worth representing ==')
    out['bank'] = run_bank()

    print('\n== ARM A (the doc\'s design): shared form feeds all three readers ==')
    for seed in range(3):
        out['runs'].append(run(seed=seed))

    print('\n== gain sweep: does the top eigenvalue track reader multiplicity x gain? ==')
    out['gain_sweep'] = []
    for gains in ((1., 1., 1.), (1., 1., 0.), (1., .5, .25), (2., 2., 2.)):
        out['gain_sweep'].append(run(seed=0, gains=gains, tag=f'gains{gains}'))

    print('\n== private-strength sweep: when does the shared form stop leading? ==')
    out['priv_sweep'] = []
    for pv in ((0.5, 0.5, 0.5), (1., 1., 1.), (2., 2., 2.), (4., 4., 4.)):
        out['priv_sweep'].append(run(seed=0, priv=pv, tag=f'priv{pv}'))

    print('\n== NULL 1: random weights ==')
    out['null_random'] = []
    for seed in range(2):
        P = planted(seed)
        p = init_params(D, H, MH, seed=500 + seed, device=DEV)
        p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
        Qu = reader_forms(interaction(p), P['U'])
        ev = spectrum(reader_moment(Qu))
        iu, ju = torch.triu_indices(D, D)
        sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Qu)
        evec = torch.linalg.eigh(reader_moment(Qu))[1][:, -1]
        top = torch.zeros(D, D, device=DEV, dtype=Qu.dtype)
        top[iu, ju] = evec / sc
        top[ju, iu] = evec / sc
        o = {'seed': seed, 'eig_ratio_top_to_next': float(ev[0] / ev[1].clamp_min(1e-30)),
             'top_eigvec_cos_shared': abs(lam_cos(top[None], P['S0'][None])),
             'participation_ratio': participation_ratio(ev)}
        print(f"  [random {seed}] top/next {o['eig_ratio_top_to_next']:.2f} | "
              f"top eigvec vs shared {o['top_eigvec_cos_shared']:.4f} | "
              f"participation {o['participation_ratio']:.2f}")
        out['null_random'].append(o)

    print('\n== NULL 2: gauge scramble ==')
    out['null_gauge'] = []
    P = planted(0)
    p = init_params(D, H, MH, seed=20, device=DEV)
    p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
    gen = data_fn(P)
    for k in p:
        p[k].requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for s in range(STEPS):
        x, y = gen()
        loss = (((forward(p, x) @ P['U'].T) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    for k in p:
        p[k].requires_grad_(False)
    Qu = reader_forms(interaction(p), P['U'])
    pg, resid = gauge_refactor(p, seed=31)
    Qug = reader_forms(interaction(pg), P['U'])
    ev, evg = spectrum(reader_moment(Qu)), spectrum(reader_moment(Qug))
    o = {'refactor_residual': resid, 'hidden_before': H, 'hidden_after': int(pg['L'].shape[0]),
         'eig_ratio_before': float(ev[0] / ev[1].clamp_min(1e-30)),
         'eig_ratio_after': float(evg[0] / evg[1].clamp_min(1e-30))}
    print(f"  residual {resid:.1e} | hidden {H} -> {o['hidden_after']} | "
          f"top/next {o['eig_ratio_before']:.3f} -> {o['eig_ratio_after']:.3f}")
    out['null_gauge'] = o

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a5_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
