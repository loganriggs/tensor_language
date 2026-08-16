"""A2 follow-ups: the controls the two strongest A2 observations need.

(1) "Projecting the weights onto the Fourier blocks gives test acc 1.000 from
    step 2500, when the model itself is at 0.006." A projection onto a 44-dim
    subspace of a 1081-dim space is a strong operation, so it needs matched
    controls: the same projection shape onto random subspaces, and onto the
    frequency blocks of a RELABELLED group (Fourier-like, but not this group's).
(2) "Ablating a frequency block zeroes that frequency's logit variance but
    leaves accuracy at 1.000." Accuracy is saturated by redundancy across 11
    frequencies — the informative version is a dose-response curve.
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
from bq_common import interaction, forward_Q

torch.set_default_dtype(torch.float64)
P, DEV = m.P, m.DEV


def rand_block_basis(seed, n_blocks=11, dim=4):
    """n_blocks mutually orthogonal random `dim`-dim subspaces of R^{2p}."""
    g = torch.Generator().manual_seed(seed)
    Qm = torch.linalg.qr(torch.randn(2 * P, n_blocks * dim, generator=g).to(DEV))[0]
    return [Qm[:, i * dim:(i + 1) * dim] for i in range(n_blocks)]


def relabelled_fourier(seed):
    """The Fourier blocks of a randomly relabelled group: same 4-dim shapes, same
    'Fourier-like' character, but not aligned with THIS task's group structure."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(P, generator=g).to(DEV)
    out = []
    for w in range(1, P // 2 + 1):
        B = m.FBLOCKS[w].clone()
        Bp = torch.zeros_like(B)
        Bp[:P] = B[:P][perm]
        Bp[P:] = B[P:][perm]
        out.append(torch.linalg.qr(Bp)[0])
    return out


def project(Q, subs):
    out = torch.zeros_like(Q)
    for B in subs:
        Pb = B @ B.T
        out = out + torch.einsum('ij,mjk,kl->mil', Pb, Q, Pb)
    return out


@torch.no_grad()
def sc(Q, x, y, idx):
    o = forward_Q(Q, x)
    o = o - o.mean(1, keepdim=True)
    return (float((o[idx].argmax(1) == y[idx]).to(Q.dtype).mean()),
            float(F.cross_entropy(o[idx], y[idx])))


def main():
    x, y, a, b = m.all_pairs()
    basis = m.identifiable_basis(x)
    cache = torch.load('a2_cache.pt', weights_only=False)
    tr, te = m.split(0)
    out = {}
    FB = [m.FBLOCKS[w] for w in range(1, P // 2 + 1)]

    print('== (1) dynamics with matched projection controls (seed 0, every 500 steps) ==')
    rows = []
    ctrl_rand = [rand_block_basis(s) for s in range(3)]
    ctrl_relab = [relabelled_fourier(s) for s in range(3)]
    for rec, ck in zip(cache[0]['hist'], cache[0]['ck']):
        Q = interaction(ck)
        Qid, _ = m.canonicalise(Q, basis)
        acc_f, ce_f = sc(project(Qid, FB), x, y, te)
        ar = [sc(project(Qid, c), x, y, te)[0] for c in ctrl_rand]
        al = [sc(project(Qid, c), x, y, te)[0] for c in ctrl_relab]
        rows.append({'step': rec['step'], 'test_acc': rec['test_acc'],
                     'train_acc': rec['train_acc'],
                     'off_block': m.fourier_power(Qid)['off_block'],
                     'fourier_proj_test_acc': acc_f, 'fourier_proj_test_ce': ce_f,
                     'ctrl_random_blocks_acc': sum(ar) / len(ar),
                     'ctrl_relabelled_fourier_acc': sum(al) / len(al)})
    out['dynamics_controlled'] = rows
    first = next((r for r in rows if r['fourier_proj_test_acc'] >= 0.999), None)
    first_model = next((r for r in rows if r['test_acc'] >= 0.999), None)
    for r in rows[:14] + rows[14::8]:
        print(f"  step {r['step']:6d} model test {r['test_acc']:.3f} | Fourier-proj "
              f"{r['fourier_proj_test_acc']:.3f} (CE {r['fourier_proj_test_ce']:.3f}) | ctrl random "
              f"{r['ctrl_random_blocks_acc']:.3f} | ctrl relabelled-Fourier "
              f"{r['ctrl_relabelled_fourier_acc']:.3f}")
    out['first_step_projection_generalises'] = first['step'] if first else None
    out['first_step_model_generalises'] = first_model['step'] if first_model else None
    print(f"  --> projection reaches test 1.000 at step {out['first_step_projection_generalises']}; "
          f"the model itself at step {out['first_step_model_generalises']}")

    print('\n== (2) ablation dose-response (seed 0, block-projected model) ==')
    Q0 = interaction(cache[0]['p'])
    Qid0, _ = m.canonicalise(Q0, basis)
    Qb = project(Qid0, FB)
    dose = []
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(11, generator=g).tolist()
    for k in range(0, 12):
        keep = [FB[i] for i in order[k:]]
        acc, ce = sc(project(Qb, keep), x, y, te) if keep else (0.0, float('nan'))
        # control: remove k random 4-dim subspaces of the same total dimension
        accs, ces = [], []
        for s in range(3):
            rb = rand_block_basis(100 + s, n_blocks=11)
            Pk = torch.eye(2 * P, device=DEV, dtype=Qb.dtype)
            for i in range(k):
                Pk = Pk - rb[i] @ rb[i].T
            ac, cc = sc(torch.einsum('ij,mjk,kl->mil', Pk, Qb, Pk), x, y, te)
            accs.append(ac)
            ces.append(cc)
        dose.append({'n_removed': k, 'acc': acc, 'ce': ce,
                     'ctrl_acc': sum(accs) / 3, 'ctrl_ce': sum(ces) / 3})
        print(f"  removed {k:2d} frequency blocks: test acc {acc:.4f} CE {ce:.4f} | "
              f"random 4-dim subspaces of equal dimension: acc {dose[-1]['ctrl_acc']:.4f} "
              f"CE {dose[-1]['ctrl_ce']:.4f}")
    out['ablation_dose'] = dose

    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_extra_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
