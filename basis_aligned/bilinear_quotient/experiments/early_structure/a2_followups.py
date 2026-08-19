"""Reproducibility script for A2's follow-up numbers.

Reviewer 2's finding 4: several of A2's headline numbers existed only in
RESULTS.md, produced by ad-hoc inline commands. The reviewer verified all of them
by hand and found them correct, but "correct and unreproducible" is not good
enough, and REVIEW_RESPONSE.md then claimed they were "now in a2_followups.py"
before that file existed. This is that file.

It regenerates, from the committed checkpoint cache:
  1. A2-4's residual-only table  — is the deleted part a lookup table?
  2. A2-4's corrected dynamics   — the two logit scales and the alignment to final,
                                    which is what retracted the "decay" claim.
  3. A2-8's nulls                — symmetrisation on random and task-shuffled models.
  4. A2-8's symmetry split       — symmetry-preserving vs symmetry-breaking halves
                                    of the residual, rescaled to equal off-block mass.
  5. A2-7's long run             — does crystallisation actually arrest past 40k?
                                    (the columns withdrawn from RESULTS.md pending this)
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a2_modular as m
from a2_jade import frontier
from bq_common import interaction, forward_Q, lam_cos, init_params

torch.set_default_dtype(torch.float64)
P = m.P
LONG_STEPS = 260000


def swap_op(dev, dt):
    S = torch.zeros(2 * P, 2 * P, device=dev, dtype=dt)
    S[:P, P:] = torch.eye(P, device=dev, dtype=dt)
    S[P:, :P] = torch.eye(P, device=dev, dtype=dt)
    return S


def main():
    t0 = time.time()
    x, y, a, b = m.all_pairs()
    basis = m.identifiable_basis(x)
    cache = torch.load('a2_cache.pt', weights_only=False)
    S = swap_op(x.device, x.dtype)
    allw = list(range(0, P // 2 + 1))
    out = {}

    print('== 1. A2-4: what is the deleted part, on its own? ==')
    out['residual_only'] = []
    for seed in range(3):
        Qid, _ = m.canonicalise(interaction(cache[seed]['p']), basis)
        Qb = m.block_project(Qid, allw)
        Res = Qid - Qb
        tr, te = m.split(seed)
        o = forward_Q(Res, x)
        o = o - o.mean(1, keepdim=True)
        rec = {'seed': seed,
               'train_acc': float((o[tr].argmax(1) == y[tr]).double().mean()),
               'test_acc': float((o[te].argmax(1) == y[te]).double().mean()),
               'correct_logit_boost_train': float(o[tr, y[tr]].mean()),
               'correct_logit_boost_test': float(o[te, y[te]].mean())}
        out['residual_only'].append(rec)
        print(f"  seed {seed}: train {rec['train_acc']:.4f} test {rec['test_acc']:.4f} | "
              f"correct-class logit boost train {rec['correct_logit_boost_train']:+.3f} "
              f"test {rec['correct_logit_boost_test']:+.3f}")

    print('\n== 2. A2-4 corrected dynamics: which term is actually moving? ==')
    Qfin = m.block_project(m.canonicalise(interaction(cache[0]['p']), basis)[0], allw)
    dyn = []
    for rec, ck in list(zip(cache[0]['hist'], cache[0]['ck']))[::6]:
        Q = interaction(ck)
        Qid, _ = m.canonicalise(Q, basis)
        Qb = m.block_project(Qid, allw)
        Res = Qid - Qb

        def rms(QQ):
            o = forward_Q(QQ, x)
            return float((o - o.mean(1, keepdim=True)).pow(2).mean().sqrt())

        d = {'step': rec['step'], 'test_acc': rec['test_acc'],
             'circuit_logit_rms': rms(Qb), 'residual_logit_rms': rms(Res),
             'off_block': m.fourier_power(Qid)['off_block'],
             'functional_residual': m.functional_residual(Qid, Qb, x),
             'cos_circuit_to_final': lam_cos(Qb, Qfin)}
        dyn.append(d)
        print(f"  step {d['step']:6d} test {d['test_acc']:.3f} | circuit rms "
              f"{d['circuit_logit_rms']:.3f} | residual rms {d['residual_logit_rms']:.3f} | "
              f"off-block {d['off_block']:.3f} | fn residual {d['functional_residual']:.3f} | "
              f"cos->final {d['cos_circuit_to_final']:.3f}")
    out['dynamics_corrected'] = dyn

    print('\n== 3. A2-8 nulls: does symmetrising help a model with no circuit? ==')
    def sym(Q):
        return 0.5 * (Q + torch.einsum('ij,mjk,kl->mil', S, Q, S))

    def acc(Q, idx):
        o = forward_Q(Q, x)
        o = o - o.mean(1, keepdim=True)
        return float((o[idx].argmax(1) == y[idx]).double().mean())

    tr0, te0 = m.split(0)
    out['symmetry_nulls'] = []
    pr = init_params(2 * P, m.H, P, seed=1000, device=m.DEV)
    pr = {k: v.to(torch.get_default_dtype()) for k, v in pr.items()}
    Qr, _ = m.canonicalise(interaction(pr), basis)
    ps, hist, _ = m.train_model(0, shuffle_labels=True, log=False)
    Qs_, _ = m.canonicalise(interaction(ps), basis)
    for tag, Q in (('random weights', Qr), ('task-shuffled', Qs_)):
        eq = float((sym(Q) ** 2).sum() / (Q ** 2).sum())
        rec = {'model': tag, 'equivariant_fraction': eq,
               'test_before': acc(Q, te0), 'test_after_symmetrising': acc(sym(Q), te0)}
        r = frontier(sym(Q), f'sym-{tag}', verbose=False)
        rec['jade_after_symmetrising'] = r['best_n_freq_full']
        out['symmetry_nulls'].append(rec)
        print(f"  {tag:15s}: equivariant fraction {eq:.4f} | test "
              f"{rec['test_before']:.4f} -> {rec['test_after_symmetrising']:.4f} | "
              f"JADE after symmetrising {rec['jade_after_symmetrising']}/11")
    for seed in range(3):
        Qid, _ = m.canonicalise(interaction(cache[seed]['p']), basis)
        print(f"  trained seed {seed}: equivariant fraction "
              f"{float((sym(Qid)**2).sum()/(Qid**2).sum()):.4f}  (for comparison)")

    print('\n== 4. A2-8: which half of the residual is the hard half? ==')
    Qid, _ = m.canonicalise(interaction(cache[0]['p']), basis)
    Qb = m.block_project(Qid, allw)
    Res = Qid - Qb
    Rsym = 0.5 * (Res + torch.einsum('ij,mjk,kl->mil', S, Res, S))
    Ranti = Res - Rsym
    target = m.fourier_power(Qid)['off_block']
    out['residual_split'] = {'symmetric_share': float((Rsym ** 2).sum() / (Res ** 2).sum()),
                             'antisymmetric_share': float((Ranti ** 2).sum() / (Res ** 2).sum()),
                             'target_off_block': target, 'rows': []}
    print(f"  residual is {out['residual_split']['symmetric_share']:.3f} symmetry-preserving "
          f"and {out['residual_split']['antisymmetric_share']:.3f} symmetry-breaking")
    for tag, R in (('symmetry-preserving half', Rsym), ('symmetry-breaking half', Ranti),
                   ('both (the real residual)', Res)):
        lo, hi = 0.0, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if m.fourier_power(Qb + mid * R)['off_block'] < target:
                lo = mid
            else:
                hi = mid
        Qa = Qb + lo * R
        r = frontier(Qa, tag, verbose=False)
        row = {'half': tag, 'off_block': m.fourier_power(Qa)['off_block'],
               'freqs_recovered': r['best_n_freq_full'], 'eps': r['best_tol']}
        out['residual_split']['rows'].append(row)
        print(f"    {tag:26s} rescaled to off-block {row['off_block']:.4f}: "
              f"{row['freqs_recovered']}/11")

    print(f'\n== 5. A2-7: does crystallisation arrest? (long run to {LONG_STEPS} steps) ==')
    xa, ya, _, _ = m.all_pairs()
    tr, te = m.split(0)
    p = init_params(2 * P, m.H, P, seed=0, device=m.DEV)
    p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
    for k in p:
        p[k].requires_grad_(True)
    opt = torch.optim.AdamW(list(p.values()), lr=m.LR, weight_decay=m.WD)
    long = []
    for step in range(LONG_STEPS + 1):
        loss = F.cross_entropy(m.forward(p, xa[tr]), ya[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20000 == 0:
            with torch.no_grad():
                Q = interaction({k: v.detach() for k, v in p.items()})
                Qi, _ = m.canonicalise(Q, basis)
                Qbb = m.block_project(Qi, allw)
                o = m.forward(p, xa)
                rec = {'step': step,
                       'test_acc': float((o[te].argmax(1) == ya[te]).double().mean()),
                       'off_block_raw': m.fourier_power(Q)['off_block'],
                       'off_block_canonical': m.fourier_power(Qi)['off_block'],
                       'functional_residual': m.functional_residual(Qi, Qbb, xa)}
                long.append(rec)
                print(f"  step {step:7d} test {rec['test_acc']:.4f} | off-block raw "
                      f"{rec['off_block_raw']:.4f} canonical {rec['off_block_canonical']:.4f} "
                      f"| fn residual {rec['functional_residual']:.4f}", flush=True)
    out['long_run'] = long
    tail = [r['off_block_canonical'] for r in long if r['step'] >= 100000]
    out['arrests'] = bool(max(tail) - min(tail) < 0.02) if len(tail) > 1 else None
    print(f"  canonical off-block over the last {len(tail)} checkpoints: "
          f"{min(tail):.4f} to {max(tail):.4f}  -> "
          f"{'ARRESTS' if out['arrests'] else 'still moving'}")

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a2_followups_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
