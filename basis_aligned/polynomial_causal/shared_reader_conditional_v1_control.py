"""Dense independent controls and bounded planted overlapping-group recovery."""
import json
import time
from pathlib import Path
import torch
from shared_reader_conditional_v1 import (
    partner_update, reader_update, reader_force, sphere_psd_lowrank,
    group_cp, residual_cp, dense_cp,
)


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1327)
    d, outputs, rank = 9, 7, 2
    target = (torch.randn(11, d), torch.randn(11, d), torch.randn(outputs, 11))
    dense = dense_cp(*target)
    u, v = torch.randn(outputs, rank), torch.randn(d, rank)
    m = u @ v.T
    force = reader_force(*target, u, v)
    dense_force = torch.einsum('oik,ok->i', dense, m)
    force_error = float((force-dense_force).norm()/dense_force.norm())
    cases = []
    q = m.T @ m
    # Generic, zero force, and exact-range force with a subunit pseudoinverse.
    base = v @ torch.randn(rank)
    base *= .3/base.norm()
    for name, rhs in [('generic', 2*force), ('zero', torch.zeros(d)), ('hard', q@base)]:
        a, diagnostics = sphere_psd_lowrank(u, v, rhs)
        z = torch.randn(200, d)
        z /= z.norm(dim=1, keepdim=True)
        fz = .5*torch.einsum('ni,ij,nj->n', z, q, z)-z@rhs
        fa = .5*a@q@a-a@rhs
        delta = z-a
        certificate = .5*torch.einsum('ni,ij,nj->n', delta,
                                     q+diagnostics['lagrange']*torch.eye(d), delta)
        replay = float((fz-fa-certificate).abs().max()/(q.norm()+rhs.norm()))
        cases.append(dict(name=name, **diagnostics, certificate_error=replay,
                          minimum_sample_objective_gap=float((fz-fa).min())))
    variable = torch.randn(d)
    variable /= variable.norm()
    variable.requires_grad_()
    prediction = dense_cp(*group_cp(variable, u, v))
    # Dense unconstrained gradient includes the ||a||^2 term; remove its radial part.
    gradient = torch.autograd.grad((dense-prediction).square().sum(), variable)[0]
    analytical = q@variable-2*force
    tangent_dense = gradient-variable*(variable@gradient)
    tangent_formula = analytical-variable*(variable@analytical)
    gradient_error = float((tangent_dense-tangent_formula).norm()/tangent_dense.norm())
    recoveries = []
    for count in [1, 2]:
        planted = []
        for j in range(count):
            a = torch.randn(d); a /= a.norm()
            planted.append((a, torch.randn(outputs, rank), torch.randn(d, rank)))
        native = tuple(torch.cat([group_cp(*g)[k] for g in planted], dim=1 if k==2 else 0)
                       for k in range(3))
        truth = dense_cp(*native)
        total = truth.square().sum()
        for initialization in ['near', 'random']:
            groups = []
            for a, u, v in planted:
                guess = a+.05*torch.randn(d) if initialization=='near' else torch.randn(d)
                guess /= guess.norm()
                groups.append((guess, u.clone() if initialization=='near' else torch.zeros_like(u), v.clone()))
            def loss():
                return float((truth-sum(dense_cp(*group_cp(*g)) for g in groups)).square().sum()/total)
            initial = previous = loss()
            max_increase = max_kkt = 0.
            history = []
            for sweep in range(500):
                for j in range(count):
                    residual = residual_cp(native, groups, j)
                    a, _, _ = groups[j]
                    u, v = partner_update(a, *residual, rank)
                    groups[j] = (a, u, v)
                    current = loss(); max_increase = max(max_increase, current-previous); previous=current
                    a, diag = reader_update(*residual, u, v)
                    max_kkt=max(max_kkt, diag['kkt_relative_residual'])
                    groups[j]=(a, u, v)
                    current = loss(); max_increase=max(max_increase, current-previous); previous=current
                if sweep%25==0: history.append(dict(sweep=sweep, relative_error=current))
                if current<=1e-12: break
            recoveries.append(dict(groups=count, initialization=initialization, sweeps=sweep+1,
                                   initial_error=initial, final_error=current, maximum_step_increase=max_increase,
                                   maximum_kkt_error=max_kkt, history=history))
    result = dict(predictions={
        'pred_a_conditional_instrument': max(force_error, gradient_error,
            *[max(c['kkt_relative_residual'], c['certificate_error'], c['unit_error']) for c in cases]) <= 1e-9,
        'pred_b_monotonic_updates': max(r['maximum_step_increase'] for r in recoveries)<=1e-10,
        'pred_c_near_planted_recovery': all(r['final_error']<=1e-8 for r in recoveries if r['initialization']=='near')},
        force_error=force_error, tangent_gradient_error=gradient_error, sphere_cases=cases,
        recovery=recoveries, seconds=time.monotonic()-started,
        scope='Small planted recovery and exact conditional updates. Random starts descriptive. No native multi-group fit, joint/global certificate, or behavioral circuit claim.')
    Path(__file__).with_name('SHARED_READER_CONDITIONAL_V1_CONTROL.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['predictions']['pred_a_conditional_instrument']
    assert result['predictions']['pred_b_monotonic_updates']


if __name__=='__main__':
    main()
