"""Actual-weight control for a conditional, reusable intervention-product basis."""
from pathlib import Path
import json
import torch
from head17_source_interface_v1 import CHECKPOINT
from directional_mlp_response_context_v1 import prepare, evaluate

P = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    torch.manual_seed(348)
    sd = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
    program = torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt', weights_only=True)
    z = torch.randn(4, 1152, dtype=torch.float64)
    L, R, D = [sd['transformer.h.9.mlp.'+k+'.weight'].double()
               for k in ('Left', 'Right', 'Down')]
    rho0 = z.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps
    base = ((z@L.T)*(z@R.T))@D.T/rho0
    ctx = prepare(z, base, program)
    lam = float(sd['transformer.h.10.lambdas'][0])
    basis = torch.stack([ctx['direction'].expand_as(z), base, ctx['Jz'],
                         ctx['Jw'].expand_as(z)], dim=1)*lam
    L, R, D = [sd['transformer.h.10.mlp.'+k+'.weight'].double()
               for k in ('Left', 'Right', 'Down')]

    def cross(x, y):
        return ((x@L.T)*(y@R.T)+(y@L.T)*(x@R.T))@D.T

    pairs = [(i, j) for i in range(4) for j in range(i, 4)]
    products = torch.stack([cross(basis[:, i], basis[:, j])
                            for i, j in pairs], dim=1)

    def coefficients(a):
        a = torch.full((4, 1), a, dtype=torch.float64)
        rho = ctx['perpendicular_rms'] + ctx['writer_rms']*(a-ctx['parallel']).square()
        return torch.cat([-a, a*(2*ctx['cross_rms']-a*ctx['writer_rms'])/rho,
                          -a/rho, a.square()/(2*rho)], dim=-1)

    numerator_error = numerator_ref = response_error = response_ref = 0.0
    zero_error = 0.0
    strengths = [-1., -.5, -.2, -.05, 0., .03, .1, .25, .5, .75, 1.]
    for a in strengths:
        u = coefficients(a)
        x = lam*evaluate(torch.full((4, 1), a, dtype=torch.float64), ctx)
        response_error += float((torch.einsum('bi,bid->bd', u, basis)-x).square().sum())
        response_ref += float(x.square().sum())
        for b in strengths:
            v = coefficients(b)
            weights = torch.stack([u[:, i]*v[:, j] if i == j else
                                   u[:, i]*v[:, j]+u[:, j]*v[:, i]
                                   for i, j in pairs], dim=1)
            actual = torch.einsum('bi,bid->bd', weights, products)
            y = lam*evaluate(torch.full((4, 1), b, dtype=torch.float64), ctx)
            expected = cross(x, y)
            numerator_error += float((actual-expected).square().sum())
            numerator_ref += float(expected.square().sum())
            if a == 0 or b == 0:
                zero_error = max(zero_error, float(actual.abs().max()))
    result = dict(response_relative_error=(response_error/response_ref)**.5,
                  product_relative_error=(numerator_error/numerator_ref)**.5,
                  zero_edit_maxabs=zero_error, contexts=4, strength_pairs=121,
                  basis_vectors=4, symmetric_products=10,
                  product_storage_scalars_per_context=10*1152,
                  scope='Actual weights; synthetic backgrounds; conditional residual/residual numerator only. '
                  'A supplied common joint denominator can divide both sides. Attention partners, '
                  'background generation and full native behavioral validation remain external. '
                  'Basis depends on context; ten is an upper bound, not proven minimal rank. '
                  'Potential amortization for many strengths, no measured runtime speedup.')
    assert result['response_relative_error'] < 1e-12
    assert result['product_relative_error'] < 1e-12 and zero_error == 0
    (P/'RESPONSE_PRODUCT_BASIS_V1_CONTROL.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
