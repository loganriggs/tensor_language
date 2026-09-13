"""Bound downstream squared-error distortion for the existing Gaussian marginal.

Same 256 queries/seed/current-value/zero-displacement scope as the original
shared-key/value weight diagnostic. Predict generalized condition <=1.2.
No normalization, first-layer values, retained gates, text or fitting.
"""
import json
from pathlib import Path
import torch
from shared_key_value_moment_v1 import moment
from head17_source_interface_v1 import CHECKPOINT


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7131206)
    sd = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    def head(name):
        return sd[f'transformer.h.17.attn.{name}.weight'].reshape(9,128,1152)[2].double()
    q = torch.randn(256,1152,dtype=torch.float64)
    a = (q @ head('c_q').T) @ head('c_k')
    b = (q @ head('c_q2').T) @ head('c_k2')
    f = (1-float(sd['transformer.h.17.attn.lamb'])) * head('c_v')
    c = a.square().sum(-1)*b.square().sum(-1)+2*(a*b).sum(-1).square()
    h0 = c.mean()*(f@f.T)
    h = moment(f,a,b).mean(0)
    chol = torch.linalg.cholesky(h0)
    x = torch.linalg.solve_triangular(chol,h,upper=False)
    white = torch.linalg.solve_triangular(chol,x.T,upper=False).T
    ev, vectors = torch.linalg.eigh((white+white.T)/2)
    low, high = float(ev[0]), float(ev[-1])
    # Recover witnesses to the extremal Rayleigh quotients in original coordinates.
    e = torch.linalg.solve_triangular(chol.T,vectors[:,[0,-1]],upper=True).T
    quotients = ((e@h)*e).sum(-1)/((e@h0)*e).sum(-1)
    replay = float((quotients-torch.tensor([low,high],dtype=torch.float64)).abs().max())
    assert low>0 and replay<1e-10
    result = dict(pred_a=high/low<=1.2, lambda_min=low, lambda_max=high,
                  squared_objective_distortion=high/low,
                  error_norm_factor=(high/low)**.5,
                  extremal_witness_replay_error=replay,
                  scope=__doc__,
                  consequence='For every downstream linear error map E, low*tr(E H0 E.T) <= tr(E H E.T) <= high*tr(E H0 E.T). Within the same feasible class a GLOBAL H0 optimizer is within high/low of the optimal H squared error. No guarantee about a merely local optimizer or native behavior.')
    out=Path(__file__).with_name('SHARED_KEY_VALUE_METRIC_BOUND_V1_RESULT.json')
    with out.open('x') as handle:
        json.dump(result,handle,indent=2)
        handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
