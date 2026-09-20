"""Exact fourth-moment quadrature checks the trace-aware pruning metric."""
import itertools
import json
from pathlib import Path
import torch
from joint_atom_pruning import atom_gram


def main():
    torch.manual_seed(633)
    C=torch.randn(3,7,dtype=torch.float64)
    A=torch.randn(7,4,dtype=torch.float64)
    B=torch.randn(7,4,dtype=torch.float64)
    K=atom_gram(C,A,B)
    trace=(A*B).sum(-1)
    analytic=2*K+(C.T@C)*torch.outer(trace,trace)
    # Independent 3-point Gaussian quadrature, exact for degree <=5 per axis.
    nodes=torch.tensor([-3**.5,0.,3**.5],dtype=torch.float64)
    weights=torch.tensor([1/6,2/3,1/6],dtype=torch.float64)
    indices=torch.tensor(list(itertools.product(range(3),repeat=4)))
    x=nodes[indices];mass=weights[indices].prod(-1)
    products=(x@A.T)*(x@B.T)
    evaluated=(C.T@C)*(products.T@(mass[:,None]*products))
    error=float((analytic-evaluated).norm()/evaluated.norm())
    assert error<1e-12
    # Tripwire: omitting trace must fail this nonzero-trace control.
    omitted=float((2*K-evaluated).norm()/evaluated.norm())
    assert omitted>.01
    result=dict(relative_quadrature_error=error,trace_omission_error=omitted,
                quadrature_points=len(x),scope='exact Gaussian fourth-moment control; not native distribution')
    Path(__file__).with_name('ISOTROPIC_ATOM_METRIC_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
