"""FP32 search matvecs; authoritative metadata and verification remain FP64."""
import json
from pathlib import Path
import torch
from congruence_block_operator_v1 import Operator
from congruence_eigensolver_v1 import solve


class MixedOperator:
    def __init__(self,reference):
        self.reference=reference
        self.second=reference.second
        self.total=reference.total
        gram=reference.output_gram.float()
        self.search=Operator(reference.left.float(),reference.right.float(),(gram+gram.T)/2)
        self.search.second=(self.search.second+self.search.second.T)/2

    def __call__(self,z):
        return self.search(z.float()).double()


def verify(reference,values,vectors):
    d=reference.second.shape[0]
    identity=torch.eye(d,dtype=reference.second.dtype,device=reference.second.device)
    bound=4*float(torch.linalg.eigvalsh((reference.second+reference.second.T)/2)[-1])/float(reference.total)
    records=[]
    for value,vector in zip(values,vectors):
        z=vector.to(reference.second.device)
        action=reference(z)/reference.total
        rayleigh=float((z*action).sum()/z.square().sum())
        records.append(dict(search_eigenvalue=float(value),fp64_rayleigh=rayleigh,
            fp64_residual_over_bound=float((action-rayleigh*z).norm()/z.norm()/bound),
            search_value_difference_over_bound=abs(rayleigh-float(value))/bound,
            identity_overlap=float(z.trace().abs()/d**.5/z.norm())))
    return records


def control():
    torch.manual_seed(983);torch.set_default_dtype(torch.float64)
    l,r,w=torch.randn(23,9),torch.randn(23,9),torch.randn(7,23)
    reference=Operator(l,r,w.T@w);mixed=MixedOperator(reference)
    z=torch.randn(9,9)
    action_error=float((mixed(z)-reference(z)).norm()/reference(z).norm())
    result,vectors=solve(mixed,k=4,ncv=32,maxiter=300,tol=1e-5,seed=985)
    checked=verify(reference,result['eigenvalues'],vectors)
    passed=result['converged'] and action_error<1e-5 and max(x['fp64_residual_over_bound'] for x in checked)<1e-6
    receipt=dict(instrument_passed=passed,fp32_action_relative_error=action_error,
                 fp64_verification=checked,search_converged=result['converged'],
                 scope='Synthetic precision/verification control only. Native final convergence bar remains1e-7 of operator bound.')
    Path(__file__).with_name('MIXED_PRECISION_CONGRUENCE_V1_CONTROL.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert passed,receipt
    return receipt


if __name__=='__main__':print(json.dumps(control(),indent=2))
