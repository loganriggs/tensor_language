"""Joint small-problem red-team of the frozen spectral initialization miss."""
import json
import time
import hashlib
import itertools
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from shared_reader_conditional_v1 import partner_update, group_cp, residual_cp, dense_cp
from shared_reader_partner_rcg_v1 import optimize


def main():
    start=time.monotonic();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    torch.manual_seed(1401); d,o,r=9,7,2
    # Replay exactly the RNG draws for the earlier one-group problem and its start.
    torch.randn(d);torch.randn(o,r);torch.randn(d,r);torch.randn(d,1)
    planted=[]
    for _ in range(2):
        a=torch.randn(d);a/=a.norm();planted.append((a,torch.randn(o,r),torch.randn(d,r)))
    target=tuple(torch.cat([group_cp(*g)[k] for g in planted],dim=1 if k==2 else 0) for k in range(3))
    truth=dense_cp(*target); total=truth.square().sum()
    spectrum=torch.linalg.eigh(torch.einsum('oij,ojk->ik',truth,truth)).eigenvectors.flip(1)
    groups=[(spectrum[:,j],torch.zeros(o,r),torch.zeros(d,r)) for j in range(2)]
    for _ in range(200):
        for j in range(2):
            residual=residual_cp(target,groups,j)
            a,_=optimize(groups[j][0],*residual,torch.eye(o),total,r,max_steps=30,tolerance=1e-9)
            u,v=partner_update(a,*residual,r);groups[j]=(a,u,v)
    initial_error=float((truth-sum(dense_cp(*group_cp(*g)) for g in groups)).square().sum()/total)
    parent=json.loads(Path(__file__).with_name('SHARED_READER_INITIALIZATION_V1_AUDIT.json').read_text())
    expected=next(row['final_error'] for row in parent['rows'] if row['groups']==2 and row['initialization']=='spectral' and row['method']=='reduced_rcg')
    replay=abs(initial_error-expected)
    assert replay<=1e-10
    initial_groups=groups
    shapes=[(d,),(o,r),(d,r)]*2
    sizes=[int(np.prod(shape)) for shape in shapes]
    def unpack(flat):
        tensors=[x.reshape(shape) for x,shape in zip(flat.split(sizes),shapes)]
        return [tuple(tensors[j:j+3]) for j in [0,3]]
    initial=torch.cat([x.flatten() for g in groups for x in g])
    def residual_tensor(flat):
        fitted=sum(dense_cp(*group_cp(*g)) for g in unpack(flat))
        return ((fitted-truth)/total.sqrt()).flatten()
    def fun(x):return residual_tensor(torch.from_numpy(x)).detach().numpy()
    def jac(x):return torch.autograd.functional.jacobian(residual_tensor,torch.from_numpy(x),vectorize=True).detach().numpy()
    fit=least_squares(fun,initial.numpy(),jac=jac,method='trf',max_nfev=1000,
                      ftol=1e-12,xtol=1e-12,gtol=1e-12)
    final=unpack(torch.from_numpy(fit.x));error=float(np.sum(fit.fun**2))
    actual=[dense_cp(*group_cp(*g)).flatten() for g in final]
    native=[dense_cp(*group_cp(*g)).flatten() for g in planted]
    best=max(min(float(actual[j]@native[p[j]]/(actual[j].norm()*native[p[j]].norm()))
                 for j in range(2)) for p in itertools.permutations(range(2)))
    cache=Path('/dev/shm/bilin18_shared_reader_joint_polish_v1.pt')
    torch.save(dict(target=target,planted=planted,initial=initial_groups,final=final),cache)
    result=dict(predictions=dict(pred_a_parent_replay=replay<=1e-10,pred_b_joint_recovery=error<=1e-8,
                                 pred_c_group_identification=best>=.99),
                initial_error=initial_error,parent_replay_error=replay,final_error=error,
                matched_minimum_function_cosine=best,evaluations=fit.nfev,jacobian_evaluations=fit.njev,
                optimality=fit.optimality,termination=fit.message,seconds=time.monotonic()-start,
                cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
                scope='Exact small dense joint Jacobian/trust-region polish of a reproduced stalled toy. Not a scalable native implementation or global identifiability certificate.')
    Path(__file__).with_name('SHARED_READER_JOINT_POLISH_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
