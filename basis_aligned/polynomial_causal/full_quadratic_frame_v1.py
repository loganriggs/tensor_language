"""All quadratic interactions in small overlapping input subspaces.

Learn only Grassmann input subspaces; conditional output writers solved exactly.
Complete orthonormal symmetric cores give a block-Frobenius penalty, not the
old four-output block's squared-nuclear penalty after gauge minimization.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.optimizers import ConjugateGradient
from orthogonal_multioutput_pymanopt_v2 import View, evaluate
from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective
from joint_quadratic_fit_v1 import product_cross


def cores_for(bank):
    groups, _, rank = bank.shape
    count = rank*(rank+1)//2
    return torch.eye(count,device=bank.device,dtype=bank.dtype).repeat(1,groups)


def fit(objective, initial, seconds=120, tolerance=1e-8):
    device=initial.device
    groups,dim,rank=initial.shape
    manifold=Grassmann(dim,rank,k=groups)
    cores=cores_for(initial)
    count=rank*(rank+1)//2
    # Pymanopt drops the leading dimension for k=1.
    def tensor(x):
        return torch.from_numpy(np.array(x,copy=True)).to(device).reshape(groups,dim,rank)
    @pymanopt.function.numpy(manifold)
    def cost(x):return evaluate(objective,tensor(x),cores,count)[0]
    @pymanopt.function.numpy(manifold)
    def gradient(x):
        return evaluate(objective,tensor(x),cores,count)[1][0].cpu().numpy().reshape(x.shape)
    point=initial.cpu().numpy()
    if groups==1:point=point[0]
    result=ConjugateGradient(max_iterations=10000,max_time=seconds,
        min_gradient_norm=tolerance,verbosity=0).run(
            Problem(manifold,cost,euclidean_gradient=gradient),initial_point=point)
    final=tensor(result.point)
    loss,grad,details=evaluate(objective,final,cores,count)
    tangent=grad[0]-final@(final.transpose(-1,-2)@grad[0])
    return final,dict(loss=loss,residual=details['residual'],
        gradient_norm=float(tangent.norm()),iterations=result.iterations,
        stop=result.stopping_criterion)


def control():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(544)
    dim,rank=7,2
    truth=torch.linalg.qr(torch.randn(dim,rank)).Q
    l=torch.stack([truth[:,0],truth[:,1],truth[:,0]])
    r=torch.stack([truth[:,0],truth[:,1],truth[:,1]])
    down=torch.randn(dim,3);metric=torch.eye(dim)
    total=((down.T@down)*product_cross(l,r,l,r)).sum()
    obj=MultioutputWeightObjective(metric,total,l=l,r=r,d=down,penalty=1e-8)
    initial=torch.linalg.qr(torch.randn(1,dim,rank)).Q
    core=cores_for(initial)
    loss,grad,_=evaluate(obj,initial,core,3)
    direction=torch.randn_like(initial);direction/=direction.norm()
    eps=1e-5
    finite=(evaluate(obj,initial+eps*direction,core,3)[0]-evaluate(obj,initial-eps*direction,core,3)[0])/(2*eps)
    gradient_error=abs(finite-float((grad[0]*direction).sum()))/max(abs(finite),1e-12)
    rotation=torch.linalg.qr(torch.randn(rank,rank)).Q
    gauge_error=abs(loss-evaluate(obj,initial@rotation,core,3)[0])
    # Explicit flattened symmetric matrices provide an independent writer solve.
    e,c=View(initial,core,3).components()
    features=torch.einsum('gri,gmrq,gqj->gmij',e,c,e).reshape(3,dim*dim)
    native=torch.einsum('ok,ki,kj->oij',down,l,r)
    native=(native+native.transpose(-1,-2))/2
    target=native.reshape(dim,-1)
    gram=features@features.T
    writer=torch.linalg.solve(gram+1e-8*torch.diag(gram.diag()),features@target.T).T
    dense=((writer@features-target).square().sum()+1e-8*(writer.square().sum(0)*gram.diag()).sum())/total
    dense_error=abs(float(dense)-loss)
    rows=[];start=time.perf_counter()
    for seed in (544,545,546):
        torch.manual_seed(seed)
        point=torch.linalg.qr(torch.randn(1,dim,rank)).Q
        final,report=fit(obj,point,seconds=20,tolerance=1e-10)
        report['seed']=seed
        report['projector_error']=float((final[0]@final[0].T-truth@truth.T).norm()/rank**.5)
        rows.append(report)
    result=dict(gradient_relative_error=gradient_error,gauge_error=gauge_error,
                dense_objective_error=dense_error,starts=rows,seconds=time.perf_counter()-start)
    assert max(gradient_error,gauge_error,dense_error)<1e-8
    assert max(row['projector_error'] for row in rows)<1e-5
    with Path(__file__).with_name('FULL_QUADRATIC_FRAME_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':control()
