"""Standard manifold adapter for the existing overlapping block objective.

Within-block basis columns are orthonormal; different blocks may overlap.
Each packed symmetric core has unit norm. No changed objective or penalty.
"""
import json,math,time
from pathlib import Path
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Product,Stiefel,Oblique
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher
from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective


class View:
    def __init__(self,bank,packed,outputs):
        self.bank=bank;self.packed=packed;self.groups=bank.shape[0];self.rank=bank.shape[-1];self.outputs=outputs
    def components(self):
        v=self.packed.T.reshape(self.groups,self.outputs,-1)
        i,j=torch.triu_indices(self.rank,self.rank,device=v.device)
        scale=torch.where(i==j,torch.ones_like(i,dtype=v.dtype),torch.full_like(i,1/math.sqrt(2),dtype=v.dtype))
        c=v.new_zeros(self.groups,self.outputs,self.rank,self.rank)
        c[...,i,j]=v*scale;c[...,j,i]=v*scale
        return self.bank.transpose(-1,-2),c


def manifold_for(bank,packed):
    groups,dim,rank=bank.shape
    return Product([Stiefel(dim,rank,k=groups),Oblique(*packed.shape)])


def evaluate(objective,bank,packed,outputs):
    bank=bank.detach().requires_grad_();packed=packed.detach().requires_grad_()
    loss,residual,energy,_,_,_=objective.terms(View(bank,packed,outputs))
    grads=torch.autograd.grad(loss,(bank,packed))
    return float(loss),[g.detach() for g in grads],dict(residual=float(residual),component_energy=float(energy))


def fit(objective,bank,packed,outputs,seconds=120,tolerance=1e-5):
    manifold=manifold_for(bank,packed);cache={};evaluations=0
    def calc(b,c):
        nonlocal evaluations
        if 'b' not in cache or not np.array_equal(cache['b'],b) or not np.array_equal(cache['c'],c):
            bt=torch.as_tensor(b,device=bank.device,dtype=bank.dtype);ct=torch.as_tensor(c,device=packed.device,dtype=packed.dtype)
            value,grad,details=evaluate(objective,bt,ct,outputs)
            cache.update(b=b.copy(),c=c.copy(),value=value,grad=[g.cpu().numpy() for g in grad],details=details);evaluations+=1
        return cache
    @pymanopt.function.numpy(manifold)
    def cost(b,c):return calc(b,c)['value']
    @pymanopt.function.numpy(manifold)
    def gradient(b,c):return calc(b,c)['grad']
    problem=Problem(manifold,cost,euclidean_gradient=gradient)
    solver=ConjugateGradient(beta_rule='PolakRibiere',min_gradient_norm=tolerance,min_step_size=1e-18,
        max_iterations=10000,max_time=seconds,max_cost_evaluations=100000,verbosity=0,
        line_searcher=BackTrackingLineSearcher(max_iterations=50))
    initial=[bank.detach().cpu().numpy(),packed.detach().cpu().numpy()];start=time.perf_counter()
    result=solver.run(problem,initial_point=initial);row=calc(*result.point)
    tangent=manifold.euclidean_to_riemannian_gradient(result.point,row['grad'])
    station=float(manifold.norm(result.point,tangent))
    return [torch.from_numpy(x.copy()) for x in result.point],dict(loss=row['value'],**row['details'],
        tangent_stationarity=station,converged=station<=tolerance,seconds=time.perf_counter()-start,
        iterations=int(result.iterations),evaluations=evaluations,stopping_criterion=result.stopping_criterion)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1499)
    g,dim,rank,outputs,native=2,7,3,2,11
    bank=torch.linalg.qr(torch.randn(g,dim,rank)).Q
    packed=torch.randn(rank*(rank+1)//2,g*outputs);packed/=packed.norm(dim=0)
    l,r=torch.randn(2,native,dim);d=torch.randn(dim,native);u=torch.randn(13,dim);metric=u.T@u
    forms=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
    target=torch.einsum('oj,jab->oab',d,forms);total=(u@target.flatten(1)).square().sum()
    objective=MultioutputWeightObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    value,grad,_=evaluate(objective,bank,packed,outputs)
    directions=[torch.randn_like(bank),torch.randn_like(packed)];directions=[x/x.norm() for x in directions]
    delta=1e-5
    plus=evaluate(objective,bank+delta*directions[0],packed+delta*directions[1],outputs)[0]
    minus=evaluate(objective,bank-delta*directions[0],packed-delta*directions[1],outputs)[0]
    analytic=sum(float((a*b).sum()) for a,b in zip(grad,directions));fd=abs((plus-minus)/(2*delta)-analytic)
    view=View(bank,packed,outputs);e,c=view.components()
    dense=torch.einsum('gri,gmrs,gsj->gmij',e,c,e).flatten(0,1)
    cross,gram=objective.cross_gram(view)
    err=max(float((gram-dense.flatten(1)@dense.flatten(1).T).abs().max()),
        float((cross-target.flatten(1)@dense.flatten(1).T).abs().max()))
    fitted,report=fit(objective,bank,packed,outputs,seconds=3,tolerance=1e-5)
    result=dict(instrument_passed=max(fd,err)<1e-8 and report['loss']<value,
        finite_gradient_error=fd,dense_contraction_error=err,initial_loss=value,solver=report,
        scope='Same penalized overlapping-block objective, conditional-gradient and dense function controls. Toy solver improvement required; native convergence untested.')
    with Path(__file__).with_name('ORTHOGONAL_MULTIOUTPUT_PYMANOPT_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']

if __name__=='__main__':control()
