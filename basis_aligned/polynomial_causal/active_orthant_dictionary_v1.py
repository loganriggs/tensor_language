"""Coupled fixed-orthant dictionary/code polish using SciPy L-BFGS-B.

Keep current nonzero code positions/signs; bound magnitudes below by zero.
Normalize dictionary rows in the objective. Used atoms can be unit norm at an
L1 optimum: rescale a shorter atom and its code column reciprocally, preserving
reconstruction and reducing the penalty. Full-code refresh/checks belong to
the caller; fixed-orthant solver success is not joint Lasso stationarity.
"""
import time
import numpy as np
import torch
from scipy.optimize import Bounds,minimize


class OrthantObjective:
    def __init__(self,x,codes,dictionary,penalty):
        self.x=x;self.penalty=penalty;self.shape=dictionary.shape;self.n_basis=dictionary.numel()
        norm=dictionary.norm(dim=1);assert bool((norm>1e-12).all())
        normalized=dictionary/norm[:,None];canonical=codes*norm[None,:]
        self.row,self.column=canonical.nonzero(as_tuple=True)
        self.sign=canonical[self.row,self.column].sign();self.code_shape=codes.shape
        self.initial=torch.cat((normalized.flatten(),canonical[self.row,self.column].abs())).detach().cpu().numpy().copy()
        before=.5*(codes@dictionary-x).square().sum()+penalty*codes.abs().sum()
        after=.5*(canonical@normalized-x).square().sum()+penalty*canonical.abs().sum()
        self.canonicalization=dict(reconstruction_error=float((codes@dictionary-canonical@normalized).norm()/x.norm()),
            aggregate_objective_change=float(after-before),original_atom_norm_min=float(norm.min()),original_atom_norm_max=float(norm.max()))
        self.last_point=None;self.last_value=None;self.evaluations=0

    def unpack(self,point):
        vector=torch.as_tensor(point,dtype=self.x.dtype,device=self.x.device)
        raw=vector[:self.n_basis].reshape(self.shape);norm=raw.norm(dim=1,keepdim=True)
        assert bool((norm>1e-12).all()),'Vanishing dictionary parameter row'
        basis=raw/norm;magnitudes=vector[self.n_basis:]
        codes=self.x.new_zeros(self.code_shape);codes[self.row,self.column]=self.sign*magnitudes
        return basis,codes,raw,norm,magnitudes

    @torch.no_grad()
    def value_gradient(self,point):
        basis,codes,raw,norm,magnitudes=self.unpack(point)
        residual=codes@basis-self.x
        value=.5*residual.square().sum()+self.penalty*magnitudes.sum()
        basis_grad=codes.T@residual
        raw_grad=(basis_grad-(basis_grad*basis).sum(1,keepdim=True)*basis)/norm
        code_grad=(residual@basis.T)[self.row,self.column]*self.sign+self.penalty
        gradient=torch.cat((raw_grad.flatten(),code_grad))
        assert bool(torch.isfinite(value)) and bool(torch.isfinite(gradient).all())
        self.last_point=np.array(point,copy=True);self.last_value=float(value);self.evaluations+=1
        return self.last_value,gradient.cpu().numpy().copy()


class PolishBudget(Exception):pass


def polish(x,codes,dictionary,penalty=.05,max_iterations=200,seconds=30.):
    objective=OrthantObjective(x,codes,dictionary,penalty);initial=objective.initial
    lower=np.full(len(initial),-np.inf);lower[objective.n_basis:]=0.
    bounds=Bounds(lower,np.full(len(initial),np.inf));started=time.perf_counter()
    accepted=initial.copy();history=[];initial_value=objective.value_gradient(initial)[0]
    def callback(point):
        nonlocal accepted
        accepted=np.array(point,copy=True)
        value=objective.last_value if np.array_equal(point,objective.last_point) else objective.value_gradient(point)[0]
        history.append(dict(iteration=len(history)+1,mean_objective=value/len(x),seconds=time.perf_counter()-started))
        if time.perf_counter()-started>=seconds:raise PolishBudget()
    try:
        result=minimize(objective.value_gradient,initial,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
            options=dict(maxiter=max_iterations,maxfun=max_iterations*25,maxls=20,maxcor=10,ftol=1e-15,gtol=1e-9))
        accepted=result.x;status=dict(success=bool(result.success),status=int(result.status),message=str(result.message))
    except PolishBudget:
        status=dict(success=False,status=None,message='Time budget at accepted iteration')
    value,gradient=objective.value_gradient(accepted)
    basis,new_codes,raw,norm,magnitudes=objective.unpack(accepted)
    projected=gradient.copy();tail=projected[objective.n_basis:]
    tail[(accepted[objective.n_basis:]<=1e-12)&(tail>0)]=0.
    report=dict(solver=status,initial_mean_objective=initial_value/len(x),final_mean_objective=value/len(x),
        evaluations=objective.evaluations,iterations=len(history),seconds=time.perf_counter()-started,
        fixed_code_positions=len(objective.row),projected_parameter_gradient_max=float(np.max(np.abs(projected))),
        raw_dictionary_norm_min=float(norm.min()),raw_dictionary_norm_max=float(norm.max()),
        minimum_code_magnitude=float(magnitudes.min()) if len(magnitudes) else None,
        canonicalization=objective.canonicalization,history=history,
        scope='Fixed positions/signs with nonnegative magnitudes; normalized dictionary. Full problem convergence must be measured separately.')
    return new_codes.detach(),basis.detach(),report
