"""Distinct quadratic inductive biases with exact shared-output least squares.

Shared-reader models reuse a linear dictionary; block models tie the output
writer across several signed squares. Neither imposes sparse token support.
"""
import torch
from torch import nn
import torch.nn.functional as F
from joint_quadratic_fit_v1 import product_cross

class QuadraticModel(nn.Module):
    def __init__(self, kind, dim, products=128, readers=64, groups=32, block_size=8, device='cpu'):
        super().__init__();self.kind=kind;self.dim=dim;self.products=products;self.readers=readers;self.groups=groups;self.block_size=block_size
        def parameter(*shape):return nn.Parameter(torch.randn(*shape,device=device,dtype=torch.float64))
        if kind=='product':self.a=parameter(products,dim);self.b=parameter(products,dim)
        elif kind=='square':self.a=parameter(2*products,dim)
        elif kind=='shared_reader':self.bank=parameter(readers,dim);self.ca=parameter(products,readers);self.cb=parameter(products,readers)
        elif kind=='block':self.a=parameter(groups*block_size,dim);self.coefficients=parameter(groups,block_size)
        else:raise ValueError(kind)

    def components(self):
        if self.kind=='shared_reader':
            bank=F.normalize(self.bank,dim=1);a=F.normalize(self.ca@bank,dim=1);b=F.normalize(self.cb@bank,dim=1)
        else:
            a=F.normalize(self.a,dim=1);b=F.normalize(self.b,dim=1) if self.kind=='product' else a
        if self.kind=='block':
            coefficients=F.normalize(self.coefficients,dim=1)
            c=torch.block_diag(*[row[:,None] for row in coefficients])
        else:c=torch.eye(len(a),device=a.device,dtype=a.dtype)
        return a,b,c

    def storage_numbers(self):return sum(p.numel() for p in self.parameters())

class QuadraticObjective:
    def __init__(self,metric,total, *, l=None,r=None,d=None,x=None,y=None):
        self.metric=metric;self.total=total;self.l=l;self.r=r;self.d=d;self.x=x;self.y=y

    def cross_gram(self,model):
        a,b,c=model.components()
        if self.x is None:
            cross=self.d@(product_cross(self.l,self.r,a,b)@c)
            gram=c.T@product_cross(a,b,a,b)@c
        else:
            features=((self.x@a.T)*(self.x@b.T))@c
            cross=self.y.T@features;gram=features.T@features
        return cross,(gram+gram.T)/2

    def loss(self,model):
        cross,gram=self.cross_gram(model)
        # No ridge that silently changes the fitting objective. Singular/ill-
        # conditioned fits are optimizer failures to diagnose, not structure nulls.
        with torch.no_grad():w=torch.linalg.solve(gram,cross.T).T
        loss=(self.total+((w.T@self.metric@w)*gram).sum()-2*((self.metric@cross)*w).sum())/self.total
        return loss,w,gram

    def diagnostics(self,model):
        model.zero_grad(set_to_none=True);loss,w,gram=self.loss(model);loss.backward()
        captured=max(1-float(loss.detach()),1e-12)
        stationarity=max(float(p.grad.norm()*p.detach().norm().clamp_min(1))/captured for p in model.parameters())
        return dict(squared_relative_error=float(loss.detach()),captured_energy_fraction=captured,
            relative_stationarity=stationarity,gradient_max_abs=max(float(p.grad.abs().max()) for p in model.parameters()),
            gram_condition=float(torch.linalg.cond(gram.detach()))),w.detach()
