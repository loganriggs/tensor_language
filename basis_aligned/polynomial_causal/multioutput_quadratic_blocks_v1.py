"""Weight-only block terms: several quadratic outputs share each input basis.

Exact Gram contractions; no explicit1152x1152 input matrices or token tensor.
Signed, nonorthogonal blocks may overlap; symmetry enforced without dead skew
parameters. Each feature core has unit Frobenius norm, with amplitude in writer.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective

class MultioutputQuadraticBlocks(nn.Module):
    def __init__(self,dim,groups=16,rank=16,outputs=4,device='cpu'):
        super().__init__();self.kind='multioutput_block';self.dim=dim;self.groups=groups;self.rank=rank;self.outputs=outputs
        self.bank=nn.Parameter(torch.randn(groups,rank,dim,device=device,dtype=torch.float64))
        self.core=nn.Parameter(torch.randn(groups,outputs,rank*(rank+1)//2,device=device,dtype=torch.float64))

    def components(self):
        bank=F.normalize(self.bank,dim=-1);v=F.normalize(self.core,dim=-1);i,j=torch.triu_indices(self.rank,self.rank,device=v.device)
        scale=torch.where(i==j,torch.ones_like(i,dtype=v.dtype),torch.full_like(i,1/math.sqrt(2),dtype=v.dtype))
        c=v.new_zeros(self.groups,self.outputs,self.rank,self.rank)
        c[...,i,j]=v*scale;c[...,j,i]=v*scale
        return bank,c

    def storage_numbers(self):return sum(p.numel() for p in self.parameters())

class MultioutputWeightObjective(EnergyRegularizedObjective):
    def cross_gram(self,model):
        e,c=model.components()
        le=torch.einsum('ki,gri->gkr',self.l,e);re=torch.einsum('ki,gri->gkr',self.r,e)
        native=torch.einsum('gki,gmij,gkj->kgm',le,c,re).reshape(len(self.l),-1)
        cross=self.d@native
        overlap=torch.einsum('gri,hsi->ghrs',e,e)
        transported=torch.einsum('ghik,hnkl,ghjl->ghnij',overlap,c,overlap)
        gram=torch.einsum('gmij,ghnij->gmhn',c,transported).reshape(model.groups*model.outputs,-1)
        return cross,(gram+gram.T)/2
