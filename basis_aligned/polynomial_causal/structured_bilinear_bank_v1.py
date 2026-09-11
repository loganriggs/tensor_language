"""Full-rank structured product banks and exact folded-coefficient backprop."""
import json
from pathlib import Path
import torch
from torch import nn
from mixed_radix_bilinear_v1 import MixedRadix
from chunked_bilinear_coefficient_v1 import value_gradient,dense


class StructuredBank(nn.Module):
    def __init__(self,radices,branches=4):
        super().__init__();self.radices=tuple(radices)
        self.maps=nn.ModuleList([nn.ModuleList([MixedRadix(radices) for _ in range(3)]) for _ in range(branches)])

    def matrices(self):
        return [[model.matrix() for model in branch] for branch in self.maps]

    def factors(self):
        matrices=self.matrices()
        return torch.cat([m[0] for m in matrices]),torch.cat([m[1] for m in matrices]),torch.cat([m[2] for m in matrices],dim=1)

    @torch.no_grad()
    def initialize(self,targets):
        """Independent orthogonal small stages, scaled to native matrix norms."""
        for maps,matrices in zip(self.maps,targets):
            for model,target in zip(maps,matrices):
                for p in model.stages:p.copy_(torch.linalg.qr(torch.randn_like(p)).Q)
                model.stages[-1].mul_(target.norm()/(model.dim**.5))

    def matrix_loss(self,targets):
        actual=self.matrices()
        losses=[(a-b).square().sum()/b.square().sum() for aa,bb in zip(actual,targets) for a,b in zip(aa,bb)]
        return torch.stack(losses).mean()

    def coefficient_loss(self,native,whitener,total,penalty=.01,chunk=256,backward=True):
        a,b,w=self.factors();white=whitener@w
        loss,grad,details=value_gradient(*native,a,b,white,total,penalty,chunk)
        if backward:torch.autograd.backward((a,b,white),grad)
        return loss,details


def control():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(620)
    model=StructuredBank([2,2,2],branches=2)
    targets=[[torch.randn(8,8) for _ in range(3)] for _ in range(2)]
    model.initialize(targets)
    matrices=model.matrices()
    norm_error=max(abs(float(a.norm()/b.norm())-1) for aa,bb in zip(matrices,targets) for a,b in zip(aa,bb))
    na,nb,nw=torch.randn(11,8),torch.randn(11,8),torch.randn(8,11)
    u=torch.randn(13,8);native=(na,nb,u@nw);target=dense(*native);total=target.square().sum()
    a,b,w=model.factors();white=u@w;prediction=dense(a,b,white)
    energy=(white.square().sum(0)*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())/2).sum()
    direct=((prediction-target).square().sum()+.01*energy)/total
    expected=torch.autograd.grad(direct,tuple(model.parameters()))
    model.zero_grad(set_to_none=True)
    found,_=model.coefficient_loss(native,u,total,penalty=.01,chunk=5)
    errors=[float((p.grad-g).norm()/g.norm().clamp_min(1e-30)) for p,g in zip(model.parameters(),expected)]
    result=dict(matrix_norm_initialization_error=norm_error,loss_error=abs(float(found-direct)),
                parameter_gradient_error=max(errors),parameter_count=sum(p.numel() for p in model.parameters()))
    with Path(__file__).with_name('STRUCTURED_BILINEAR_BANK_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert max(norm_error,result['loss_error'],max(errors))<1e-10


if __name__=='__main__':control()
