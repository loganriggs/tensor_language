"""Executable fixed bilinear candidates; shared readers computed once per input."""
import torch
from torch import nn
import torch.nn.functional as F

class CompiledQuadraticLayer(nn.Module):
    def __init__(self,description):
        super().__init__();self.kind=description['kind']
        for name in ['bank','left','right','writer','bias']:
            self.register_buffer(name,description[name].detach().clone())

    def forward(self,x):
        h=F.linear(x,self.bank) if self.kind=='shared_reader' else x
        return F.linear(F.linear(h,self.left)*F.linear(h,self.right),self.writer)+self.bias

    def numbers(self):return sum(b.numel() for b in self.buffers())

def compile_description(model,writer,bias):
    if model.kind=='shared_reader':
        bank=F.normalize(model.bank,dim=1)
        left=model.ca/(model.ca@bank).norm(dim=1,keepdim=True)
        right=model.cb/(model.cb@bank).norm(dim=1,keepdim=True)
    elif model.kind=='product':
        bank=writer.new_empty((0,model.dim));left,right,c=model.components()
        assert torch.equal(c,torch.eye(len(left),device=c.device,dtype=c.dtype))
    else:raise ValueError('Only fixed shared-reader and free-product circuits here')
    return dict(kind=model.kind,**{k:v.detach().cpu() for k,v in dict(bank=bank,left=left,right=right,writer=writer,bias=bias).items()})
