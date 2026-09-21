"""Separate actual residual sources without changing native model execution."""
import torch
from native_feature_capture import capture as original_capture
from native_path_error_terms import split as previous_split

def capture(model,tokens):
 extra={}
 def pre(module,args):
  extra['incoming17']=args[0].clone();extra['x0']=args[2].clone()
 handle=model.transformer.h[17].register_forward_pre_hook(pre)
 try:result=original_capture(model,tokens)
 finally:handle.remove()
 result.update(extra)
 return result

def projected_carry(model,raw,reader):
 block=model.transformer.h[17];lam=block.lambdas.double()
 inherited=lam[0]*(raw['incoming17'].double()-raw['m16'].double())
 embedding=lam[1]*raw['x0'].double()
 attention=raw['attn17'].double()
 bias=lam[0]*model.transformer.h[16].mlp.Down_bias.double()
 return torch.stack((inherited@reader,embedding@reader,attention@reader,torch.zeros_like(embedding@reader)+bias@reader),-1)

def split(h_read,scale,alpha,beta,true_reads,fitted_reads,carry_reads):
 old=previous_split(h_read,scale,alpha,beta,true_reads,fitted_reads)
 db=(fitted_reads-true_reads)[...,1]
 # The native h, m and donor arithmetic are FP32; retain their closure residual.
 rounding=h_read-true_reads[...,0]-carry_reads.sum(-1)
 contributions=torch.cat((carry_reads,rounding[...,None]),-1)*db[...,None]/scale.square()[...,None]
 return torch.cat((contributions,old[...,1:]),-1)

def control():
 generator=torch.Generator().manual_seed(916)
 reads=torch.randn(2,7,2,generator=generator,dtype=torch.float64);fit=reads+torch.randn(2,7,2,generator=generator,dtype=torch.float64)*.3
 h=torch.randn(2,7,generator=generator,dtype=torch.float64);scale=torch.rand(2,7,generator=generator,dtype=torch.float64)+.1;carry=torch.randn(2,7,4,generator=generator,dtype=torch.float64)
 terms=split(h,scale,.7,-.4,reads,fit,carry)
 replay=float((terms.sum(-1)-previous_split(h,scale,.7,-.4,reads,fit).sum(-1)).abs().max())
 assert replay<1e-12
 return dict(replay=replay,shape=list(terms.shape))
