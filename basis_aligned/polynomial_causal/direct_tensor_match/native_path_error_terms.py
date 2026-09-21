"""Native residual/source split of the exact component error."""
import torch

def split(h_read,scale,alpha,beta,true_reads,fitted_reads):
 qa,qb=true_reads.unbind(-1);da,db=(fitted_reads-true_reads).unbind(-1)
 carry=h_read-qa
 return torch.stack((carry*db/scale.square(),.5*qa*db/scale.square(),-alpha*db/scale,-.5*da*qb/scale.square(),.5*beta*da/scale,-.5*da*db/scale.square()),-1)

def control():
 from read_error_terms import split as original
 generator=torch.Generator().manual_seed(915)
 reads=torch.randn(2,7,2,generator=generator,dtype=torch.float64);fit=reads+torch.randn(2,7,2,generator=generator,dtype=torch.float64)*.3
 h=torch.randn(2,7,generator=generator,dtype=torch.float64);scale=torch.rand(2,7,generator=generator,dtype=torch.float64)+.1
 parts=split(h,scale,.7,-.4,reads,fit)
 residual=float((parts.sum(-1)-original(h,scale,.7,-.4,reads,fit).sum(-1)).abs().max())
 assert residual<1e-12
 return dict(replay=residual,shape=list(parts.shape))
