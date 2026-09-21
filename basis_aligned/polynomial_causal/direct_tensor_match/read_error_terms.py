"""Exact signed error expansion for the normalized two-read component."""
import torch

def split(h_read, scale, alpha, beta, true_reads, fitted_reads):
    qa,qb=true_reads.unbind(-1)
    da,db=(fitted_reads-true_reads).unbind(-1)
    a=(h_read-.5*qa)/scale-alpha
    b=qb/scale-beta
    return torch.stack((-.5*da*b/scale,a*db/scale,-.5*da*db/scale.square()),-1)

def baseline_reads(z,program):
    t=z@program['shared_reader']
    i,j,kind=program['product_indices'].long()
    u,v=t[...,i],t[...,j]
    terms=torch.where(kind==1,u+v,u)*torch.where(kind==1,u-v,v)
    quadratic=terms@program['product_weights']
    return quadratic+torch.stack((z@program['a_linear']+program['a_bias'],z@program['b_linear']+program['b_bias']),-1)

def control():
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(914)
    reads=torch.randn(2,7,2,generator=generator,dtype=torch.float64)
    fit=reads+torch.randn(2,7,2,generator=generator,dtype=torch.float64)*.3
    h=torch.randn(2,7,generator=generator,dtype=torch.float64)
    scale=torch.rand(2,7,generator=generator,dtype=torch.float64)+.1
    alpha,beta=.7,-.4
    def value(r):return ((h-.5*r[...,0])/scale-alpha)*(r[...,1]/scale-beta)
    terms=split(h,scale,alpha,beta,reads,fit)
    error=value(fit)-value(reads)
    residual=float((terms.sum(-1)-error).abs().max())
    gram=terms.flatten(0,1).T@terms.flatten(0,1)
    energy_residual=float((gram.sum()-error.square().sum()).abs())
    change=terms[1]-terms[0]
    change_residual=float((change.sum(-1)-(error[1]-error[0])).abs().max())
    assert residual<1e-12 and energy_residual<1e-10 and change_residual<1e-12
    return dict(max_error=residual,energy_error=energy_residual,change_error=change_residual,shape=list(terms.shape))
