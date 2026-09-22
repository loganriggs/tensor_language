"""Profile new quartic products with disjoint per-output readouts."""
import torch
from mixed_gaussian_cp import gram_dynamic
from noncentral_gaussian_cp import cross
from quartic_cp import cp_gram,directional

def objective(teacher,transformed,location,projection,S,mu,parent_f,parent_C,factors,per_output,weights=None,coefficient_weight=0.,ridge=1e-6):
    nout=parent_C.shape[0];n=nout*per_output
    assert factors[0].shape[0]==n
    fw=[f@S for f in factors];b=[f@mu for f in factors];pw=[f@S for f in parent_f];pb=[f@mu for f in parent_f]
    G=gram_dynamic(fw,b,fw,b)
    X=cross(transformed,location,projection,fw,b)-parent_C@gram_dynamic(pw,pb,fw,b)
    if coefficient_weight:
        G=G+coefficient_weight*cp_gram(factors,factors)
        X=X+coefficient_weight*(directional(*teacher,factors).T-parent_C@cp_gram(parent_f,factors))
    G=G/(1+coefficient_weight);X=X/(1+coefficient_weight)
    weights=G.new_ones(nout) if weights is None else weights
    C=G.new_zeros(nout,n);loss=G.new_zeros(());normal=[]
    for out in range(nout):
        sl=slice(out*per_output,(out+1)*per_output);g=G[sl,sl];x=X[out,sl];K=g+ridge*torch.eye(per_output,dtype=g.dtype,device=g.device)
        c=torch.linalg.solve(K,x).detach();C[out,sl]=c
        loss=loss+weights[out]*(c@g@c-2*c@x+ridge*c.square().sum())
        normal.append(((K@c-x).norm()/x.norm().clamp_min(1e-30)).detach())
    return loss,C,dict(normal_residual=float(torch.stack(normal).max()))
