"""Profile output-local quartics against Gaussian and sensitivity-weighted moments."""
import torch
from mixed_gaussian_cp import gram_dynamic
from noncentral_gaussian_cp import cross

def profile(G,X,phi,residual,sensitivity,per_output,output_weights,eta=.5,ridge=1e-6,detach=True):
    assert 0<=eta<=1 and X.shape[0]==residual.shape[1]==sensitivity.shape[1]
    assert torch.isfinite(sensitivity).all() and (sensitivity>=0).all()
    assert (sensitivity.mean(0)>0).all()
    w=sensitivity/sensitivity.mean(0,keepdim=True)
    C=G.new_zeros(X.shape);loss=G.new_zeros(());normal=[]
    for g in range(X.shape[0]):
        sl=slice(g*per_output,(g+1)*per_output);z=phi[:,sl]
        empiricalG=z.T@(w[:,g,None]*z)/len(z)
        empiricalX=z.T@(w[:,g]*residual[:,g])/len(z)
        gram=(1-eta)*G[sl,sl]+eta*empiricalG
        target=(1-eta)*X[g,sl]+eta*empiricalX
        K=gram+ridge*torch.eye(per_output,dtype=G.dtype,device=G.device)
        solved=torch.linalg.solve(K,target);c=solved.detach() if detach else solved
        C[g,sl]=c
        loss=loss+output_weights[g]*(c@gram@c-2*c@target+ridge*c.square().sum())
        normal.append(((K@solved-target).norm()/target.norm().clamp_min(1e-30)).detach())
    return loss,C,dict(normal_residual=float(torch.stack(normal).max()))

def objective(teacher,transformed,location,projection,S,mu,parent_f,parent_C,factors,per_output,weights,x,residual,sensitivity,eta=.5,ridge=1e-6):
    fw=[f@S for f in factors];b=[f@mu for f in factors]
    pw=[f@S for f in parent_f];pb=[f@mu for f in parent_f]
    G=gram_dynamic(fw,b,fw,b)
    X=cross(transformed,location,projection,fw,b)-parent_C@gram_dynamic(pw,pb,fw,b)
    phi=torch.stack([x@f.T for f in factors]).prod(0)
    return profile(G,X,phi,residual,sensitivity,per_output,weights,eta,ridge)
