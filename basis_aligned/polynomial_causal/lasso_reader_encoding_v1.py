"""Existing proximal Lasso support proposal plus exact selected-support LS."""
import torch
from quadratic_token_dictionary_v1 import conditional

@torch.no_grad()
def encode(dictionary,readers,k=128,penalty=.05,max_steps=5000):
    norms=readers.norm(dim=1)
    assert float(norms.min())>0
    x=readers/norms[:,None]
    codes,report=conditional(torch.zeros(len(x),len(dictionary),device=x.device,dtype=x.dtype),
        dictionary@dictionary.T,x@dictionary.T,penalty,'codes',max_steps,1e-7)
    gradient=(codes@dictionary-x)@dictionary.T
    kkt=torch.where(codes.abs()>1e-8,(gradient+penalty*codes.sign()).abs(),
        (gradient.abs()-penalty).clamp_min(0))
    report['maximum_code_kkt']=float(kkt.max())
    indices=codes.abs().topk(k,dim=1).indices
    del codes,gradient,kkt
    values=[];normal=0.
    for start in range(0,len(readers),64):
        sl=slice(start,start+64);atoms=dictionary[indices[sl]]
        gram=atoms@atoms.transpose(-1,-2);rhs=atoms@readers[sl,:,None]
        chol=torch.linalg.cholesky((gram+gram.transpose(-1,-2))/2)
        fitted=torch.cholesky_solve(rhs,chol)
        normal=max(normal,float((gram@fitted-rhs).norm()/rhs.norm().clamp_min(1e-30)))
        values.append(fitted.squeeze(-1))
    report['support_ls_normal_residual']=normal
    return indices,torch.cat(values),report
