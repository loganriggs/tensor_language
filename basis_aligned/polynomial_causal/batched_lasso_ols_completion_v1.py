"""Batch the existing Lasso-support/OLS-completion algorithm, without new fitting.
Group equal initial support sizes and retain low-rank Schur factors on device.
"""
import torch
from quadratic_token_dictionary_v1 import conditional


@torch.no_grad()
def encode(dictionary,readers,k=128,penalty=.05,batch_size=64):
    norms=readers.norm(dim=1);assert bool((norms>0).all())
    x=readers/norms[:,None];gram=dictionary@dictionary.T;rhs=x@dictionary.T
    codes,report=conditional(torch.zeros_like(rhs),gram,rhs,penalty,'codes',5000,1e-7)
    gradient=codes@gram-rhs
    kkt=torch.where(codes.abs()>1e-8,(gradient+penalty*codes.sign()).abs(),(gradient.abs()-penalty).clamp_min(0))
    report['maximum_code_kkt']=float(kkt.max())
    order=codes.abs().argsort(dim=1,descending=True)[:,:k]
    counts=(codes.abs()>1e-8).sum(1).clamp_max(k)
    del codes,gradient,kkt
    all_ids=torch.empty((len(x),k),dtype=torch.long,device=x.device)
    all_values=x.new_empty((len(x),k));normal=0.;threshold=gram.diag().max()*1e-12
    for count in counts.unique().cpu().tolist():
        members=(counts==count).nonzero().flatten()
        for start in range(0,len(members),batch_size):
            rows=members[start:start+batch_size];n=len(rows);cross=rhs[rows]
            ids=torch.empty((n,k),dtype=torch.long,device=x.device);ids[:,:count]=order[rows,:count]
            u=gram.new_empty((n,k,len(gram)))
            if count:
                selected=ids[:,:count];ss=gram[selected[:,:,None],selected[:,None,:]]
                chol=torch.linalg.cholesky((ss+ss.transpose(1,2))/2)
                u[:,:count]=torch.linalg.solve_triangular(chol,gram[selected],upper=False)
                h=torch.linalg.solve_triangular(chol,cross.gather(1,selected)[...,None],upper=False).squeeze(-1)
                correlation=cross-torch.bmm(h[:,None,:],u[:,:count]).squeeze(1)
                diagonal=gram.diag()[None,:]-u[:,:count].square().sum(1)
            else:
                correlation=cross.clone();diagonal=gram.diag()[None,:].expand(n,-1).clone()
            for step in range(count,k):
                valid=diagonal>threshold
                if step:valid.scatter_(1,ids[:,:step],False)
                assert bool(valid.any(1).all()),'Insufficient independent atoms; no hidden padding or ridge'
                scores=correlation.square()/diagonal.clamp_min(threshold)
                chosen=scores.masked_fill(~valid,-torch.inf).argmax(1)
                norm=diagonal.gather(1,chosen[:,None]).sqrt()
                if step:
                    coefficients=u[:,:step].gather(2,chosen[:,None,None].expand(-1,step,1)).squeeze(-1)
                    projected=torch.bmm(coefficients[:,None,:],u[:,:step]).squeeze(1)
                else:projected=torch.zeros_like(cross)
                v=(gram[chosen]-projected)/norm;z=correlation.gather(1,chosen[:,None])/norm
                correlation-=z*v;diagonal-=v.square();u[:,step]=v;ids[:,step]=chosen
            ss=gram[ids[:,:,None],ids[:,None,:]];chol=torch.linalg.cholesky((ss+ss.transpose(1,2))/2)
            target=cross.gather(1,ids);values=torch.cholesky_solve(target[...,None],chol).squeeze(-1)
            residual=(torch.bmm(ss,values[...,None]).squeeze(-1)-target).norm(dim=1)/target.norm(dim=1).clamp_min(1e-30)
            normal=max(normal,float(residual.max()));all_ids[rows]=ids;all_values[rows]=values*norms[rows,None]
    report.update(support_ls_normal_residual=normal,initial_support_counts=counts.cpu().tolist(),
                  rows_requiring_completion=int((counts<k).sum()),batch_size=batch_size)
    return all_ids,all_values,report
