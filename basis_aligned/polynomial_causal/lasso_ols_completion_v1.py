"""Complete nonzero Lasso supports with exact one-step OLS gains.

Same selection algebra as shared_dictionary_ols_v1.select, retaining low-rank
Schur factors rather than updating a full candidate Gram after every addition.
Support selection remains greedy and is not a global sparse optimum.
"""
import torch
from quadratic_token_dictionary_v1 import conditional


@torch.no_grad()
def complete(gram,rhs,initial,count):
    support=list(initial);size=len(gram);threshold=float(gram.diag().max())*1e-12
    if support:
        ss=gram[support][:,support];chol=torch.linalg.cholesky((ss+ss.T)/2)
        u=torch.linalg.solve_triangular(chol,gram[support],upper=False)
        h=torch.linalg.solve_triangular(chol,rhs[support,None],upper=False).flatten()
    else:u=gram.new_empty((0,size));h=rhs.new_empty((0,))
    diagonal=gram.diag().clone()-u.square().sum(0);correlation=rhs-h@u;gains=[]
    while len(support)<count:
        valid=diagonal>threshold
        if support:valid[support]=False
        assert bool(valid.any()),'Insufficient independent atoms; no hidden padding or ridge'
        scores=correlation.square()/diagonal.clamp_min(threshold);scores[~valid]=-torch.inf
        j=int(scores.argmax());norm=diagonal[j].sqrt()
        v=(gram[j]-u[:,j]@u)/norm;z=correlation[j]/norm
        gains.append(float(z.square()));support.append(j)
        correlation-=z*v;diagonal-=v.square();u=torch.cat((u,v[None]),0)
    return support,gains


@torch.no_grad()
def encode(dictionary,readers,k=128,penalty=.05):
    norms=readers.norm(dim=1);x=readers/norms[:,None]
    gram=dictionary@dictionary.T;rhs=x@dictionary.T
    codes,report=conditional(torch.zeros_like(rhs),gram,rhs,penalty,'codes',5000,1e-7)
    gradient=codes@gram-rhs
    kkt=torch.where(codes.abs()>1e-8,(gradient+penalty*codes.sign()).abs(),(gradient.abs()-penalty).clamp_min(0))
    report['maximum_code_kkt']=float(kkt.max());counts=[];all_ids=[];all_values=[];normal=0.
    for row in range(len(x)):
        order=codes[row].abs().argsort(descending=True)
        initial=order[codes[row,order].abs()>1e-8][:k].tolist();counts.append(len(initial))
        support,_=complete(gram,rhs[row],initial,k)
        ss=gram[support][:,support];chol=torch.linalg.cholesky((ss+ss.T)/2)
        values=torch.cholesky_solve(rhs[row,support,None],chol).flatten()
        normal=max(normal,float((ss@values-rhs[row,support]).norm()/rhs[row,support].norm().clamp_min(1e-30)))
        all_ids.append(torch.tensor(support,device=x.device));all_values.append(values*norms[row])
    report.update(support_ls_normal_residual=normal,initial_support_counts=counts,
                  rows_requiring_completion=sum(c<k for c in counts))
    return torch.stack(all_ids),torch.stack(all_values),report
