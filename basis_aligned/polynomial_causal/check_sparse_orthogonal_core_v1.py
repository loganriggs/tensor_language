import json,torch
from pathlib import Path
from sparse_orthogonal_quadratic_core_v1 import input_marginal,orthogonal_core

torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(711)
l,r,d=torch.randn(7,5),torch.randn(7,5),torch.randn(4,7)
native=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
t=torch.einsum('ok,kij->oij',d,native)
m=input_marginal(l,r,d.T@d);direct=torch.einsum('oij,okj->ik',t,t)
q,_=torch.linalg.qr(torch.randn(5,3));basis=q.T;w,edges=orthogonal_core(l,r,d,basis,chunk=2)
i,j=edges;features=(basis[i,:,None]*basis[j,None,:]+basis[j,:,None]*basis[i,None,:])/torch.where(i==j,2.,2**.5)[:,None,None]
expected=torch.einsum('oij,fij->of',t,features);gram=features.flatten(1)@features.flatten(1).T
order=w.square().sum(0).argsort(descending=True);kept=order[:3];approx=torch.einsum('of,fij->oij',w[:,kept],features[kept])
error=(t-approx).square().sum();spectral=t.square().sum()-w[:,kept].square().sum()
result=dict(input_marginal_relative_error=float((m-direct).norm()/direct.norm()),core_relative_error=float((w-expected).norm()/expected.norm()),feature_gram_error=float((gram-torch.eye(len(i))).abs().max()),selected_error_identity=float(abs(error-spectral)/t.square().sum()))
result['passed']=max(result.values())<=1e-10;assert result['passed'],result
Path(__file__).with_name('SPARSE_ORTHOGONAL_CORE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
