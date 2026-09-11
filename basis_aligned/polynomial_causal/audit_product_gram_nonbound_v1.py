"""Orthogonal native components can share readers and admit fewer new products."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
torch.manual_seed(720)
d=8
basis=torch.linalg.qr(torch.randn(d,d)).Q
forms=[]
pairs=[]
for i in range(d):
    for j in range(i,d):
        factor=1. if i==j else 2.**.5
        form=factor*(torch.outer(basis[i],basis[j])+torch.outer(basis[j],basis[i]))/2
        forms.append(form);pairs.append((i,j,factor))
forms=torch.stack(forms)
gram=forms.flatten(1)@forms.flatten(1).T
x=torch.randn(31,d)
h=x@basis.T
shared=sum(factor*h[:,i]*h[:,j] for i,j,factor in pairs)
total=forms.sum(0)
direct=torch.einsum('ni,ij,nj->n',x,total,x)
eigenvalues,eigenvectors=torch.linalg.eigh(total)
spectral=((x@eigenvectors).square()*eigenvalues).sum(-1)
errors=dict(gram_identity=float((gram-torch.eye(len(forms))).norm()),
    shared_execution=float((shared-direct).norm()/direct.norm()),
    spectral_execution=float((spectral-direct).norm()/direct.norm()))
result=dict(predictions={
    'pred_a_orthogonal_components':errors['gram_identity']<=1e-12,
    'pred_b_shared_features':errors['shared_execution']<=1e-12,
    'pred_c_fewer_new_products':errors['spectral_execution']<=1e-12 and d<len(forms)},
    errors=errors,original_products=len(forms),shared_input_features=d,
    stored_shared_basis_numbers=d*d,naive_two_reader_numbers=2*len(forms)*d,
    refactored_square_products=d,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Constructive counterexample: high component-Gram rank does not lower-bound new product count or shared-intermediate complexity. Not evidence native tensors share this structure.')
with Path(__file__).with_name('PRODUCT_GRAM_NONBOUND_V1_CONTROL.json').open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
