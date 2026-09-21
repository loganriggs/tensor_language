"""Fixed-affine rank bound from an orthogonal sector of the cubic Hermite tensor.
P projects away both fixed affine vectors u,v. The sector with two P slots and
one span(u,v) slot has squared Gaussian error
2[||v||^2||P dA P||F^2+||u||^2||P dB P||F^2+2(u.v)<P dA P,P dB P>].
Apply Schur complements and best-rank-r matrix tail bounds.
"""
import json
from pathlib import Path
import torch
from gaussian_quartic_lowrank import prepare_teacher
PTH=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_grad_enabled(False)
f=torch.load(PTH/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True)['frames']['covariance']
A=f['A'];B=f['B'];u=2*A[:-1,-1];v=2*B[:-1,-1]
span=torch.linalg.qr(torch.stack([u,v],1),mode='reduced').Q
def project(M):
 t=span.T@M
 return M-span@t-t.T@span.T+span@(t@span)@span.T
Ap=project(A[:-1,:-1]);Bp=project(B[:-1,:-1])
K=torch.stack([torch.stack([v@v,v@u]),torch.stack([u@v,u@u])])
eigK=torch.linalg.eigvalsh(K);assert eigK.min()>0
eigenvalues=[torch.linalg.eigvalsh(M).abs().sort(descending=True).values for M in [Ap,Bp]]
teacher=prepare_teacher(A,B);rows=[]
for rank in [8,16,32,64,128,256,512]:
 tails=[ev[rank:].square().sum() for ev in eigenvalues]
 terms=torch.stack([eigK[0]*sum(tails),(K[0,0]-K[0,1].square()/K[1,1])*tails[0],(K[1,1]-K[0,1].square()/K[0,0])*tails[1]])*2
 rows.append(dict(rank_per_source=rank,gaussian_variation_error_lower_bound=float((terms.max()/teacher['variance']).sqrt()),squared_bound_candidates=terms.tolist()))
out=dict(affine_gram=K.tolist(),records=rows,scope='Numerical lower bound for fixed affine vectors and rank-r quadratic forms in the current exact Gaussian numerator objective. Uses only an orthogonal cubic Hermite sector. Does not bound arbitrary DAGs, changed affine terms, native-function error or semantic circuit complexity.')
(PTH/'GAUSSIAN_AFFINE_RANK_BOUND_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
