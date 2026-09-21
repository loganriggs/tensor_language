"""A lower bound for fixed-affine, rank-r source forms under Gaussian loss.
Project affine features onto Gaussian directions orthogonal to the source span.
That independent-context x centered-quadratic error is orthogonal to all
source-only terms and cannot be canceled by source-only quartic edits.
"""
import json
from pathlib import Path
import torch
from gaussian_quartic_lowrank import prepare_teacher
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
data=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True)
f=data['frames']['covariance'];A=f['A'];B=f['B'];J=f['J'][:-1]
# The whitening basis spans source coordinates in the Gaussian frame.
orth=float((J.T@J-torch.eye(J.shape[1],dtype=J.dtype)).norm()/J.shape[1]**.5)
raw_orth=orth
J=torch.linalg.qr(J,mode="reduced").Q
orth=float((J.T@J-torch.eye(J.shape[1],dtype=J.dtype)).norm()/J.shape[1]**.5)
assert orth<1e-10
projection_errors=[float((M[:-1,:-1]-J@(J.T@M[:-1,:-1]@J)@J.T).norm()/M[:-1,:-1].norm()) for M in [A,B]]
assert max(projection_errors)<1e-8
u=2*A[:-1,-1];v=2*B[:-1,-1]
up=u-J@(J.T@u);vp=v-J@(J.T@v)
K=torch.stack([torch.stack([vp@vp,vp@up]),torch.stack([up@vp,up@up])])
eigK=torch.linalg.eigvalsh(K);assert eigK.min()>0
teacher=prepare_teacher(A,B)
eigs=[torch.linalg.eigvalsh(J.T@M[:-1,:-1]@J).abs().sort(descending=True).values for M in [A,B]]
records=[]
for rank in [8,16,32,64,128,256,512]:
 tails=[ev[rank:].square().sum() for ev in eigs]
 schur_a=K[0,0]-K[0,1].square()/K[1,1]
 schur_b=K[1,1]-K[0,1].square()/K[0,0]
 terms=torch.stack([eigK[0]*sum(tails),schur_a*tails[0],schur_b*tails[1]])*2
 bound=terms.max()
 records.append(dict(rank_per_source=rank,gaussian_variation_error_lower_bound=float((bound/teacher['variance']).sqrt()),squared_bound_candidates=terms.tolist()))
out=dict(raw_source_frame_orthogonality_error=raw_orth,source_frame_orthogonality_error=orth,source_matrix_projection_errors=projection_errors,context_gram=K.tolist(),context_gram_eigenvalues=eigK.tolist(),records=records,scope='Lower bound only for two rank-r source quadratic forms with fixed exact centered affine branches and fixed outer product. Exact Gaussian jointframe; not arbitrary DAG, native distribution or semantic-circuit lower bound.')
(P/'GAUSSIAN_CONTEXT_RANK_BOUND_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
