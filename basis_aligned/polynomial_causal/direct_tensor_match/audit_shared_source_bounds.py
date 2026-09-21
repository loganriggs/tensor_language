"""Shared-input quadratic dictionary coefficient lower bound and HOSVD upper bound.
For symmetric M_j, any common rank-k input projection has two-sided squared error
at least the sum of eigenvalues beyond k of sum_j M_j M_j.T. The leading-space
construction has error <= twice that lower bound. This bounds coefficient error,
not native behavioral error, and says nothing about arbitrary arithmetic DAGs.
"""
from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);e=torch.load(p/'MIDPOINT_SOURCE_COVARIANCE_PROGRAMS_V1.pt',weights_only=True)['covariance_16'];data=torch.load(p/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);delta=data['z'].flatten(0,1).double()-e['mu'];cov=delta.T@delta/len(delta);ev,V=torch.linalg.eigh(cov);root=(V*ev.clamp_min(0).sqrt())@V.T;fold=torch.load(p/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.pt',weights_only=True);results={}
for target,Qs in [('frozen_rank16',[(e[k+'_reader']*e[k+'_eigenvalues'])@e[k+'_reader'].T for k in ['a','b']]),('original_quadratic',[fold[k]['matrix'] for k in ['a','b']])]:
 Ms=[root@Q@root for Q in Qs];gram=sum(M@M.T for M in Ms);ev,P=torch.linalg.eigh(gram);order=ev.argsort(descending=True);ev=ev[order].clamp_min(0);P=P[:,order];norm=sum(float(M.square().sum()) for M in Ms);rows=[]
 for rank in [8,16,24,32]:
  basis=P[:,:rank];error=sum(float((M-basis@(basis.T@M@basis)@basis.T).square().sum()) for M in Ms);lower=float(ev[rank:].sum());assert error>=lower-1e-10*norm and error<=2*lower+1e-10*norm
  rows.append(dict(shared_rank=rank,relative_error_lower_bound=(lower/norm)**.5,construction_relative_error=(error/norm)**.5,construction_over_lower_bound=(error/lower)**.5 if lower>norm*1e-12 else None))
 results[target]=rows
out=dict(results=results,scope='Covariance-weighted coefficient norm of centered source quadratic residuals; linear and constant terms separate. Global lower bound within common-k-dimensional-input dictionaries, not all low-arithmetic-cost DAGs. Does not certify empirical fourth-moment or native-effect fidelity.');(p/'MIDPOINT_SHARED_SOURCE_BOUNDS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
