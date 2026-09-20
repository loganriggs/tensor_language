"""Small-tensor oracle for the optimal degree<=2 Gaussian quartic projection.
Native execution must use contractions; this dense oracle is only for small tests.
"""
import torch

def project(H,mu,M):
 constant0=torch.einsum('vijkl,i,j,k,l->v',H,mu,mu,mu,mu)
 mean=constant0+6*torch.einsum('vijkl,i,j,kl->v',H,mu,mu,M)+3*torch.einsum('vijkl,ij,kl->v',H,M,M)
 linear=4*torch.einsum('vijkl,j,k,l->vi',H,mu,mu,mu)+12*torch.einsum('vijkl,j,kl->vi',H,mu,M)
 quadratic=6*torch.einsum('vijkl,k,l->vij',H,mu,mu)+6*torch.einsum('vijkl,kl->vij',H,M)
 constant=mean-torch.einsum('vij,ij->v',quadratic,M)
 return constant,linear,quadratic,mean

def check():
 import json
 from pathlib import Path
 from implicit_quartic import entries
 from dag_square_optimizer import rule
 from centered_quartic import degree_terms
 from gaussian_quartic_mean import native_mean
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(2010);teacher=[torch.randn(*shape) for shape in [(2,3),(3,5),(3,5),(5,6),(6,4),(6,4)]];indices=torch.cartesian_prod(*[torch.arange(4)]*4);H=entries(*teacher,indices).T.reshape(2,4,4,4,4);mu=torch.randn(4);raw=torch.randn(4,4);M=raw@raw.T+.2*torch.eye(4);L=torch.linalg.cholesky(M);xx,w=rule(5);delta=xx[:,:4]@L.T;x=delta+mu;c,J,Q,mean=project(H,mu,M);terms=degree_terms(teacher,x,mu);truth=terms.sum(1);prediction=c+delta@J.T+torch.einsum('vij,ni,nj->nv',Q,delta,delta);residual=truth-prediction;basis=torch.cat([torch.ones(len(x),1),delta,(delta[:,:,None]*delta[:,None,:]-M).flatten(1)],1);correlation=(basis.T@(w[:,None]*residual));scale=(w[:,None]*truth.square()).sum().sqrt()*(w[:,None]*basis.square()).sum().sqrt();orthogonal=float(correlation.norm()/scale);assert orthogonal<1e-12
 f=lambda z:native_mean(teacher,z,M);jac=torch.func.jacrev(f)(mu);hessian=torch.func.hessian(f)(mu)/2;jerr=float((jac-J).norm()/J.norm());qerr=float((hessian-Q).norm()/Q.norm());assert max(jerr,qerr)<1e-12
 def error(p):return float(((w[:,None]*(truth-p).square()).sum()/(w[:,None]*truth.square()).sum()).sqrt())
 taylor=terms[:,:3].sum(1);corrected=taylor+mean-(w[:,None]*taylor).sum(0);result=dict(projected_relative_error=error(prediction),taylor_relative_error=error(taylor),mean_corrected_taylor_relative_error=error(corrected),residual_orthogonality=orthogonal,mean_gradient_linear_error=jerr,half_mean_hessian_quadratic_error=qerr,scope='Exact small quartic projection onto degree<=2 underN(mu,M), independentquadrature andnativeGaussianmean derivativechecks. DenseH isonlya toy oracle; no nativeprojection computed.');assert result['projected_relative_error']<=min(result['taylor_relative_error'],result['mean_corrected_taylor_relative_error'])+1e-12;(Path(__file__).resolve().parent/'GAUSSIAN_LOW_DEGREE_PROJECTION_CHECK_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':check()
