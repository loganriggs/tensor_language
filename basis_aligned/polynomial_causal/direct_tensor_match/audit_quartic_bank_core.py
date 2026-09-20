"""Exact compressed tensor for the four learned quadratic bank features.
Root sensitivity is a declared local weighting, not full quartic loss.
"""
import json
from pathlib import Path
import torch
from gaussian_quartic_mean import quadratic_moments
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);s={k:v.double() for k,v in torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True)['programs'][8].items()};panel=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];mu=panel['mean'].double();M=panel['covariance'].double();L=torch.linalg.cholesky(M);U,V=s['U'],s['V'];a=U.flatten(0,1)@L;b=V.flatten(0,1)@L;basis,_=torch.linalg.qr(torch.cat([a,b]).T,mode='reduced');A=a@basis;B=b@basis;terms=.5*(A[:,:,None]*B[:,None,:]+B[:,:,None]*A[:,None,:]);core=terms.reshape(4,4,32,32).sum(1)
 m,G=quadratic_moments(U.flatten(0,1),V.flatten(0,1),mu,M);m=m.reshape(4,4).sum(1);G=G.reshape(4,4,4,4).sum((1,3));second=G+m[:,None]*m[None,:];writer=s['W']@s['Z'];i,j=torch.triu_indices(4,4);Z=torch.zeros(len(writer),4,4,dtype=torch.float64)
 for k,(u,v) in enumerate(zip(i,j)):
  if u==v:Z[:,u,v]=writer[:,k]
  else:Z[:,u,v]=writer[:,k]/2;Z[:,v,u]=writer[:,k]/2
 K=4*torch.einsum('vaj,vbk,jk->ab',Z,Z,second);eig,Q=torch.linalg.eigh(K);sqrtK=(Q*eig.clamp_min(0).sqrt())@Q.T;weighted=torch.einsum('ab,bij->aij',sqrtK,core);sv=torch.linalg.svdvals(weighted.permute(1,0,2).reshape(32,-1));energy=sv.square().cumsum(0)/sv.square().sum();torch.manual_seed(2041);x=torch.randn(17,1152,dtype=torch.float64);direct=((x@a.T)*(x@b.T)).reshape(17,4,4).sum(2);compressed=torch.einsum('ni,gij,nj->ng',x@basis,core,x@basis);replay=float((direct-compressed).norm()/direct.norm());assert replay<1e-12
 cancellation=float(torch.einsum('ag,gkij,ag,gkij->',sqrtK,terms.reshape(4,4,32,32),sqrtK,terms.reshape(4,4,32,32))/weighted.square().sum())
 result=dict(core_shape=list(core.shape),primitive_products=16,core_replay_error=replay,root_sensitivity_eigenvalues=eig.tolist(),root_sensitivity_condition=float(eig.max()/eig.min()),input_mode_cumulative_energy=energy.tolist(),weighted_primitive_component_energy_ratio=cancellation,scope='Exact 4x32x32 bank coefficient tensor under centered covariance input scaling. Root sensitivity K=E[J(q)^T J(q)] is a local surrogate; compressing it need not preserve full quartic behavior or Gaussian function loss with nonzero mean. Final quartic and fresh response checks required.')
 torch.save(dict(core=core,weighted_core=weighted,input_mapback=torch.linalg.solve_triangular(L.T,basis,upper=True).T,root_sensitivity=K,sqrt_root_sensitivity=sqrtK,A=A,B=B),P/'QUARTIC_BANK_CORE_V1.pt');(P/'QUARTIC_BANK_CORE_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
