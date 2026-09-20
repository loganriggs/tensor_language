"""Exact fourth mixed moments of Gaussian quadratic forms by polarization."""
import itertools,math,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def fourth(Q,m):
 power=Q;k=[]
 for r in range(1,5):
  value=power.diagonal(dim1=-2,dim2=-1).sum(-1)+r*torch.einsum('i,...ij,j->...',m,power,m);k.append(2**(r-1)*math.factorial(r-1)*value)
  if r<4:power=power@Q
 a,b,c,d=k;return a**4+6*a*a*b+3*b*b+4*a*c+d

def root_moments(Q,m,pairs=None):
 n=len(Q);pairs=list(itertools.combinations_with_replacement(range(n),2)) if pairs is None else list(pairs);indices=sorted({tuple(sorted((i,j,k,l))) for i,j in pairs for k,l in pairs});signs=torch.tensor(list(itertools.product([-1.,1.],repeat=4)),dtype=Q.dtype,device=Q.device);mixed={}
 for ix in indices:
  forms=torch.einsum('sk,kij->sij',signs,Q[list(ix)]);mixed[ix]=(fourth(forms,m)*signs.prod(1)).sum()/384
 mean=Q.diagonal(dim1=-2,dim2=-1).sum(-1)+torch.einsum('i,vij,j->v',m,Q,m);qm=torch.einsum('vij,j->vi',Q,m);cov=2*torch.einsum('aij,bij->ab',Q,Q)+4*qm@qm.T;second=cov+mean[:,None]*mean[None,:];phi_mean=torch.stack([second[i,j] for i,j in pairs]);raw=torch.stack([torch.stack([mixed[tuple(sorted((i,j,k,l)))] for k,l in pairs]) for i,j in pairs]);G=raw-phi_mean[:,None]*phi_mean[None,:];return phi_mean,(G+G.T)/2

def check():
 from dag_square_optimizer import rule
 torch.manual_seed(2048);Q=torch.randn(4,3,3,dtype=torch.float64);Q=(Q+Q.transpose(-1,-2))/2;m=torch.randn(3,dtype=torch.float64);mean,G=root_moments(Q,m);x,w=rule(5);x=x[:,:3]+m;q=torch.einsum('ni,vij,nj->nv',x,Q,x);i,j=torch.triu_indices(4,4);phi=q[:,i]*q[:,j];directmean=(w[:,None]*phi).sum(0);center=phi-directmean;directG=center.T@(w[:,None]*center);errors=dict(mean_relative_error=float((mean-directmean).norm()/directmean.norm()),covariance_relative_error=float((G-directG).norm()/directG.norm()));assert max(errors.values())<1e-10;return errors

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);oracle=check();core=torch.load(P/'QUARTIC_BANK_CORE_V1.pt',weights_only=True);fits=torch.load(P/'BANK_WIDTH_FRONTIER_V1.pt',weights_only=True);result=json.loads((P/'BANK_WIDTH_FRONTIER_V1.json').read_text());w=next(r for r in result['winners'] if r['width']==6);f=fits['allfits'][(6,w['optimizer'],w['lr'],w['seed'])];a,b,c=f['a'],f['b'],f['c'];C=torch.linalg.solve(core['sqrt_root_sensitivity'],c);Q=torch.einsum('vk,kij->vij',C,.5*(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:]));mu=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['mean'].double();m=core['input_mapback']@mu;mean,G=root_moments(Q,m);ev,E=torch.linalg.eigh(G);assert ev.min()>-1e-10*ev.max();sqrtG=(E*ev.clamp_min(0).sqrt())@E.T;program={k:t.double() for k,t in fits['programs'][6].items()};Z=program['Z'];sv=torch.linalg.svdvals(Z@sqrtG).square();bound=sv.cumsum(0)/sv.sum();out=dict(oracle=oracle,root_covariance_eigenvalues=ev.tolist(),condition=float(ev.max()/ev.min()),centered_root_output_rank_retention_bounds=bound.tolist(),root_centered_energy=float(sv.sum()),scope='Exact degree8 Gaussian-input moments of four quadratic bank features. Bank output is not assumed Gaussian. Native covariance uses pre-export FP64 bank factors; FP32 archive equivalence needs separate check before fitting.');torch.save(dict(Q=Q,m=m,mean=mean,covariance=G,sqrt_covariance=sqrtG,root_writer=Z),P/'ROOT_FUNCTION_METRIC_V1.pt');(P/'ROOT_FUNCTION_METRIC_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
