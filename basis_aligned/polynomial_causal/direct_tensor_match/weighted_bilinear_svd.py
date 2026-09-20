"""Optimal matrix-rank approximation for a separable second-moment metric."""
import torch

def roots(moment,ridge=1e-6):
 d=moment.shape[0];reg=moment+ridge*moment.trace()/d*torch.eye(d,dtype=moment.dtype,device=moment.device)
 e,V=torch.linalg.eigh(reg);assert float(e.min())>0
 return (V*e.sqrt()[None,:])@V.T,(V*e.rsqrt()[None,:])@V.T

def decompose(K,left,right):
 Sn,In=left;Sm,Im=right;U,s,Vh=torch.linalg.svd(Sn@K@Sm,full_matrices=False)
 return (In@U)*s.sqrt()[None,:],(Im@Vh.T)*s.sqrt()[None,:],s

def toy_check():
 gen=torch.Generator().manual_seed(261050);rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
 n=rand(19,5);m=rand(19,4);K=rand(5,4);Mn=n.T@n/len(n);Mm=m.T@m/len(m);left=roots(Mn);right=roots(Mm);A,B,s=decompose(K,left,right);full=float((A@B.T-K).norm()/K.norm());assert full<1e-12
 r=2;delta=K-A[:,:r]@B[:,:r].T;error=(left[0]@delta@right[0]).square().sum();tail=s[r:].square().sum();spectral=float((error-tail).abs()/tail);assert spectral<1e-12
 # Uniform independent empirical pairs match the unregularized moment metric.
 values=torch.einsum('bi,ij,cj->bc',n,delta,m);direct=values.square().mean();exact=torch.trace(delta.T@Mn@delta@Mm);metric=float((direct-exact).abs()/exact);assert metric<1e-12
 return dict(full_replay=full,optimal_tail_replay=spectral,independent_pair_metric_replay=metric)
if __name__=='__main__':
 import json
 from pathlib import Path
 r=toy_check();Path(__file__).with_name('WEIGHTED_BILINEAR_SVD_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
