"""Exact top-K distinct-pair selection for a fixed orthonormal Tucker input basis.
Group sparsity over input pairs counts each computed product once across outputs.
This is not an optimized Tucker basis or a general DAG search.
"""
from pathlib import Path
import json,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def select(T,U,k):
 C=torch.einsum('ia,oij,jb->oab',U,T,U);ij=torch.triu_indices(U.shape[1],U.shape[1]);multiplicity=torch.where(ij[0]==ij[1],1.,2.).to(T)
 value=C[:,ij[0],ij[1]];energy=value.square().sum(0)*multiplicity
 selected=energy.topk(k).indices;pair=ij[:,selected];weights=(value[:,selected]*multiplicity[selected]).T
 used,inverse=torch.unique(pair,sorted=True,return_inverse=True);basis=U[:,used];pair=inverse.reshape(2,k)
 raw=torch.einsum('ik,ko,jk->oij',basis[:,pair[0]],weights,basis[:,pair[1]]);hat=(raw+raw.transpose(-1,-2))/2
 explicit=float((hat-T).norm()/T.norm());implicit=float((1-energy[selected].sum()/T.square().sum()).clamp_min(0).sqrt())
 return basis,pair,weights,hat,explicit,implicit
checks=[]
for seed in range(5):
 g=torch.Generator().manual_seed(10300+seed);U=torch.linalg.qr(torch.randn(12,8,dtype=torch.float64,generator=g)).Q;ij=torch.triu_indices(8,8);k=seed+1;chosen=torch.randperm(len(ij[0]),generator=g)[:k];weights=torch.randn(k,3,dtype=U.dtype,generator=g)
 raw=torch.einsum('ik,ko,jk->oij',U[:,ij[0,chosen]],weights,U[:,ij[1,chosen]]);T=(raw+raw.transpose(-1,-2))/2
 basis,pair,w,hat,error,_=select(T,U,k);x=torch.randn(64,12,dtype=U.dtype,generator=g);s=x@basis;y=(s[:,pair[0]]*s[:,pair[1]])@w;ref=torch.einsum('ni,oij,nj->no',x,T,x);replay=float((y-ref).norm()/ref.norm());assert max(error,replay)<1e-12
 checks.append(dict(seed=seed,distinct_products=k,active_input_features=basis.shape[1],dense_recovery_error=error,execution_replay=replay,core_coefficients=w.numel()))
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);Q=torch.stack([q for p in d['pairs'] for q in p['Qs']]);energy=Q.square().sum((-1,-2)).reshape(3,2).sum(1);scale=energy.repeat_interleave(2).sqrt();T=Q/scale[:,None,None]
_,U=torch.linalg.eigh(torch.einsum('oij,okj->ik',T,T));meta=json.loads((P/'EMPIRICAL_SOURCE_DIRECTIONS_V1.json').read_text());parent=torch.load(P/'EMPIRICAL_SOURCE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True)[meta['winners']['1']];A,_,_=torch.linalg.svd(torch.cat([parent['left_reader'],parent['right_reader'],parent['square_reader']],1),full_matrices=False)
rows=[]
for name,basis in [('native_HOSVD766',U[:,-766:]),('current_span_SVD766',A)]:
 for k in (399,592):
  B,pair,w,hat,error,implicit=select(T,basis,k);assert abs(error-implicit)<1e-10
  Qhat=hat*scale[:,None,None];delta=Q-Qhat;linear=2*torch.einsum('oij,j->io',delta,d['mu']);bias=torch.einsum('ij,oji->o',d['old_covariance'],delta)-torch.einsum('i,oij,j->o',d['mu'],delta,d['mu'])
  ids=d['indices'];z=d['z'][ids];h=d['h'][ids];s=z@B;reads=(s[:,pair[0]]*s[:,pair[1]])@(w*scale)+z@linear+bias
  dense=torch.einsum('ni,oij,nj->no',z,Qhat,z)+z@linear+bias;replay=float((reads-dense).norm()/dense.norm());assert replay<1e-10
  rms=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();values=[]
  for j,p in enumerate(d['pairs']):
   phi=((h@p['a']-.5*reads[:,2*j])/rms-p['alpha'])*(reads[:,2*j+1]/rms-p['beta']);truth=p['truth'][ids];values.append(float((phi-truth).norm()/(truth-truth.mean()).norm()))
  price=B.numel()+w.numel()+linear.numel()+bias.numel()+3*1152+6+1152
  rows.append(dict(basis=name,distinct_products=k,eligible_input_features=basis.shape[1],active_input_features=B.shape[1],core_coefficients=w.numel(),stored_float_coefficients=price,pair_index_integers=pair.numel(),native_isotropic_equal_pair_error=error,energy_replay=abs(error-implicit),execution_replay=replay,per_mode_errors=values))
out=dict(planted_checks=checks,records=rows,scope='Native isotropic coefficient-optimal group-sparse core in each fixed orthonormal basis. Product pair shared across all6outputs, unused feature columns removed before price. Basis optimization, low-rank core slices and further graph factoring not performed. Nativez/h and existing affine convention retained; opened448 is diagnostic, not fresh.')
(P/'SPARSE_PAIR_CORE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
