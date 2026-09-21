"""Independent lower-bound check and feasible dense-core span relaxations.
Dense-core projections are mathematical controls, not cost-matched circuits.
"""
from pathlib import Path
import json,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def span_error(T,U):
 core=torch.einsum('ia,oij,jb->oab',U,T,U)
 return float((1-core.square().sum()/T.square().sum()).clamp_min(0).sqrt())
checks=[]
for seed in range(5):
 g=torch.Generator().manual_seed(10200+seed);q=torch.randn(3,12,12,dtype=torch.float64,generator=g);q=(q+q.transpose(-1,-2))/2
 r=5;S=torch.einsum('oij,okj->ik',q,q);e,U=torch.linalg.eigh(S);tail=float(e[:-r].sum()/q.square().sum());unfold=q.permute(1,0,2).reshape(12,-1);sv=torch.linalg.svdvals(unfold);svtail=float(sv[r:].square().sum()/q.square().sum())
 random=torch.linalg.qr(torch.randn(12,r,dtype=q.dtype,generator=g)).Q;error=span_error(q,random);assert abs(tail-svtail)<1e-12 and error**2>=tail-1e-12
 # Simultaneously diagonal forms attain the one-mode bound with selected coordinate axes.
 diagonal=torch.diag_embed(torch.randn(3,12,dtype=q.dtype,generator=g));energy=diagonal.square().sum((0,2));order=energy.argsort(descending=True);basis=torch.eye(12,dtype=q.dtype)[:,order[:r]];sharp=span_error(diagonal,basis)**2;expected=float(energy[order[r:]].sum()/energy.sum());assert abs(sharp-expected)<1e-12
 checks.append(dict(seed=seed,svd_gram_tail_replay=abs(tail-svtail),random_span_squared_error=error**2,lower_bound_squared=tail,diagonal_sharpness_replay=abs(sharp-expected)))
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);energies=Q.square().sum((-1,-2)).reshape(3,2).sum(1);T=Q/energies.repeat_interleave(2).sqrt()[:,None,None]
e,U=torch.linalg.eigh(torch.einsum('oij,okj->ik',T,T));norm=T.square().sum();rows=[]
for rank in [766,1024,1152]:
 rows.append(dict(span_width=rank,necessary_native_error=float((e[:-rank].sum()/norm).clamp_min(0).sqrt()) if rank<1152 else 0.,feasible_dense_core_projection_error=span_error(T,U[:,-rank:]),full_symmetric_core_entries=6*rank*(rank+1)//2))
meta=json.loads((P/'EMPIRICAL_SOURCE_DIRECTIONS_V1.json').read_text());p=torch.load(P/'EMPIRICAL_SOURCE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True)[meta['winners']['1']];readers=torch.cat([p['left_reader'],p['right_reader'],p['square_reader']],1);A,s,_=torch.linalg.svd(readers,full_matrices=False);rank=int((s>1e-10*s[0]).sum());fixed=dict(reader_span_rank=rank,reader_condition_number=float(s[0]/s[-1]),best_dense_core_error_in_current_span=span_error(T,A[:,:rank]))
out=dict(toy_checks=checks,native_span_relaxations=rows,current_dictionary_span=fixed,scope='Native isotropic equal-pair coefficient geometry. Dense-core projections allow arbitrary quadratic forms within the chosen input span, so their storage/product prices exceed the sparse dictionary. HOSVD span is feasible but not asserted globally optimal for a symmetric two-mode projection. This analysis separates span and atom restrictions; it does not validate circuits.')
(P/'SOURCE_SPAN_RELAXATION_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
