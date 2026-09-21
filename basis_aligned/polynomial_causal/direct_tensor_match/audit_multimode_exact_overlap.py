"""Exact common-linear-bank lower bound and overlap diagnostics for frozen programs.
If a quadratic pair's core pencil is nonsingular, its matrix ranges span the
reader subspace. Any exact common linear bank for all pairs must contain their
union. This does not lower-bound general nonlinear arithmetic circuit size.
"""
from pathlib import Path
import torch,json
from itertools import combinations
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
programs=torch.load(P/'MULTIMODE_PAIR_BASELINES_V1.pt',weights_only=True)
root=torch.load(P/'EXPANDED_SOURCE_METRIC_GEOMETRY_V1.pt',weights_only=True)['root']
bases=[];pair_checks=[]
for key,p in programs.items():
 reader=root@p['shared_reader'];U,S,V=torch.linalg.svd(reader,full_matrices=False)
 bases.append(U);n=reader.shape[1];cores=[]
 for out in range(2):
  G=torch.zeros(n,n,dtype=torch.float64)
  for pos,(i,j,kind) in enumerate(p['product_indices'].T.tolist()):
   w=p['product_weights'][pos,out]
   if kind==0:G[i,i]+=w
   elif kind==1:G[i,i]+=w;G[j,j]-=w
   else:G[i,j]+=w/2;G[j,i]+=w/2
  cores.append(G)
 # Reader's orthonormal basis removes the compiler's gauge from conditioning.
 transform=S[:,None]*V
 pencil=transform@(cores[0]+cores[1])@transform.T
 singular=torch.linalg.svdvals(pencil)
 pair_checks.append(dict(mode=int(key)+1,reader_min_over_max=float(S[-1]/S[0]),pencil_min_over_max=float(singular[-1]/singular[0]),reader_numerical_rank=int((S>S[0]*1e-10).sum()),pencil_numerical_rank=int((singular>singular[0]*1e-10).sum())))
sv=torch.linalg.svdvals(torch.cat(bases,dim=1))
pairwise=[]
for i,j in combinations(range(3),2):
 cosines=torch.linalg.svdvals(bases[i].T@bases[j])
 pairwise.append(dict(modes=[i+1,j+1],cosine_max=float(cosines.max()),cosine_median=float(cosines.median()),cosine_min=float(cosines.min()),cosine_above_099=int((cosines>.99).sum()),numerical_exact_intersection=int((cosines>1-1e-10).sum())))
rank=int((sv>sv[0]*1e-10).sum())
valid=all(p['reader_numerical_rank']==256 and p['pencil_numerical_rank']==256 for p in pair_checks)
out=dict(pair_checks=pair_checks,pairwise_principal_cosines=pairwise,union_singular_min=float(sv[-1]),union_singular_max=float(sv[0]),numerical_union_rank=rank,rank_tolerance_relative=1e-10,exact_shared_bank_dimension_lower_bound=rank if valid else None,
 dense_input_cost_separate=3*1152*256,dense_common_input_cost_lower_bound=1152*rank,dense_common_plus_inner_cost=1152*rank+rank*3*256,
 interpretation='Numerically certified full subspace union only at stated tolerance. Exact linear bank must contain quadratic ranges when pencil nonsingular. No exact feature overlap found if union full rank; approximation/nonlinear sharing remain possible. Dense cost assumes generic matrices, not a lower bound on sparse arithmetic operations.')
(P/'MULTIMODE_EXACT_OVERLAP_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
