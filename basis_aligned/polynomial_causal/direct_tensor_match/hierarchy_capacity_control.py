import itertools,json,math
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1715);checks=[]
for d in [4,8,16]:
 eye=torch.eye(d);H=(torch.einsum('ij,kl->ijkl',eye,eye)+torch.einsum('ik,jl->ijkl',eye,eye)+torch.einsum('il,jk->ijkl',eye,eye))/3;ev=torch.linalg.eigvalsh(H.reshape(d*d,d*d));m=d*(d+1)//2;expected=torch.tensor([0.]*(d*d-m)+[2/3]*(m-1)+[(d+2)/3]);spectrum=float((ev-expected).abs().max());assert spectrum<1e-11
 x=torch.randn(11,d);got=torch.einsum('ijkl,ni,nj,nk,nl->n',H,x,x,x,x);ref=x.square().sum(1).square();error=float((got-ref).norm()/ref.norm());assert error<1e-11
 ranks=[]
 for r in [1,2]:
  factors=[torch.randn(r,d) for _ in range(4)];raw=torch.einsum('ai,aj,ak,al->ijkl',*factors);S=sum(raw.permute(p) for p in itertools.permutations(range(4)))/24;rank=int(torch.linalg.matrix_rank(S.reshape(d*d,d*d),rtol=1e-10));assert rank<=6*r;ranks.append(dict(atoms=r,unfolding_rank=rank,rank_bound=6*r))
 checks.append(dict(d=d,spectrum_max_absolute_error=spectrum,polynomial_relative_error=error,cp_rank_checks=ranks))
rows=[]
for d,r in itertools.product([16,32,128,1152],[1,2,4,8,32]):
 m=d*(d+1)//2;allowed=min(6*r,m);tail=max(0,m-allowed)*4/9;norm=d*(d+2)/3;rows.append(dict(d=d,atoms=r,symmetric_unfolding_rank=m,cp_unfolding_rank_bound=6*r,relative_frobenius_lower_bound=math.sqrt(tail/norm),shared_scalar_multiplications=d+1,shared_scalar_additions=d-1,coordinate_CP_expansion_terms=m))
out=dict(checks=checks,bounds=rows,scope='Known radial quartic, symmetric coefficient objective. Rank relaxation is a rigorous necessary error bound, not sufficient CP capacity or a native-model statement. Shared q computation exact without symmetric coefficient storage.')
(P/'HIERARCHY_CAPACITY_CONTROL_V1.json').write_text(json.dumps(out,indent=2)+'\n');print('checks',checks);print('eight-atom bounds',[(r['d'],r['relative_frobenius_lower_bound']) for r in rows if r['atoms']==8])
