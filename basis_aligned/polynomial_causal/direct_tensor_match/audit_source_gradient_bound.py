"""Right-unfolding rank bound for source-gradient metric, not native phi."""
from pathlib import Path
import torch,json
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);T=d['teacher'][:4];H=d['inverse_root']@d['inverse_root']
K=sum(Q@H@Q for Q in T);e=torch.linalg.eigvalsh((K+K.T)/2).flip(0);assert e.min()>-1e-10*e.max();e=e.clamp_min(0)
rows=[dict(input_rank=r,relative_source_gradient_lower_bound=float((e[r:].sum()/e.sum()).sqrt())) for r in [256,384,512,640]]
# Explicit tiny stacked weighted unfolding verifies the algebra and tail identity.
g=torch.Generator().manual_seed(755);A=torch.randn(3,8,8,generator=g,dtype=torch.double);A=(A+A.transpose(-1,-2))/2;B=torch.randn(8,8,generator=g,dtype=torch.double);HH=B@B.T+torch.eye(8,dtype=torch.double)
u,V=torch.linalg.eigh(HH);sqrtH=(V*u.sqrt())@V.T;stack=torch.cat([sqrtH@Q for Q in A]);sv=torch.linalg.svdvals(stack);eig=torch.linalg.eigvalsh(sum(Q@HH@Q for Q in A)).flip(0)
replay=float((sv.square()-eig).abs().max()/eig.max());assert replay<1e-10
out=dict(records=rows,explicit_unfolding_replay=replay,scope='Necessary right-mode rank bound. Each256mixed-product symmetric matrix lies in common span[L,R] of dimension<=512. Relaxation drops left-mode symmetry/product restrictions, so it need not be attainable. Source-gradient metric only; no native component/causal error bound.')
(P/'SOURCE_GRADIENT_BOUND_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
