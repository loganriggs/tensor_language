import itertools,json
from pathlib import Path
import torch
from implicit_quartic import entries
from quartic_cp import cp_inner,teacher_cross
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1640);d=5;teacher=[torch.randn(*s) for s in [(3,4),(4,5),(4,5),(5,7),(7,d),(7,d)]];C=torch.randn(3,4,requires_grad=True);factors=[torch.randn(4,d,requires_grad=True) for _ in range(4)];idx=torch.tensor(list(itertools.product(range(d),repeat=4)));H=entries(*teacher,idx).T.reshape(3,d,d,d,d);raw=torch.einsum('va,ai,aj,ak,al->vijkl',C,*factors);S=sum(raw.permute((0,)+tuple(i+1 for i in p)) for p in itertools.permutations(range(4)))/24
norm=cp_inner(C,factors,C,factors);cross=teacher_cross(teacher,C,factors);refnorm=S.square().sum();refcross=(H*S).sum();value=max(float((abs(norm-refnorm)/refnorm).detach()),float((abs(cross-refcross)/refcross.abs()).detach()));params=[C]+factors;g1=torch.autograd.grad(norm-2*cross,params,retain_graph=True);g2=torch.autograd.grad(refnorm-2*refcross,params);gradient=max(float((a-b).norm()/b.norm()) for a,b in zip(g1,g2));assert max(value,gradient)<1e-11
out=dict(value_relative_error=value,gradient_relative_error=gradient,scope='Exact full input-symmetry CP quartic self norm and native two-layer cross contraction; independently dense24permutation checked. Teacher norm is a constant irrelevant to parameter gradient; exact normalized reconstruction error still needs teacher norm.')
(P/'QUARTIC_CP_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
