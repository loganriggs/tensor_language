import json
from pathlib import Path
import torch
from quadratic_product_gram import product_gram
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1737)
raw=[torch.randn(n,4,4,requires_grad=True) for n in [2,2,3,3]];Q,R,S,T=[(a+a.transpose(-1,-2))/2 for a in raw]
def tensor(a,b):
 return (torch.einsum('aij,akl->aijkl',a,b)+torch.einsum('aij,akl->aijkl',b,a)+torch.einsum('aik,ajl->aijkl',a,b)+torch.einsum('aik,ajl->aijkl',b,a)+torch.einsum('ail,ajk->aijkl',a,b)+torch.einsum('ail,ajk->aijkl',b,a))/6
ref=tensor(Q,R).flatten(1)@tensor(S,T).flatten(1).T;got=product_gram(Q,R,S,T);error=float(((got-ref).norm()/ref.norm()).detach());g=torch.autograd.grad(got.square().sum(),raw,retain_graph=True);h=torch.autograd.grad(ref.square().sum(),raw);grad=max(float((a-b).norm()/b.norm()) for a,b in zip(g,h));assert max(error,grad)<1e-11
out=dict(value_relative_error=error,gradient_relative_error=grad,scope='Exact symmetrized quartic coefficient Gram of products of symmetric quadratic forms; independent dense expansion, indefinite noncommuting inputs. Native full-teacher contraction not yet executed.')
(P/'QUADRATIC_PRODUCT_GRAM_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
