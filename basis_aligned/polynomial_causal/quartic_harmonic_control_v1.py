from pathlib import Path
import itertools,json,torch,numpy as np
from quartic_repeated_input_v1 import dense_trace
from quartic_harmonic_v1 import lower,evaluate_lower
P=Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(91730);d=3
raw=torch.randn(2,d,d,d,d);t=torch.stack([raw.permute(0,*[x+1 for x in perm]) for perm in itertools.permutations(range(4))]).mean(0)
a=dense_trace(t);m,c=lower(a);eye=torch.eye(d)
def sym(a,b):
 return (torch.einsum('oab,cd->oabcd',a,b)+torch.einsum('oac,bd->oabcd',a,b)+torch.einsum('oad,bc->oabcd',a,b)+torch.einsum('ab,ocd->oabcd',b,a)+torch.einsum('ac,obd->oabcd',b,a)+torch.einsum('ad,obc->oabcd',b,a))/6
h4=t-sym(m,eye)-sym(c[:,None,None]*eye,eye)
e0=float(dense_trace(h4).norm()/t.norm());e1=float(m.diagonal(dim1=-2,dim2=-1).sum(-1).norm()/m.norm())
nodes,weights=np.polynomial.hermite.hermgauss(5);nodes=torch.tensor(nodes)*2**.5;weights=torch.tensor(weights)/np.pi**.5;ids=torch.tensor(list(itertools.product(range(5),repeat=d)));x=nodes[ids];qw=weights[ids].prod(-1)
p0,p2=evaluate_lower(a,x);p4=torch.einsum('oabcd,na,nb,nc,nd->no',h4,x,x,x,x);values=torch.stack([p0,p2,p4],-1)
g=torch.einsum('noi,noj,n->oij',values,values,qw);diag=g.diagonal(dim1=-2,dim2=-1);normal=g/diag.sqrt()[:,:,None]/diag.sqrt()[:,None,:];off=float((normal-torch.eye(3)).abs().max())
r=dict(pred_a=max(e0,e1)<=1e-10,pred_b=off<=1e-10,harmonic_trace_error=e0,quadratic_trace_error=e1,normalized_quadrature_offdiagonal=off)
p=P/'QUARTIC_HARMONIC_V1_CONTROL.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(r);assert r['pred_a'] and r['pred_b']
