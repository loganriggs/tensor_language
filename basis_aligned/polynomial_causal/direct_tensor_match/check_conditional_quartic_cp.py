"""Independent Gauss-Hermite integration controls for conditional program."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from conditional_quartic_cp import construct,evaluate
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);dtype=torch.float64
 nodes,weights=np.polynomial.hermite.hermgauss(3);nodes=torch.tensor(nodes*2**.5,dtype=dtype);weights=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype);rows=[]
 for seed,name in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
  torch.manual_seed(16000+seed);f=[torch.randn(5,4,dtype=dtype) for _ in range(4)];C=torch.randn(2,5,dtype=dtype);S=torch.randn(4,4,dtype=dtype)+4*torch.eye(4,dtype=dtype);mu=torch.randn(4,dtype=dtype);Q=torch.linalg.qr(torch.randn(4,4,dtype=dtype)).Q
  if name=='shared_input':f[1]=f[0].clone()
  if name=='shared_output':C[1]=C[0]
  if name=='squares':f[2:]=[a.clone() for a in f[:2]]
  if name=='cancellation':
   for a in f:a[1]=a[0]
   C[:,1]=-C[:,0]
  for rank in [1,2,4]:
   V=Q[:,:rank];program=construct(f,C,S,mu,V);eta=torch.randn(7,rank,dtype=dtype);idx=torch.tensor(list(itertools.product(range(3),repeat=4-rank)),dtype=torch.long).reshape(-1,4-rank) if rank<4 else torch.empty(1,0,dtype=torch.long)
   z=eta[:,None,:]@V.T+nodes[idx][None,:,:]@Q[:,rank:].T;xx=z@S.T+mu;phi=torch.stack([xx@a.T for a in f]).prod(0);w=weights[idx].prod(-1);ref=((phi@C.T)*w[None,:,None]).sum(1);x=eta@V.T@S.T+mu;pred=evaluate(program,x);error=float((pred-ref).norm()/ref.norm());assert error<1e-11,(name,rank,error)
   rows.append(dict(family=name,input_rank=rank,quadrature_error=error))
 (P/'CONDITIONAL_QUARTIC_CP_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS15quadraturecontrols',max(r['quadrature_error'] for r in rows))
if __name__=='__main__':main()
