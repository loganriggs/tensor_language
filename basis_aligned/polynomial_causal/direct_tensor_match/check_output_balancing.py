"""Check what output balancing can change in a shared bilinear dictionary.
Independent fixed-feature readouts are invariant when the ridge is balanced too.
Feature gradients change. Unbalanced ridge changes readouts via regularization.
"""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.manual_seed(220922);dtype=torch.float64;rows=[]
 for family in ['shared_sum','shared_product','dense_rank_one','sparse_pairs','cancellation']:
  x=torch.randn(97,6,dtype=dtype);a,b,c,d,e,f=x.T
  if family=='shared_sum':base=torch.stack([a*(b+c),d*(b+c),e*(b+c)],1)
  elif family=='shared_product':base=torch.stack([a*b*c,a*b*d,a*b*e],1)
  elif family=='dense_rank_one':base=torch.stack([(a+b+c)*(d+e+f),(a+b+c).square(),(d+e+f).square()],1)
  elif family=='sparse_pairs':base=torch.stack([a*b+c*d,a*f,c*e],1)
  else:base=torch.stack([(a*a+b*b).square()-(a*a-b*b).square(),a*a*b*b,(a*b).square()],1)
  y=base*torch.tensor([10.,1.,.1],dtype=dtype);w=1/y.square().mean(0);ridge=.1
  A=torch.randn(4,6,dtype=dtype,requires_grad=True);B=torch.randn(4,6,dtype=dtype,requires_grad=True)
  phi=(x@A.T)*(x@B.T);g=phi.T@phi/len(x);cross=phi.T@y/len(x);eye=torch.eye(4,dtype=dtype);C=torch.linalg.solve(g+ridge*eye,cross).detach()
  balanced=torch.stack([torch.linalg.solve(w[j]*g+ridge*w[j]*eye,w[j]*cross[:,j]) for j in range(3)],1)
  same=float(((balanced-C).norm()/C.norm()).detach());assert same<1e-12
  no_ridge_balance=torch.stack([torch.linalg.solve(w[j]*g+ridge*eye,w[j]*cross[:,j]) for j in range(3)],1)
  losses=(phi@C-y).square().mean(0)+ridge*C.square().sum(0)
  grad=torch.autograd.grad(losses.sum(),(A,B),retain_graph=True);bg=torch.autograd.grad((w*losses).sum(),(A,B))
  v=torch.cat([z.flatten() for z in grad]);vv=torch.cat([z.flatten() for z in bg]);cos=float(torch.nn.functional.cosine_similarity(v,vv,dim=0));assert torch.isfinite(vv).all()
  rows.append(dict(family=family,matched_ridge_readout_replay=same,unbalanced_ridge_readout_change=float(((no_ridge_balance-C).norm()/C.norm()).detach()),producer_gradient_cosine=cos))
 result=dict(rows=rows,scope='Toy algebra/control. No native model improvement. Output weighting with independent unconstrained readouts changes only producer optimization when regularization is consistently weighted. Fixed-readout changes under unweighted ridge are a regularization effect.')
 (P/'OUTPUT_BALANCING_CONTROL_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
