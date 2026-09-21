"""Dense coefficient/gradient validation for selected shared feature pairs."""
import json
from pathlib import Path
import torch
from sparse_quartic_bank import gram,native_cross,entries,features,support
from shared_quadratic_bank import normalize_bank
from quartic_cp import directional
from quartic_cp_profile import profile
P=Path(__file__).resolve().parent


def main():
 torch.set_num_threads(2);rows=[]
 for seed in range(5):
  torch.manual_seed(10000+seed);dtype=torch.float64;d=3
  u=torch.randn(4,2,d,dtype=dtype,requires_grad=True);v=torch.randn(4,2,d,dtype=dtype,requires_grad=True)
  with torch.no_grad():
   if seed==1:u[1]=u[0]
   if seed==2:v[1]=v[0]
   if seed==3:v.copy_(u)
   if seed==4:u[1]=u[0];v[1]=-v[0]
  U,V=normalize_bank(u,v);pairs=support(4,7,seed=45);idx=torch.cartesian_prod(*[torch.arange(d) for _ in range(4)])
  teacher=[torch.randn(2,4,dtype=dtype),torch.randn(4,3,dtype=dtype),torch.randn(4,3,dtype=dtype),torch.randn(3,4,dtype=dtype),torch.randn(4,d,dtype=dtype),torch.randn(4,d,dtype=dtype)]
  eye=torch.eye(d,dtype=dtype);target=directional(*teacher,[eye[idx[:,s]] for s in range(4)])
  phi=entries(U,V,pairs,idx);g=gram(U,V,pairs);cross=native_cross(teacher,U,V,pairs)
  loss,c=profile(g,cross,ridge=1e-6)
  dense=(phi@c.T-target).square().sum()-target.square().sum()+1e-6*c.square().sum()
  grad=torch.autograd.grad(loss,[u,v],retain_graph=True);other=torch.autograd.grad(dense,[u,v])
  rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-15)).detach())
  x=torch.randn(9,d,dtype=dtype);monomials=x[:,idx].prod(-1)
  row=dict(seed=seed,gram=rel(g,phi.T@phi),cross=rel(cross,target.T@phi),gradient=rel(torch.cat([a.flatten() for a in grad]),torch.cat([a.flatten() for a in other])),functional=rel(features(x,U,V,pairs),monomials@phi))
  assert max(row[k] for k in ['gram','cross','gradient','functional'])<1e-8,row
  rows.append(row)
 pairs=support();price=dict(quadratic_features=144,products_per_quadratic=4,selected_pairs=pairs.shape[1],variable_products=144*4+pairs.shape[1],coefficients=2*144*4*1152+16*pairs.shape[1]+1152*16,integer_indices=pairs.numel(),maximum_input_span=1152)
 result=dict(controls=rows,proposed_native_price=price,scope='Exact selected-pair machinery and literal proposed price; native sparse-bank optimization not run. Support is seeded, not learned or proven optimal.')
 (P/'SPARSE_QUARTIC_BANK_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
