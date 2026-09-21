"""Independent lifted least-squares and envelope-gradient controls."""
from pathlib import Path
import json,torch
from learned_quadratic_factors import Objective,dense
from centered_product_response import features
from profiled_learned_quadratic_factors import readout
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.manual_seed(970);rows=[]
for d in range(3,8):
 rand=lambda *s:torch.randn(*s,dtype=torch.float64);tc,ta,tb=rand(3,4),rand(4,d),rand(4,d);x,v=rand(23,d),rand(23,d);obj=Objective(tc,ta,tb,x,v);a,b=rand(3,d),rand(3,d);a.requires_grad_();b.requires_grad_();C=readout(obj,a,b)
 identity=torch.eye(3,dtype=a.dtype);design=torch.cat([dense(identity,a,b).flatten(1).T/obj.energy.sqrt(),features(x,v,a,b)/obj.response_energy.sqrt()]);target=torch.cat([dense(tc,ta,tb).flatten(1).T/obj.energy.sqrt(),obj.truth/obj.response_energy.sqrt()]);unregularized=torch.linalg.lstsq(design.detach(),target.detach()).solution.T;old_error=float(((design@(C-unregularized).T).norm()/target.norm()).detach());ridge=1e-10*design.square().sum(0).mean().detach();augmented=torch.cat([design.detach(),ridge.sqrt()*identity]);rhs=torch.cat([target.detach(),torch.zeros(3,3,dtype=a.dtype)]);ref=torch.linalg.lstsq(augmented,rhs).solution.T;solve=float(((design@(C-ref).T).norm()/target.norm()).detach())
 loss=sum(obj.losses(C.detach(),a,b));ga,gb=torch.autograd.grad(loss,(a,b));da,db=rand(*a.shape),rand(*b.shape);prediction=(ga*da).sum()+(gb*db).sum()
 def evaluate(t):
  aa,bb=a.detach()+t*da,b.detach()+t*db;cc=readout(obj,aa,bb);return sum(obj.losses(cc,aa,bb))
 step=1e-5;fd=(evaluate(step)-evaluate(-step))/(2*step);err=float(abs(fd-prediction)/max(abs(fd),1e-8));rows.append(dict(dimension=d,unregularized_reference_error=old_error,lifted_lstsq_error=solve,envelope_directional_error=err))
print(rows,flush=True)
assert max(r['lifted_lstsq_error'] for r in rows)<1e-8 and max(r['envelope_directional_error'] for r in rows)<1e-6
(p/'PROFILED_QUADRATIC_CONTROLS_V1.json').write_text(json.dumps(dict(records=rows,correction='Initial reference omitted the 1e-10 Gram ridge and failed at2.70e-8. Correct augmented least squares includes the identical ridge; numerical tolerances unchanged.'),indent=2)+'\n');print(rows)
