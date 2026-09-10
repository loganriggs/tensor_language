import json
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective
from joint_quadratic_fit_v1 import product_cross
from square_gauge_diagnostics_v1 import stationarity

torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(838)
m=QuadraticModel('square',5,products=3);l,r,d=torch.randn(8,5),torch.randn(8,5),torch.randn(4,8);metric=torch.eye(4);total=((d.T@d)*product_cross(l,r,l,r)).sum();objective=EnergyRegularizedObjective(metric,total,l=l,r=r,d=d,penalty=.01)
with torch.no_grad():m.a.mul_(torch.logspace(-2,2,6)[:,None])
norm=m.a.detach().norm(dim=1,keepdim=True);old,_=objective.diagnostics(m);old_gradient=m.a.grad.detach().clone()
with torch.no_grad():m.a.div_(norm)
new,_=objective.diagnostics(m);stats=stationarity(m,new['captured_energy_fraction'],norm)
transport=m.a.grad.detach()/norm
result=dict(function_error=abs(old['optimization_loss']-new['optimization_loss']),gradient_transport_relative_error=float((transport-old_gradient).norm()/old_gradient.norm()),stationarity_transport_relative_error=abs(stats['original_gauge_relative_stationarity']/old['relative_stationarity']-1))
result['passed']=max(result.values())<=1e-10;assert result['passed'],result
Path(__file__).with_name('SQUARE_GAUGE_DIAGNOSTICS_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
