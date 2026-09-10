"""Independent tangent, finite-difference and deterministic resume controls."""
import json
from pathlib import Path
import torch
from fit_multioutput_manifold_v1 import fit, tangent, retract, assign, dot
from multioutput_quadratic_blocks_v1 import MultioutputQuadraticBlocks, MultioutputWeightObjective
from joint_quadratic_fit_v1 import product_cross


def main():
    torch.set_num_threads(2); torch.manual_seed(741)
    model = MultioutputQuadraticBlocks(6, groups=2, rank=2, outputs=2)
    point = retract((model.bank.detach(), model.core.detach()), (torch.zeros_like(model.bank), torch.zeros_like(model.core)), 0)
    assign(model, point)
    l, r, d = torch.randn(8,6,dtype=torch.float64), torch.randn(8,6,dtype=torch.float64), torch.randn(5,8,dtype=torch.float64)
    metric = torch.eye(5,dtype=torch.float64)
    total = ((d.T@d) * product_cross(l,r,l,r)).sum()
    obj = MultioutputWeightObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    loss = obj.terms(model)[0]; loss.backward()
    g = tangent(point,(model.bank.grad,model.core.grad))
    h = tangent(point,tuple(torch.randn_like(x) for x in point))
    eps = 1e-6
    assign(model,retract(point,h,eps)); plus=float(obj.terms(model)[0].detach())
    assign(model,retract(point,h,-eps)); minus=float(obj.terms(model)[0].detach())
    finite = abs((plus-minus)/(2*eps)-float(dot(g,h)))
    bank_tangent = float((h[0]@point[0].transpose(-1,-2)+point[0]@h[0].transpose(-1,-2)).abs().max())
    core_tangent = float((h[1]*point[1]).sum(-1).abs().max())
    assign(model,point); full=fit(model,obj,seconds=60,max_steps=40)
    assign(model,point); half=fit(model,obj,seconds=60,max_steps=20)
    resumed=fit(model,obj,seconds=60,max_steps=20,state=half)
    resume=max(float((full['model'][k]-resumed['model'][k]).abs().max()) for k in full['model'])
    decrease=full['initial_loss']-full['history'][-1]['optimization_loss']
    result=dict(finite_gradient_absolute_error=finite,bank_tangent_error=bank_tangent,core_tangent_error=core_tangent,resume_max_difference=resume,objective_decrease=decrease,maximum_increase=full['maximum_objective_increase'])
    result['passed']=max(finite,bank_tangent,core_tangent,resume)<1e-8 and decrease>.01 and full['maximum_objective_increase']<=1e-12
    print(json.dumps(result,indent=2))
    Path(__file__).with_name('MULTIOUTPUT_MANIFOLD_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['passed']

if __name__=='__main__':main()
