import json,math
from pathlib import Path
import torch
from symmetric_product_gauss_newton_v1 import pack,unpack,normal_action,normal_diagonal,loss_gradient
from symmetric_product_als_v1 import pcg

torch.set_num_threads(2);torch.manual_seed(43);torch.set_default_dtype(torch.float64)
q,d,o=3,6,5
pars=(torch.randn(q,d),torch.randn(q,d),torch.randn(o,q));theta=pack(pars)
lam=.01

def components(parts):
    a,b,z=parts;s=(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:])/2
    return z.T[:,:,None,None]*s[:,None,:,:]

native=(torch.randn(4,d),torch.randn(4,d),torch.randn(o,4))
target=components(native).sum(0);total=target.square().sum()
def residual(v):
    terms=components(unpack(v,pars))
    return torch.cat([(terms.sum(0)-target).flatten(),math.sqrt(lam)*terms.flatten()])

j=torch.autograd.functional.jacobian(residual,theta);h=j.T@j
v=torch.randn_like(theta);actual=normal_action(pars,v,lam)
diag=normal_diagonal(pars,lam)
f,g,_=loss_gradient(pars,native,total,lam)
expected_gradient=j.T@residual(theta)/total
mu=.03;damping=mu*diag.clamp_min(1e-12)
solution,cg=pcg(lambda x:normal_action(pars,x,lam)+damping*x,-expected_gradient,lambda x:x/(diag+damping),tolerance=1e-10,max_iterations=300)
direct=torch.linalg.solve(h+torch.diag(damping),-expected_gradient)
metrics=dict(normal_action_relative_error=float((actual-h@v).norm()/(h@v).norm()),normal_diagonal_relative_error=float((diag-h.diag()).norm()/h.diag().norm()),gradient_relative_error=float((g-expected_gradient).norm()/expected_gradient.norm()),loss_absolute_error=float(abs(f-residual(theta).square().sum()/total)),pcg_relative_solution_error=float((solution-direct).norm()/direct.norm()),pcg=cg)
metrics['passed']=all(metrics[k]<=1e-9 for k in metrics if k!='pcg') and cg['converged']
assert metrics['passed'],metrics
Path(__file__).with_name('SYMMETRIC_PRODUCT_GAUSS_NEWTON_V1_CONTROL.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics))
