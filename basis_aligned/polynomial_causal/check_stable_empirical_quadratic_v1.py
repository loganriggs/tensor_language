"""Compare equivalent objectives and every parameter-block gradient on toys."""
import json
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel,QuadraticObjective
from stable_empirical_quadratic_v1 import StableEmpiricalObjective,qr_writers
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(91162200);results={}
    for kind in ['product','square','shared_reader','block']:
        model=QuadraticModel(kind,6,products=4,readers=4,groups=2,block_size=3)
        x=torch.randn(100,6,dtype=torch.float64);y=torch.randn(100,6,dtype=torch.float64)
        root=torch.randn(6,6,dtype=torch.float64);m=root@root.T+torch.eye(6,dtype=torch.float64)
        old=QuadraticObjective(m,((y@m)*y).sum(),x=x,y=y);new=StableEmpiricalObjective(m,x,y)
        oldloss,oldw,_=old.loss(model);oldloss.backward();oldgrads=[p.grad.clone() for p in model.parameters()]
        model.zero_grad();newloss,neww,_=new.loss(model);newloss.backward()
        grad_errors=[float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),oldgrads)]
        finite_errors=[]
        for p in model.parameters():
            direction=torch.randn_like(p);direction/=direction.norm();analytic=float((p.grad*direction).sum());h=1e-5
            with torch.no_grad():
                original=p.clone();p.copy_(original+h*direction);plus=float(new.loss(model)[0]);p.copy_(original-h*direction);minus=float(new.loss(model)[0]);p.copy_(original)
            finite_errors.append(abs(analytic-(plus-minus)/(2*h)))
        results[kind]=dict(loss_absolute_error=abs(float(oldloss-newloss)),writer_relative_error=float((oldw-neww).norm()/oldw.norm()),gradient_max_absolute_errors=grad_errors,finite_difference_absolute_errors=finite_errors)
        assert results[kind]['loss_absolute_error']<1e-10 and max(grad_errors)<1e-8 and max(finite_errors)<1e-6
    # Deliberately almost dependent columns; test same unconstrained LS, not ridge.
    q,_=torch.linalg.qr(torch.randn(80,6,dtype=torch.float64));v,_=torch.linalg.qr(torch.randn(6,6,dtype=torch.float64));f=q@torch.diag(torch.logspace(0,-7,6,dtype=torch.float64))@v.T
    y=torch.randn(80,3,dtype=torch.float64);beta,r=qr_writers(f,y)
    u,s,vh=torch.linalg.svd(f,full_matrices=False);reference=(vh.T/s)@(u.T@y)
    bridge=float((f@beta-f@reference).norm()/(f@reference).norm())
    assert bridge<1e-7
    result=dict(schema='stable.empirical.quadratic.control.v1',controls=results,ill_conditioned_feature_condition=float(torch.linalg.cond(f)),ill_conditioned_qr_svd_function_relative_error=bridge,passed=True,scope='First derivatives and full-rank conditional writer solution only. Does not validate a reduced Hessian or guarantee global convergence.')
    with (P/'STABLE_EMPIRICAL_QUADRATIC_V1_CONTROL.json').open('x') as h:json.dump(result,h,indent=2);h.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
