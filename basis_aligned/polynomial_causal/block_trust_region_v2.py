"""Pymanopt trust regions with the exact reduced variable-projection Hessian."""
import json,time
from pathlib import Path
import numpy as np
import torch
import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
from orthogonal_multioutput_pymanopt_v2 import evaluate,manifold_for
from block_variable_projection_curvature_v1 import hessian_vector


def fit(objective,bank,packed,outputs,seconds=120,tolerance=1e-7):
    manifold=manifold_for(bank,packed);cache={};counts=dict(evaluations=0,hessian_vectors=0,accepted_points=0);history=[];accepted={};maximum_increase=[0.]
    def tensors(b,c):return [torch.as_tensor(x,dtype=bank.dtype,device=bank.device) for x in [b,c]]
    def calc(b,c):
        if 'b' not in cache or not np.array_equal(cache['b'],b) or not np.array_equal(cache['c'],c):
            value,grad,details=evaluate(objective,*tensors(b,c),outputs)
            cache.update(b=b.copy(),c=c.copy(),value=value,grad=[g.cpu().numpy() for g in grad],details=details);counts['evaluations']+=1
        return cache
    @pymanopt.function.numpy(manifold)
    def cost(b,c):return calc(b,c)['value']
    @pymanopt.function.numpy(manifold)
    def gradient(b,c):
        row=calc(b,c)
        # This solver requests a new-point gradient only after accepting it;
        # Hessian conversion repeats gradients at the current accepted point.
        if 'b' not in accepted or not np.array_equal(accepted['b'],b) or not np.array_equal(accepted['c'],c):
            if 'cost' in accepted:maximum_increase[0]=max(maximum_increase[0],row['value']-accepted['cost'])
            step=counts['accepted_points'];counts['accepted_points']+=1
            accepted.update(b=b.copy(),c=c.copy(),cost=row['value'])
            if step%5==0:
                item=dict(iteration=step,cost=row['value']);history.append(item)
                if step%25==0:print(json.dumps(item),flush=True)
        return row['grad']
    @pymanopt.function.numpy(manifold)
    def hessian(b,c,bd,cd):
        counts['hessian_vectors']+=1
        return [x.cpu().numpy() for x in hessian_vector(objective,*tensors(b,c),outputs,tensors(bd,cd))]
    problem=Problem(manifold,cost,euclidean_gradient=gradient,euclidean_hessian=hessian)
    solver=TrustRegions(min_gradient_norm=tolerance,max_time=seconds,max_iterations=1000,verbosity=0)
    initial=[bank.detach().cpu().numpy(),packed.detach().cpu().numpy()];start=time.perf_counter()
    result=solver.run(problem,initial_point=initial,Delta0=.1,Delta_bar=1.,maxinner=50)
    end=calc(*result.point);grad=manifold.euclidean_to_riemannian_gradient(result.point,end['grad'])
    station=float(manifold.norm(result.point,grad))
    return [torch.from_numpy(x.copy()) for x in result.point],dict(loss=end['value'],**end['details'],
        tangent_stationarity=station,converged=station<=tolerance,seconds=time.perf_counter()-start,
        iterations=int(result.iterations),**counts,stopping_criterion=result.stopping_criterion,
        scalar_history=history,maximum_objective_increase=maximum_increase[0])


def control():
    from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective
    from joint_quadratic_fit_v1 import product_cross
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1531);np.random.seed(1531)
    bank=torch.linalg.qr(torch.randn(2,7,3)).Q;packed=torch.randn(6,8);packed/=packed.norm(dim=0)
    l,r=torch.randn(2,11,7);d=torch.randn(7,11);u=torch.randn(13,7);metric=u.T@u
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    obj=MultioutputWeightObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    man=manifold_for(bank,packed);point=[bank.numpy(),packed.numpy()];direction=man.random_tangent_vector(point)
    _,g,_=evaluate(obj,bank,packed,4);g=[x.numpy() for x in g]
    hv=hessian_vector(obj,bank,packed,4,[torch.from_numpy(x) for x in direction]);hv=[x.numpy() for x in hv]
    rh=man.euclidean_to_riemannian_hessian(point,g,hv,direction)
    eps=1e-5;rg=[]
    for sign in [1,-1]:
        trial=man.retraction(point,sign*eps*direction)
        _,eg,_=evaluate(obj,*[torch.from_numpy(x) for x in trial],4)
        tg=man.euclidean_to_riemannian_gradient(trial,[x.numpy() for x in eg])
        rg.append(man.projection(point,tg))
    finite=(rg[0]-rg[1])/(2*eps);error=float(man.norm(point,rh-finite)/man.norm(point,finite))
    initial=evaluate(obj,bank,packed,4)[0];_,report=fit(obj,bank,packed,4,seconds=8,tolerance=1e-7)
    result=dict(instrument_passed=error<1e-6 and report['loss']<initial and report['maximum_objective_increase']<1e-10,
        riemannian_hessian_finite_difference_relative_error=error,initial_loss=initial,
        solver={k:v for k,v in report.items() if k!='scalar_history'},scope='Correct manifold Hessian and working standard trust-region solver; toy convergence reported independently, native untested.')
    with Path(__file__).with_name('BLOCK_TRUST_REGION_V2_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']

if __name__=='__main__':control()
