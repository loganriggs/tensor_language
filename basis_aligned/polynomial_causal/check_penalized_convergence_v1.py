"""Planted regularized optimum with preserved stopping bars and checkpoint resume."""
import io,json,time
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective
from convergent_quadratic_fit_v2 import advance
from joint_quadratic_fit_v1 import product_cross

def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(9116501);dt=torch.float64
    l,r=torch.randn(1,5,dtype=dt),torch.randn(1,5,dtype=dt)
    d=torch.randn(3,1,dtype=dt);metric=torch.eye(3,dtype=dt);penalty=.01
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    objective=EnergyRegularizedObjective(metric,total,l=l,r=r,d=d,penalty=penalty)
    model=QuadraticModel('product',5,products=1)
    first=advance(model,objective,seconds=0,adam_steps=30,diagnostic_every=10)
    assert not first['converged']
    buffer=io.BytesIO();torch.save(first,buffer);buffer.seek(0)
    restored=torch.load(buffer,weights_only=False)
    model.load_state_dict(restored['model'])
    last=advance(model,objective,restored,seconds=10,adam_steps=30,diagnostic_every=10)
    best=last['best']['diagnostics']
    expected_residual=(penalty/(1+penalty))**2
    expected_objective=penalty/(1+penalty)
    assert last['converged']
    assert abs(best['optimization_loss']-expected_objective)<=1e-8
    assert abs(best['squared_relative_error']-expected_residual)<=1e-8
    assert best['optimization_loss']==min(h['optimization_loss'] for h in last['history'])
    result=dict(passed=True,converged=last['converged'],best=best,
        expected_residual=expected_residual,expected_objective=expected_objective,
        unfinished_checkpoint_preserved=True,best_tracks_actual_objective=True,
        wall_seconds=time.perf_counter()-start)
    with Path(__file__).with_name('PENALIZED_CONVERGENCE_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
