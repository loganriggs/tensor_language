"""Independent starts selected only by whole-group penalized weight objective."""
import itertools
import json
import time
from pathlib import Path
import torch
from scipy.optimize import minimize
from shared_reader_group_objective_v1 import value_gradient, cp
from shared_reader_conditional_v1 import group_cp
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    saved=torch.load('/dev/shm/bilin18_shared_reader_joint_polish_v1.pt',map_location='cpu',weights_only=False)
    target=saved['target'];truth=dense(*target);total=truth.square().sum();penalty=.01
    planted=[dense(*group_cp(*g)) for g in saved['planted']]
    feasible=float(penalty*sum(x.square().sum() for x in planted)/total)
    shapes=[(2,9),(2,2,9),(2,7,2)];sizes=[18,36,28];rows=[];candidates=[]
    def unpack(x):return tuple(t.reshape(s) for t,s in zip(torch.from_numpy(x).split(sizes),shapes))
    def objective(x):
        loss,grad,_=value_gradient(target,*unpack(x),total,penalty,chunk=4)
        return float(loss),torch.cat([g.flatten() for g in grad]).numpy()
    for seed in range(1500,1508):
        start=time.monotonic();torch.manual_seed(seed)
        a,v,w=[torch.randn(s) for s in shapes];a/=a.norm(dim=1,keepdim=True)
        _,_,details=value_gradient(target,a,v,w,total,penalty)
        w/=details['group_energy']**.5
        x=torch.cat([a.flatten(),v.flatten(),w.flatten()]).numpy();initial=objective(x)[0]
        fit=minimize(objective,x,jac=True,method='L-BFGS-B',options=dict(maxiter=2000,maxcor=20,maxls=30,ftol=0.,gtol=1e-9))
        parts=unpack(fit.x);loss,grad,details=value_gradient(target,*parts,total,penalty)
        forms=[dense(*cp(*(p[j:j+1] for p in parts))) for j in range(2)]
        direct=float(((truth-sum(forms)).square().sum()+penalty*sum(f.square().sum() for f in forms))/total)
        cos=max(min(float((forms[j]*planted[perm[j]]).sum()/(forms[j].norm()*planted[perm[j]].norm()))
                    for j in range(2)) for perm in itertools.permutations(range(2)))
        rows.append(dict(seed=seed,initial_objective=initial,final_objective=float(loss),residual=details['residual'],
                         group_energy=details['group_energy'],dense_replay=abs(float(loss)-direct),
                         gradient_inf=float(torch.cat([g.flatten() for g in grad]).abs().max()),
                         matched_minimum_group_cosine=cos,iterations=fit.nit,evaluations=fit.nfev,
                         seconds=time.monotonic()-start,termination=str(fit.message)))
        candidates.append(parts);print(json.dumps(rows[-1]),flush=True)
    selected=min(range(8),key=lambda i:rows[i]['final_objective']);best=rows[selected]
    success=sum(row['final_objective']<=1.1*feasible for row in rows)
    result=dict(predictions=dict(pred_a_instrument=all(row['dense_replay']<=1e-9 and row['final_objective']<=row['initial_objective'] for row in rows),
                                 pred_b_recovery_rate=success>=4,pred_c_selected_group_recovery=best['residual']<=.001 and best['matched_minimum_group_cosine']>=.99),
                known_planted_objective=feasible,successful_objectives=success,selected_seed=best['seed'],rows=rows,
                scope='Planted multistart diagnostic, selection by weight objective only, no native or global claim.')
    Path(__file__).with_name('SHARED_READER_RESTARTS_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(dict(parts=candidates[selected],target=target,selected_seed=best['seed']),'/dev/shm/bilin18_shared_reader_restarts_v1.pt')
    print(json.dumps(result['predictions']))


if __name__=='__main__':main()
