"""Paired random starts: full vs reduced LL1 optimization on known structure."""
import json,time,itertools,hashlib
from pathlib import Path
import torch,numpy as np
from scipy.optimize import minimize
from symmetric_ll1_objective_v1 import Objective as Joint,cp,value_gradient
from symmetric_ll1_projected_v1 import Objective as Projected,output_solve
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1901)
    m,r,d,o=3,3,12,8;penalty=.01
    a=torch.stack([torch.linalg.qr(torch.randn(d,r)).Q.T for _ in range(m)])
    s=torch.tensor([[1.,-.8,.5],[.8,.6,-.5],[-1.,.8,.4]])
    c=torch.randn(m,o);c/=c.norm(dim=1,keepdim=True)
    target=cp(a,s,c);truth=dense(*target);total=truth.square().sum()
    planted=[dense(*cp(a[j:j+1],s[j:j+1],c[j:j+1])) for j in range(m)]
    feasible=float(penalty*sum(t.square().sum() for t in planted)/total)
    rows=[];artifacts={}
    for seed in range(1902,1910):
        torch.manual_seed(seed);initial_a=torch.randn(m,r,d);initial_a/=initial_a.norm(dim=-1,keepdim=True)
        initial_s=torch.randn(m,r);initial_s/=initial_s.norm(dim=-1,keepdim=True)
        initial_c,*_=output_solve(target,initial_a,initial_s,penalty)
        for mode in ['joint','projected']:
            objective=Joint(target,(initial_a,initial_s,initial_c),total,penalty) if mode=='joint' else Projected(target,initial_a,initial_s,total,penalty)
            initial=objective.evaluate(objective.initial)[0];tic=time.perf_counter()
            fit=minimize(objective.evaluate,objective.initial,jac=True,method='L-BFGS-B',options=dict(maxiter=1000,maxcor=20,maxls=30,ftol=0.,gtol=1e-9))
            value,gradient=objective.evaluate(fit.x);parts=objective.physical(fit.x)
            if mode=='projected':parts=(*parts,objective.writer)
            aa,ss,cc=parts;forms=[dense(*cp(aa[j:j+1],ss[j:j+1],cc[j:j+1])) for j in range(m)]
            error=float((truth-sum(forms)).square().sum()/total)
            direct=error+float(penalty*sum(f.square().sum() for f in forms)/total)
            cosine=max(min(float((forms[j]*planted[p[j]]).sum()/(forms[j].norm()*planted[p[j]].norm())) for j in range(m)) for p in itertools.permutations(range(m)))
            rows.append(dict(seed=seed,mode=mode,initial_objective=initial,final_objective=value,residual=error,
                             dense_replay=abs(value-direct),matched_group_cosine=cosine,gradient_inf=float(np.max(np.abs(gradient))),
                             iterations=fit.nit,evaluations=fit.nfev,seconds=time.perf_counter()-tic,termination=str(fit.message)))
            artifacts[f'{mode}_{seed}']=parts;print(json.dumps(rows[-1]),flush=True)
    projected=[row for row in rows if row['mode']=='projected'];best=min(projected,key=lambda x:x['final_objective'])
    successes={mode:sum(row['final_objective']<=1.1*feasible for row in rows if row['mode']==mode) for mode in ['joint','projected']}
    cache=Path('/dev/shm/bilin18_ll1_projected_recovery_v1.pt');torch.save(dict(target=target,planted=(a,s,c),candidates=artifacts),cache)
    result=dict(predictions=dict(pred_a_dense=max(row['dense_replay'] for row in rows)<=1e-8,
                                 pred_b_projected_recovery=successes['projected']>=4,
                                 pred_c_selected_groups=best['residual']<=.001 and best['matched_group_cosine']>=.99),
                successes=successes,known_planted_objective=feasible,selected_projected_seed=best['seed'],rows=rows,
                cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
                scope='Paired planted weight-only starts, not native recovery or global optimum evidence.')
    Path(__file__).with_name('LL1_PROJECTED_RECOVERY_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['predictions']))


if __name__=='__main__':main()
