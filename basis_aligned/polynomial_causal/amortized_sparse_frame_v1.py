"""Fixed-support PR+ epochs with exact streamed reselection at boundaries.

The true best-support objective is monotone at epoch boundaries. Inside an
epoch the fixed-support objective is a monotonically increasing lower bound.
"""
import time
import torch
from streamed_sparse_frame_v2 import select
from sparse_core_stiefel_v1 import project_tangent, score_gradient, armijo_step


def fit(q,l,r,w,total,count,max_steps=2000,seconds=600,tolerance=1e-6,inner_steps=20,callback=None):
    start=time.perf_counter();history=[];updates=0;reason='step_limit';evaluations=0;backtracks=0
    maximum_fixed_decrease=0.;maximum_epoch_decrease=0.;old_epoch=None
    while True:
        edges,score,gap=select(q,l,r,w,total,count);evaluations+=1
        checked,g=score_gradient(q,l,r,w,edges,total)
        assert abs(float(checked-score))<1e-10
        if old_epoch is not None:maximum_epoch_decrease=max(maximum_epoch_decrease,old_epoch-float(score))
        assert maximum_epoch_decrease<1e-10
        old_epoch=float(score);stationarity=float(g.norm()/score.clamp_min(1e-30))
        row=dict(iteration=updates,capture=float(score),stationarity=stationarity,gap=float(gap),selections=evaluations)
        history.append(row)
        if callback is not None:callback(q,edges,row)
        if stationarity<=tolerance and float(gap)>1e-12:
            reason='stationary_with_support_gap';break
        if updates>=max_steps:break
        if time.perf_counter()-start>=seconds:
            reason='time_limit';break
        previous_gradient=None;previous_direction=None;step_hint=None;failed=False
        for _ in range(min(inner_steps,max_steps-updates)):
            direction=g
            if previous_gradient is not None:
                transported=project_tangent(q,previous_gradient)
                beta=max(0.,min(10.,float((g*(g-transported)).sum()/previous_gradient.square().sum().clamp_min(1e-30))))
                direction=g+beta*project_tangent(q,previous_direction)
                if float((g*direction).sum())<.1*float(g.square().sum()):direction=g
            slope=float((g*direction).sum())
            hint=min(1/max(float(direction.norm()),1e-30),step_hint or .01*float(score)/max(slope,1e-30))
            trial,receipt=armijo_step(q,g,direction,l,r,w,edges,total,initial_step=hint)
            backtracks+=receipt['backtracks']
            if not receipt['accepted']:
                failed=True;break
            previous_gradient=g;previous_direction=direction;q=trial;step_hint=receipt['step']*1.5;updates+=1
            new_score,new_gradient=score_gradient(q,l,r,w,edges,total)
            maximum_fixed_decrease=max(maximum_fixed_decrease,float(score-new_score));assert maximum_fixed_decrease<1e-10
            score,g=new_score,new_gradient
            if time.perf_counter()-start>=seconds or float(g.norm()/score.clamp_min(1e-30))<=tolerance:break
        if failed:
            # Fresh exact support and gradient before reporting an early stop.
            edges,score,gap=select(q,l,r,w,total,count);evaluations+=1
            _,g=score_gradient(q,l,r,w,edges,total)
            stat=float(g.norm()/score.clamp_min(1e-30))
            reason='stationary_with_support_gap' if stat<=tolerance and float(gap)>1e-12 else 'line_search_failed'
            row=dict(iteration=updates,capture=float(score),stationarity=stat,gap=float(gap),selections=evaluations)
            history.append(row)
            if callback is not None:callback(q,edges,row)
            break
    return q,edges,dict(history=history,reason=reason,converged=reason=='stationary_with_support_gap',
                        seconds=time.perf_counter()-start,updates=updates,selections=evaluations,backtracks=backtracks,
                        maximum_fixed_decrease=maximum_fixed_decrease,maximum_epoch_decrease=maximum_epoch_decrease)


def control():
    torch.set_default_dtype(torch.float64);gen=torch.Generator().manual_seed(120487)
    true=torch.linalg.qr(torch.randn(6,6,generator=gen)).Q
    l=r=true.T;w=torch.randn(8,6,generator=gen);total=float(w.square().sum())
    reports=[]
    for seed,noise in [(120491,.05),(120497,None)]:
        g=torch.Generator().manual_seed(seed);raw=torch.randn(6,6,generator=g)
        q=torch.linalg.qr(raw if noise is None else true+noise*raw).Q
        q,e,result=fit(q,l,r,w,total,6,max_steps=1000,seconds=30,inner_steps=20)
        _,exact,gap=select(q,l,r,w,total,6)
        result.update(seed=seed,initialization='independent' if noise is None else 'near',
                      final_selection_error=abs(float(exact)-result['history'][-1]['capture']),
                      orthogonality=float((q.T@q-torch.eye(6)).abs().max()))
        reports.append(result)
    return dict(reports=reports,passed=all(r['final_selection_error']<1e-12 and r['orthogonality']<1e-12 and r['maximum_fixed_decrease']<1e-12 and r['maximum_epoch_decrease']<1e-12 for r in reports),
                planted_recoveries=sum(r['history'][-1]['capture']>=1-1e-8 for r in reports),
                scope='Numerical ascent/reselection and two planted starts; no global guarantee.')


if __name__=='__main__':
    import json
    print(json.dumps(control()))
