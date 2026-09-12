"""Full-frame sparse support ascent using existing exact gradients and Armijo.

Avoid allocating the output-by-all-pairs core. Nonconvex, piecewise smooth;
stationarity on one support is not global recovery.
"""
import time
import torch
from full_input_sparse_core_v1 import edge_energies
from sparse_core_stiefel_v1 import score_gradient, armijo_step


def select(q,l,r,w,total,count):
    energies,edges=edge_energies(l,r,w,q)
    values,indices=energies.topk(count+1)
    return edges[:,indices[:count]],values[:count].sum()/total,(values[count-1]-values[count])/total


def fit(q,l,r,w,total,count,max_steps=200,seconds=600,tolerance=1e-6):
    start=time.perf_counter();history=[];reason='step_limit';step_hint=None
    for iteration in range(max_steps+1):
        edges,score,gap=select(q,l,r,w,total,count)
        checked,g=score_gradient(q,l,r,w,edges,total)
        assert abs(float(checked-score))<1e-10
        stationarity=float(g.norm()/score.clamp_min(1e-30))
        history.append(dict(iteration=iteration,capture=float(score),stationarity=stationarity,gap=float(gap)))
        if stationarity<=tolerance and float(gap)>1e-12:
            reason='stationary_with_support_gap';break
        if iteration==max_steps:break
        if time.perf_counter()-start>=seconds:
            reason='time_limit';break
        hint=min(1/max(float(g.norm()),1e-30),step_hint or .01*float(score)/max(float(g.square().sum()),1e-30))
        trial,receipt=armijo_step(q,g,g,l,r,w,edges,total,initial_step=hint)
        if not receipt['accepted']:
            reason='line_search_failed';break
        q=trial;step_hint=receipt['step']*1.5
    return q,edges,dict(history=history,reason=reason,converged=reason=='stationary_with_support_gap',seconds=time.perf_counter()-start)


def control():
    torch.set_default_dtype(torch.float64);gen=torch.Generator().manual_seed(120457)
    l,r=[torch.randn(9,6,generator=gen) for _ in range(2)]
    w=torch.randn(5,9,generator=gen);q=torch.linalg.qr(torch.randn(6,6,generator=gen)).Q
    total=float(edge_energies(l,r,w,q)[0].sum())
    edges,score,_=select(q,l,r,w,total,4)
    from sparse_orthogonal_quadratic_core_v1 import orthogonal_core
    dense,_=orthogonal_core(l,r,w,q.T)
    selection_error=abs(float(score)-float(dense.square().sum(0).topk(4).values.sum()/total))
    out,_,report=fit(q,l,r,w,total,4,max_steps=20,seconds=60)
    captures=[v['capture'] for v in report['history']]
    orth=float((out.T@out-torch.eye(6)).abs().max())
    return dict(selection_error=selection_error,orthogonality=orth,initial=captures[0],final=captures[-1],
                maximum_decrease=max([a-b for a,b in zip(captures,captures[1:])]+[0]),
                passed=selection_error<1e-12 and orth<1e-12 and all(b>=a-1e-12 for a,b in zip(captures,captures[1:])) and captures[-1]>captures[0],
                scope='Dense support equality and monotone ascent only; not planted/global recovery.')


if __name__=='__main__':
    import json
    print(json.dumps(control()))
