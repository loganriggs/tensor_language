"""QR-retracted descent of exact normalized cut; every accepted step is scored."""
import time
import torch
from fullu_block_rotation_v1 import objective_gradient
from fullu_input_blocks_v1 import sandwich, partition_metrics


def optimize(l,r,g,q,max_steps=1500,seconds=600,tolerance=1e-6,callback=None):
    start=time.perf_counter()
    k=sandwich(l,r,g,torch.eye(l.shape[1],device=l.device,dtype=l.dtype))
    q=torch.linalg.qr(q).Q
    p=q@q.T
    value,eg,_=objective_gradient(l,r,g,p,k)
    initial=float(value);step=100.;history=[];reason='step_limit';evaluations=1
    for iteration in range(max_steps+1):
        grad=2*(eg@q-q@(q.T@eg@q))
        norm=float(grad.norm())
        row=dict(iteration=iteration,objective=float(value),gradient_norm=norm,seconds=time.perf_counter()-start)
        history.append(row)
        if callback is not None and iteration%20==0:callback(row,q)
        if norm<=tolerance:
            reason='gradient';break
        if iteration==max_steps:break
        if time.perf_counter()-start>=seconds:
            reason='time_limit';break
        accepted=False
        for _ in range(24):
            trial=torch.linalg.qr(q-step*grad).Q
            pp=trial@trial.T
            candidate,ge,_=objective_gradient(l,r,g,pp,k);evaluations+=1
            if float(candidate)<=float(value)-1e-4*step*norm**2:
                q,p,value,eg=trial,pp,candidate,ge;accepted=True;step=min(step*1.5,1e6);break
            step*=.5
        if not accepted:
            reason='line_search';break
    metrics=partition_metrics(l,r,g,p,k)
    return q,dict(initial=initial,**metrics,converged=reason=='gradient',stop_reason=reason,
                  iterations=iteration,evaluations=evaluations,gradient_norm=norm,
                  history=history,execution_seconds=time.perf_counter()-start,
                  orthogonality_error=float((q.T@q-torch.eye(q.shape[1],device=q.device,dtype=q.dtype)).norm()))
