"""Optimize the output direction for four real products in a fixed group core.

Extends simple_product_in_span_v1's signed-eigenvalue objective to k=4.
Three starts, local tangent-gradient convergence; no global optimum claim.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from cancellation_group_v1_audit import load
from native_support_exchange_v1_audit import P
from chunked_bilinear_coefficient_v1 import dense


def evaluate(raw,forms,k=4):
    t=torch.as_tensor(raw,dtype=forms.dtype);length=t.norm();u=t/length
    q=torch.einsum('o,oij->ij',u,forms);ev,v=torch.linalg.eigh(q)
    pos=ev[-k:].clamp_min(0);neg=ev[:k].clamp_max(0)
    approximation=(v[:,-k:]*pos)@v[:,-k:].T+(v[:,:k]*neg)@v[:,:k].T
    score=pos.square().sum()+neg.square().sum()
    gradient=2*torch.einsum('oij,ij->o',forms,approximation)
    tangent=gradient-u*(u@gradient)
    return float(score),tangent/length,tangent,u,approximation


def main():
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'CANCELLATION_OUTPUT_DIRECTION_V1_AUDIT.json';assert not out.exists();started=time.perf_counter()
    prior=json.loads((P/'CANCELLATION_GROUP_V1_AUDIT.json').read_text());ids=prior['rows'][-1]['products']
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;current=load(prior['current_source'],wh)
    a,b,w=current[0][ids],current[1][ids],current[2][:,ids]
    ib=torch.linalg.qr(torch.cat((a,b)).T,mode='reduced').Q;ob=torch.linalg.qr(w,mode='reduced').Q
    core=dense(a@ib,b@ib,ob.T@w);forms=core/core.norm()
    initial_svd=torch.linalg.svd(forms.reshape(16,-1),full_matrices=False).U[:,0]
    starts=[('svd',initial_svd)]+[(str(seed),torch.randn(16,generator=torch.Generator().manual_seed(seed))) for seed in (42,43)]
    rows=[];maxfd=0.;best=None
    for label,initial in starts:
        initial=initial/initial.norm();direction=torch.randn(16,generator=torch.Generator().manual_seed(281));direction/=direction.norm()
        value,gradient,_,_,_=evaluate(initial,forms);step=1e-5
        fd=abs((evaluate(initial+step*direction,forms)[0]-evaluate(initial-step*direction,forms)[0])/(2*step)-float(gradient@direction))
        maxfd=max(maxfd,fd);history=[value];clock=time.perf_counter()
        def objective(x):
            score,g,_,_,_=evaluate(x,forms);return -score,-g.numpy()
        def callback(x):history.append(evaluate(x,forms)[0])
        fit=minimize(objective,initial.numpy(),jac=True,method='L-BFGS-B',callback=callback,
                     options=dict(maxiter=1000,ftol=0.,gtol=1e-12,maxls=50,maxcor=10))
        score,_,tangent,u,approx=evaluate(fit.x,forms)
        dense_capture=1-float((forms-u[:,None,None]*approx[None]).square().sum())
        row=dict(start=label,initial_capture=value,capture=score,tangent_gradient=float(tangent.norm()),
            converged=float(tangent.norm())<=1e-8,solver_success=bool(fit.success),message=str(fit.message),
            iterations=int(fit.nit),seconds=time.perf_counter()-clock,fd_error=fd,
            dense_capture_replay=abs(dense_capture-score),maximum_capture_decrease=max([0.]+[x-y for x,y in zip(history,history[1:])]))
        rows.append(row)
        if best is None or score>best['capture']:best=dict(capture=score,direction=u.tolist(),start=label)
    result=dict(predictions=dict(pred_a_instrument=maxfd<=1e-6 and max(r['dense_capture_replay'] for r in rows)<=1e-8,
        pred_b_local_convergence=all(r['converged'] and r['maximum_capture_decrease']<=1e-10 for r in rows),
        pred_c_capture=best['capture']>=.95),rows=rows,best=best,source=prior['current_source'],seconds=time.perf_counter()-started,
        scope='Fixed16output/32input group, rank1output with four real products; local output optimization '
              'only. Does not settle free multi-output factors, other groups or semantic stability.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
