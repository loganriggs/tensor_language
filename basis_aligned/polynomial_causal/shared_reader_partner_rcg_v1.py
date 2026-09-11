"""Sphere nonlinear conjugate gradient for the exact rank-constrained partner score."""
import json,time
from pathlib import Path
import torch
from shared_input_ranked_partner_v1 import objective,value_and_gradient
from congruence_block_operator_v1 import sandwich


def optimize(initial,l,r,d,root,total,rank,max_steps=1000,tolerance=1e-8,seconds=None):
    start=time.perf_counter();a=initial/initial.norm();history=[];evaluations=0
    value,grad=value_and_gradient(a,l,r,d,root,total,rank);evaluations+=1
    grad-=a*(a@grad);direction=grad.clone();initial_value=float(value);rate=50.;terminal='step_limit'
    for step in range(max_steps):
        station=float(grad.norm())
        if station<=tolerance:terminal='stationarity';break
        if seconds is not None and time.perf_counter()-start>=seconds:terminal='time_limit';break
        if float(grad@direction)<=.01*float(grad.square().sum()):direction=grad.clone()
        slope=float(grad@direction);accepted=False
        for search in range(50):
            candidate=a+rate*direction;candidate/=candidate.norm()
            with torch.no_grad():new=objective(candidate,l,r,d,root,total,rank)
            evaluations+=1
            if float(new)>=float(value)+1e-4*rate*slope:accepted=True;break
            rate*=.5
        if not accepted:terminal='line_search_stall';break
        new_value,new_grad=value_and_gradient(candidate,l,r,d,root,total,rank);evaluations+=1
        new_grad-=candidate*(candidate@new_grad)
        transported=grad-candidate*(candidate@grad)
        beta=max(0.,float(new_grad@(new_grad-transported))/max(float(grad.square().sum()),1e-30))
        if beta>1000:beta=0.
        direction=new_grad+beta*(direction-candidate*(candidate@direction))
        a,value,grad=candidate,new_value,new_grad
        if step%5==0:history.append(dict(step=step+1,capture=float(value),tangent_stationarity=float(grad.norm()),rate=rate,beta=beta,seconds=time.perf_counter()-start))
        rate=min(1000.,rate*1.5)
    final,gradient=value_and_gradient(a,l,r,d,root,total,rank);evaluations+=1
    station=float((gradient-a*(a@gradient)).norm())
    return a,dict(converged=station<=tolerance,terminal='stationarity' if station<=tolerance else terminal,
                  initial_capture=initial_value,capture=float(final),tangent_stationarity=station,
                  steps=step+1,evaluations=evaluations,seconds=time.perf_counter()-start,history=history,
                  continuation=dict(reader=a.detach().cpu().tolist(),gradient=grad.detach().cpu().tolist(),
                                    direction=direction.detach().cpu().tolist(),rate=rate))


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1163)
    true=torch.randn(7);true/=true.norm();l=true.repeat(2,1)
    r=torch.randn(2,7);r-=(r@true)[:,None]*true
    d=torch.randn(5,2);u=torch.randn(11,5);root=torch.linalg.cholesky(u.T@u)
    k=(u@d).T@(u@d);total=sandwich(l,r,k,torch.eye(7)).trace()
    a,fit=optimize(true+.2*torch.randn(7),l,r,d,root,total,2,max_steps=500,tolerance=1e-8)
    captures=[fit['initial_capture']]+[v['capture'] for v in fit['history']]+[fit['capture']]
    increase=max([0.]+[captures[i]-captures[i+1] for i in range(len(captures)-1)])
    result=dict(instrument_passed=fit['converged'] and fit['capture']>=1-1e-10 and increase<1e-10,
                planted_reader_cosine=float(abs(a@true)),maximum_capture_decrease=increase,fit=fit,
                scope='Planted shared reader/rank2 partner recovery; canonical sphere stationarity. No native/global optimum guarantee.')
    Path(__file__).with_name('SHARED_READER_PARTNER_RCG_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
