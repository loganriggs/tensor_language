"""Gradient controls and two planted coupled-polish trials, no native model."""
import json,time
from pathlib import Path
import numpy as np
import torch
from active_orthant_dictionary_v1 import OrthantObjective,polish
from quadratic_token_dictionary_v1 import cycle,conditional,diagnostics


def gradient_control():
    torch.manual_seed(918);x=torch.randn(8,4);b=torch.randn(6,4);b/=b.norm(dim=1,keepdim=True)
    z=torch.randn(8,6);z[z.abs()<.5]=0
    obj=OrthantObjective(x,z,b,.05);value,gradient=obj.value_gradient(obj.initial)
    point=torch.tensor(obj.initial,requires_grad=True)
    basis,codes,raw,norm,magnitudes=obj.unpack(point)
    loss=.5*(codes@basis-x).square().sum()+.05*magnitudes.sum()
    auto=torch.autograd.grad(loss,point)[0].detach().numpy()
    derivative_error=float(np.linalg.norm(gradient-auto)/np.linalg.norm(auto))
    direction=np.random.default_rng(918).normal(size=len(point));direction/=np.linalg.norm(direction)
    h=1e-5;finite=(obj.value_gradient(obj.initial+h*direction)[0]-obj.value_gradient(obj.initial-h*direction)[0])/(2*h)
    exact=float(gradient@direction);fd_error=abs(finite-exact)/max(1.,abs(exact))
    original=float(.5*(z@b-x).square().sum()+.05*z.abs().sum())
    return dict(analytic_autograd_error=derivative_error,directional_fd_error=fd_error,
        original_objective_replay=abs(value-original),canonicalization=obj.canonicalization,
        passed=derivative_error<=1e-9 and fd_error<=1e-7 and abs(value-original)<=1e-10 and obj.canonicalization['reconstruction_error']<=1e-10)


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    control=gradient_control();assert control['passed'];rows=[]
    for seed in (0,937):
        gen=torch.Generator().manual_seed(seed);truth=torch.randn(24,12,generator=gen);truth/=truth.norm(dim=1,keepdim=True)
        planted=torch.zeros(256,24)
        for i in range(256):planted[i,torch.randperm(24,generator=gen)[:2]]=torch.randn(2,generator=gen)
        x=planted@truth;x/=x.norm(dim=1,keepdim=True)
        b=truth+.15*torch.randn(truth.shape,generator=gen);b/=b.norm(dim=1,keepdim=True);z=torch.zeros(256,24)
        for _ in range(5):z,b,_=cycle(x,z,b,.05,1000,1e-8)
        initial=diagnostics(x,z,b,.05);history=[];started=time.perf_counter()
        for iteration in range(10):
            remaining=30-(time.perf_counter()-started)
            if remaining<=0:break
            z,b,report=polish(x,z,b,.05,max_iterations=200,seconds=remaining)
            z,code_report=conditional(z,b@b.T,x@b.T,.05,'codes',3000,1e-9)
            check=diagnostics(x,z,b,.05)
            history.append(dict(round=iteration+1,polish=report,code_refresh=code_report,check=check))
            print(json.dumps(dict(seed=seed,round=iteration+1,objective=check['objective'],stationarity=check['relative_stationarity'],seconds=time.perf_counter()-started)),flush=True)
            if check['relative_stationarity']<=1e-5:break
        final=diagnostics(x,z,b,.05)
        rows.append(dict(seed=seed,initial=initial,final=final,history=history,seconds=time.perf_counter()-started,
                         stationarity_reduction=initial['relative_stationarity']/max(final['relative_stationarity'],1e-30)))
    result=dict(predictions=dict(pred_a_gradient=control['passed'],
        pred_b_nonworsening=all(r['final']['objective']<=r['initial']['objective']+1e-10 for r in rows),
        pred_c_stationarity_reduction=all(r['stationarity_reduction']>=10 for r in rows),
        pred_d_full_convergence=all(r['final']['relative_stationarity']<=1e-5 for r in rows)),
        gradient_control=control,arms=rows,scope='Two planted warm starts, same L1 objective, fixed-orthant coupled steps plus global code refresh. Not a native result or global recovery theorem.')
    with Path(__file__).with_name('ACTIVE_ORTHANT_DICTIONARY_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=result['predictions'],gradient_control=control)),flush=True)


if __name__=='__main__':main()
