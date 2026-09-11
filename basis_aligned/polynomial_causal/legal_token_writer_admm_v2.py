"""Residual-balanced continuation of the same convex native writer objective."""
import json,time
from pathlib import Path
import torch
from quadratic_token_dictionary_v1 import soft
from legal_token_writer_admm_v1 import sylvester,diagnostics,solve as fixed_solve


def solve(p,g,c,penalty,state=None,rho=1.,max_steps=2000,tolerance=1e-6,seconds=None):
    f=p-p.mean(0);mean=p.mean(0);values,vectors=torch.linalg.eigh((g+g.T)/2)
    assert float(values[0])>0
    if state is None:
        z=(c@vectors/values[None,:])@vectors.T;a=f@z;dual=torch.zeros_like(a);offset=0
    else:
        z=state['z'].clone();a=state['a'].clone();dual=state['dual'].clone()
        rho=float(state['rho']);offset=int(state['step'])
        assert float(state['penalty'])==float(penalty)
    start=time.perf_counter();history=[];converged=False;adjustments=0
    for local in range(1,max_steps+1):
        step=offset+local
        z=sylvester(c+rho*(f.T@(a-dual)),values,vectors,mean,len(p),rho)
        actual=f@z;old_a=a
        a=soft(actual+dual,penalty/rho);residual=actual-a;dual+=residual
        if step%10==0 or local==max_steps:
            stats=diagnostics(p,g,c,z,a,dual,penalty,rho)
            primal=float(residual.norm());dual_residual=float((rho*f.T@(a-old_a)).norm())
            history.append(dict(step=step,elapsed=time.perf_counter()-start,rho=rho,
                                raw_primal_residual=primal,raw_dual_residual=dual_residual,**stats))
            if max(stats['relative_feasibility'],stats['relative_stationarity'],stats['relative_l1_subgradient'])<=tolerance:
                converged=True;break
            # Rescale the scaled multiplier to preserve the unscaled dual variable.
            if step%10==0:
                new_rho=min(rho*2,1e6) if primal>10*dual_residual else max(rho/2,1e-6) if dual_residual>10*primal else rho
                if new_rho!=rho:
                    dual*=rho/new_rho;rho=new_rho;adjustments+=1
        if seconds is not None and time.perf_counter()-start>=seconds:break
    final=diagnostics(p,g,c,z,a,dual,penalty,rho)
    state=dict(z=z,a=a,dual=dual,rho=rho,step=step,penalty=penalty)
    return z,state,dict(converged=converged,steps=local,total_steps=step,seconds=time.perf_counter()-start,
                       rho_adjustments=adjustments,final=final,history=history)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1049)
    p=torch.linalg.qr(torch.randn(13,4)).Q;atoms=torch.randn(3,6);g=atoms@atoms.T;c=torch.randn(4,3);penalty=.12
    reference,old=fixed_solve(p,g,c,penalty,1.,10000,1e-9)
    z,state,report=solve(p,g,c,penalty,rho=.001,max_steps=10000,tolerance=1e-9)
    error=abs(old['final']['objective']-report['final']['objective'])/max(1.,abs(old['final']['objective']))
    _,whole,_=solve(p,g,c,penalty,rho=.001,max_steps=40,tolerance=0.)
    _,half,_=solve(p,g,c,penalty,rho=.001,max_steps=20,tolerance=0.)
    _,resumed,_=solve(p,g,c,penalty,state=half,max_steps=20,tolerance=0.)
    resume_error=max(float((whole[key]-resumed[key]).abs().max()) for key in ['z','a','dual'])
    result=dict(instrument_passed=report['converged'] and old['converged'] and error<1e-8 and resume_error==0 and whole['rho']==resumed['rho'],
                fixed_rho_reference_objective_error=error,exact_resume_state_error=resume_error,
                adaptive_rho_changes=report['rho_adjustments'],final=report['final'],
                scope='Same convex objective, fixed-solver agreement and exact resume control. Adaptive residual balancing does not itself certify convergence; original residual checks still required.')
    Path(__file__).with_name('LEGAL_TOKEN_WRITER_ADMM_V2_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
