"""Resumable Adam/L-BFGS fitting; a managed chunk ending is not convergence."""
import time
import torch

def advance(model,objective,checkpoint=None,seconds=540,adam_steps=2000,diagnostic_every=50):
    state=checkpoint or {};phase=state.get('phase','adam');adam_done=state.get('adam_done',0);lbfgs_done=state.get('lbfgs_done',0)
    history=state.get('history',[]);best=state.get('best');start=time.perf_counter();closures=state.get('closures',0)
    def optimizer_for(p):
        if p=='adam':return torch.optim.Adam(model.parameters(),lr=.03)
        return torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=10,history_size=8,tolerance_grad=1e-10,tolerance_change=1e-13,line_search_fn='strong_wolfe')
    opt=optimizer_for(phase)
    if state.get('optimizer') is not None:opt.load_state_dict(state['optimizer'])
    converged=False;reason='chunk_time_limit';updates=0
    def diagnose():
        nonlocal best,converged,reason
        diag,w=objective.diagnostics(model)
        if not all(__import__('math').isfinite(v) for v in diag.values()) or diag['gram_condition']>1e12:
            raise ArithmeticError('Nonfinite or ill-conditioned fitting instrument')
        if diag['squared_relative_error'] < -1e-8:raise ArithmeticError('Negative squared-error instrument')
        record=dict(phase=phase,adam_done=adam_done,lbfgs_done=lbfgs_done,closures=closures,**diag)
        history.append(record)
        if best is None or diag['squared_relative_error']<best['diagnostics']['squared_relative_error']:
            best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},writer=w.cpu(),diagnostics=record)
        # Stable objective across five L-BFGS diagnostics plus an independently
        # computed full gradient. Do not count a max_iter allowance as actual work.
        if phase=='lbfgs':record['lbfgs_actual_inner_iterations']=int(opt.state[next(iter(model.parameters()))].get('n_iter',0))
        prior=[r for r in history[:-1] if r['phase']=='lbfgs']
        old=prior[-4] if len(prior)>=4 else None
        if old is not None and phase=='lbfgs':
            relative_progress=abs(old['squared_relative_error']-diag['squared_relative_error'])/diag['captured_energy_fraction']
            record['relative_progress_over_five_lbfgs_checks']=relative_progress
            converged=phase=='lbfgs' and relative_progress<=1e-5 and diag['relative_stationarity']<=1e-4 and diag['gradient_max_abs']<=1e-7
        if converged:reason='plateau_and_stationarity'
        print(__import__('json').dumps(record),flush=True)
    diagnose()
    while time.perf_counter()-start<seconds and not converged:
        if phase=='adam' and adam_done>=adam_steps:phase='lbfgs';opt=optimizer_for(phase)
        def closure():
            nonlocal closures
            opt.zero_grad(set_to_none=True);loss,_,_=objective.loss(model)
            if not torch.isfinite(loss):raise ArithmeticError('Nonfinite fit loss')
            loss.backward();closures+=1;return loss
        if phase=='adam':closure();opt.step();adam_done+=1
        else:opt.step(closure);lbfgs_done+=1
        updates+=1
        period=diagnostic_every if phase=='adam' else 5
        if updates%period==0:diagnose()
    if updates:diagnose()
    # Keep resumable optimizer state only while unfinished. A converged checkpoint
    # retains model, best writer and complete diagnostics rather than obsolete history vectors.
    return dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=None if converged else opt.state_dict(),
        phase=phase,adam_done=adam_done,lbfgs_done=lbfgs_done,closures=closures,history=history,best=best,
        converged=converged,terminal_reason=reason,chunk_updates=updates,chunk_seconds=time.perf_counter()-start)
