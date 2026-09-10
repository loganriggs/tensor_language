"""Riemannian PR+ ascent for a sparse symmetric quadratic core.

Piecewise smooth support selection; no global convergence claim. Exact
fixed-support ascent lower-bounds reselected support score.
"""
import time,math,torch
from sparse_orthogonal_quadratic_core_v1 import orthogonal_core
from sparse_core_stiefel_v1 import project_tangent,score_gradient,armijo_step

def fit(q,l,r,d,total,count=256,seconds=240,max_steps=100000,state=None):
    state=state or {};q=state.get('q',q).clone();previous_gradient=state.get('previous_gradient');previous_direction=state.get('previous_direction');step_hint=state.get('step_hint')
    iteration=state.get('iteration',0);first_iteration=iteration;history=list(state.get('history',[]));backtracks=0;selections=0;converged=False;start=time.perf_counter();reason='budget_limit';max_decrease=0.;restarts=0
    def evaluate():
        nonlocal selections
        with torch.no_grad():
            w,edges=orthogonal_core(l,r,d,q.T);energy=w.square().sum(0);values,indices=torch.topk(energy,count+1);chosen=edges[:,indices[:count]];score=values[:count].sum()/total;margin=(values[count-1]-values[count])/total
        checked,g=score_gradient(q,l,r,d,chosen,total);selections+=1
        if abs(float(checked-score))>1e-10:raise ArithmeticError('Core score and gradient objective mismatch')
        return score,g,chosen,margin
    score,g,edges,margin=evaluate()
    initial=float(score)
    def diagnose():
        nonlocal converged
        row=dict(iteration=iteration,captured_energy=float(score),relative_stationarity=float(g.norm()*math.sqrt(q.shape[1])/score.clamp_min(1e-12)),gradient_max=float(g.abs().max()),support_margin=float(margin),orthogonality_error=float((q.T@q-torch.eye(q.shape[1],device=q.device,dtype=q.dtype)).abs().max()))
        if row['orthogonality_error']>1e-10 or not all(math.isfinite(v) for v in row.values()):raise ArithmeticError('Invalid frame diagnostics')
        if history and history[-1]['iteration']==iteration:history[-1]=row
        else:history.append(row)
        if len(history)>=5:
            row['relative_progress_five_checks']=abs(row['captured_energy']-history[-5]['captured_energy'])/row['captured_energy']
            converged=row['relative_progress_five_checks']<=1e-5 and row['relative_stationarity']<=1e-4 and row['gradient_max']<=1e-7 and row['support_margin']>1e-12
        print(__import__('json').dumps(row),flush=True)
    diagnose()
    while time.perf_counter()-start<seconds and iteration-first_iteration<max_steps and not converged:
        direction=g
        if previous_gradient is not None:
            transported=project_tangent(q,previous_gradient)
            beta=max(0.,min(10.,float((g*(g-transported)).sum()/previous_gradient.square().sum().clamp_min(1e-30))))
            direction=g+beta*project_tangent(q,previous_direction)
            if float((g*direction).sum())<.1*float(g.square().sum()):direction=g;restarts+=1
        slope=float((g*direction).sum())
        hint=.01*float(score)/max(slope,1e-30) if step_hint is None else step_hint
        hint=min(hint,1e6,1/max(float(direction.norm()),1e-30))
        trial,receipt=armijo_step(q,g,direction,l,r,d,edges,total,initial_step=hint)
        backtracks+=receipt['backtracks']
        if not receipt['accepted']:reason='line_search_failed';break
        previous_gradient=g;previous_direction=direction;step_hint=receipt['step']*1.5
        old=float(score);q=trial;iteration+=1
        score,g,edges,margin=evaluate();max_decrease=max(max_decrease,old-float(score))
        if max_decrease>1e-10:raise ArithmeticError('Selected score decreased after guaranteed ascent')
        if iteration%5==0:diagnose()
    if not history or history[-1]['iteration']!=iteration:diagnose()
    if converged:reason='plateau_stationarity_and_support_gap'
    return dict(q=q,previous_gradient=previous_gradient,previous_direction=previous_direction,step_hint=step_hint,iteration=iteration,history=history,edges=edges,converged=converged,terminal_reason=reason,chunk_seconds=time.perf_counter()-start,chunk_selections=selections,chunk_backtracks=backtracks,chunk_direction_restarts=restarts,initial_capture=initial,maximum_score_decrease=max_decrease)
