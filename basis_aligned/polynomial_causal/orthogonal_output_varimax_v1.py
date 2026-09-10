"""Weight-only output factor rotation; sparse overlapping usage, no clustering.

Loadings and quadratic functions rotate together, preserving the tensor.
The varimax criterion does not imply statistical independence or native circuits.
"""
import math,time,json
import torch
from joint_quadratic_fit_v1 import product_cross
from sparse_core_stiefel_v1 import project_tangent,retract


def output_factors(u,l,r,d,count):
    mean=u.mean(0);uc=u-mean
    chol=torch.linalg.cholesky(uc.T@uc)
    z=chol.T@d
    native_gram=product_cross(l,r,l,r)
    covariance=z@native_gram@z.T;covariance=(covariance+covariance.T)/2
    values,vectors=torch.linalg.eigh(covariance)
    values=values.flip(0);vectors=vectors.flip(1)
    ev=values[:count];v=vectors[:,:count]
    writer=torch.linalg.solve_triangular(chol.T,v*ev.sqrt(),upper=True)
    core=(v.T@z)/ev.sqrt()[:,None]
    loadings=uc@writer
    return dict(loadings=loadings,writer=writer,core=core,mean=mean,
                eigenvalues=values,native_gram=native_gram)


def criterion(x,rotation,gradient=False):
    b=x@rotation
    second=b.square().mean(0)
    score=(b.pow(4).mean(0)-second.square()).sum()
    if not gradient:return score
    euclidean=4*x.T@(b.pow(3)-b*second)/len(x)
    return score,project_tangent(rotation,euclidean)


def fit(loadings,seconds=240,max_steps=100000,state=None):
    # Global scaling fixes numerical units without changing the maximizer.
    scale=loadings.square().mean().sqrt();x=loadings/scale
    state=state or {};rank=x.shape[1]
    rotation=state.get('rotation',torch.eye(rank,dtype=x.dtype,device=x.device)).clone()
    prev_g=state.get('previous_gradient');prev_p=state.get('previous_direction');hint=state.get('step_hint')
    iteration=state.get('iteration',0);first_iteration=iteration;history=list(state.get('history',[]))
    start=time.perf_counter();converged=False;reason='budget_limit';backtracks=restarts=0;max_decrease=0.
    score,g=criterion(x,rotation,True)
    def diagnose():
        nonlocal converged
        row=dict(iteration=iteration,score=float(score),relative_stationarity=float(g.norm()*math.sqrt(rank)/score.abs().clamp_min(1e-12)),gradient_max_abs=float(g.abs().max()),orthogonality_error=float((rotation.T@rotation-torch.eye(rank,device=x.device,dtype=x.dtype)).abs().max()))
        if not all(math.isfinite(v) for v in row.values()) or row['orthogonality_error']>1e-10:raise ArithmeticError('Invalid varimax state')
        if history and history[-1]['iteration']==iteration:history[-1]=row
        else:history.append(row)
        if len(history)>=5:
            row['relative_progress_five_checks']=abs(row['score']-history[-5]['score'])/max(abs(row['score']),1e-12)
            converged=row['relative_progress_five_checks']<=1e-7 and row['relative_stationarity']<=1e-5
        if iteration%5==0 or converged:print(json.dumps(row),flush=True)
    diagnose()
    while time.perf_counter()-start<seconds and iteration-first_iteration<max_steps and not converged:
        direction=g
        if prev_g is not None:
            transported=project_tangent(rotation,prev_g)
            beta=max(0.,min(10.,float((g*(g-transported)).sum()/prev_g.square().sum().clamp_min(1e-30))))
            direction=g+beta*project_tangent(rotation,prev_p)
            if float((g*direction).sum())<.1*float(g.square().sum()):direction=g;restarts+=1
        slope=float((g*direction).sum())
        step=.01*abs(float(score))/max(slope,1e-30) if hint is None else hint
        step=min(step,1/max(float(direction.norm()),1e-30),1e6)
        accepted=False
        for _ in range(25):
            trial=retract(rotation,direction,step);new_score=criterion(x,trial)
            if float(new_score-score)>=1e-4*step*slope:accepted=True;break
            step*=.5;backtracks+=1
        if not accepted:reason='line_search_failed';break
        prev_g=g;prev_p=direction;hint=1.5*step;old=float(score)
        rotation=trial;score,g=criterion(x,rotation,True);iteration+=1
        max_decrease=max(max_decrease,old-float(score))
        if max_decrease>1e-10:raise ArithmeticError('Varimax score decreased')
        diagnose()
    if history[-1]['iteration']!=iteration:diagnose()
    if converged:reason='projected_stationarity_and_plateau'
    return dict(rotation=rotation,previous_gradient=prev_g,previous_direction=prev_p,
                step_hint=hint,iteration=iteration,history=history,converged=converged,
                terminal_reason=reason,chunk_seconds=time.perf_counter()-start,
                chunk_backtracks=backtracks,chunk_direction_restarts=restarts,maximum_score_decrease=max_decrease)


def sparsity(loadings):
    # Row participation: effective number of factors used by each token.
    square=loadings.square();row=square.sum(1)
    nonzero=row>1e-30
    participation=(row.square()/square.square().sum(1).clamp_min(1e-30))[nonzero]
    col=square.sum(0)
    support=col.square()/square.square().sum(0).clamp_min(1e-30)
    return dict(nonzero_token_rows=int(nonzero.sum()),zero_token_rows=int((~nonzero).sum()),
                median_token_factor_participation=float(participation.median()),
                mean_token_factor_participation=float(participation.mean()),
                median_factor_token_participation=float(support.median()),
                token_top4_energy_fraction=float(square.topk(min(4,loadings.shape[1]),dim=1).values.sum()/row.sum()),
                padded_row_energy_fraction=float(square[50257:].sum()/square.sum()))
