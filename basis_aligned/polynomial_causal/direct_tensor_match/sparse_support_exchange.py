"""Exact one-product exchange for a fixed shared-feature dictionary.
The ridge-regularized multi-output readout is refitted after each edit.
"""
import torch


def fitted_score(G,X,selected,ridge):
    K=G[selected][:,selected]+ridge*torch.eye(len(selected),dtype=G.dtype,device=G.device)
    C=torch.linalg.solve(K,X[:,selected].T).T
    return (C*X[:,selected]).sum(),C


def best_exchange(G,X,selected,protected=(),ridge=1e-6):
    n=len(G);mask=torch.ones(n,dtype=torch.bool,device=G.device);mask[selected]=False
    candidates=torch.arange(n,device=G.device)[mask]
    if not len(candidates):return None
    K=G[selected][:,selected]+ridge*torch.eye(len(selected),dtype=G.dtype,device=G.device)
    inverse=torch.cholesky_inverse(torch.linalg.cholesky(K));C=X[:,selected]@inverse
    off=G[selected][:,candidates];v=inverse@off
    schur=G.diag()[candidates]+ridge-(off*v).sum(0)
    assert bool((schur>0).all()),'Regularized augmented Gram must be positive definite'
    residual=X[:,candidates]-C@off;new=residual/schur
    gain=residual.square().sum(0)/schur
    changed=C[:,:,None]-new[:,None,:]*v[None,:,:]
    removal=changed.square().sum(0)/(inverse.diag()[:,None]+v.square()/schur)
    if len(protected):
        fixed=(selected[:,None]==torch.as_tensor(protected,device=G.device)[None,:]).any(1)
        removal[fixed]=float('inf')
    costs,remove=removal.min(0);improvement=gain-costs;j=int(improvement.argmax());i=int(remove[j])
    return dict(remove=int(selected[i]),add=int(candidates[j]),position=i,predicted_score_gain=float(improvement[j]))


def exchange_path(G,X,selected,protected=(),ridge=1e-6,steps=16):
    selected=selected.clone();initial,C=fitted_score(G,X,selected,ridge);history=[];score=initial
    for step in range(steps):
        proposal=best_exchange(G,X,selected,protected,ridge)
        if proposal is None or proposal['predicted_score_gain']<=1e-12*(1+abs(float(score))):break
        next_selected=selected.clone();next_selected[proposal['position']]=proposal['add']
        newscore,newC=fitted_score(G,X,next_selected,ridge)
        actual=float(newscore-score);gap=abs(actual-proposal['predicted_score_gain'])/(1+abs(float(score)))
        assert gap<1e-8 and actual>=-1e-10*(1+abs(float(score))),(proposal,actual,gap)
        history.append(dict(step=step+1,**proposal,actual_score_gain=actual,formula_replay=gap,score=float(newscore)))
        selected,C,score=next_selected,newC,newscore
    return selected,C,dict(initial_score=float(initial),final_score=float(score),history=history)
