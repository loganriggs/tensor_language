"""Exact one-term backward deletion with refitted linear outputs in a fixed metric."""
import torch

def prune(G,X,budgets,ridge=1e-10):
    n=len(G);ids=torch.arange(n,device=G.device);history=[];snapshots={};normalizer=G.diag().clamp_min(1e-30).sqrt();g=G/(normalizer[:,None]*normalizer[None,:]);x=X/normalizer
    def solve(ids):
        K=g[ids][:,ids]+ridge*torch.eye(len(ids),dtype=G.dtype,device=G.device)
        inverse=torch.linalg.inv(K);C=x[:,ids]@inverse
        return inverse,C
    inverse,C=solve(ids);refits=0
    while True:
        if len(ids) in budgets:
            inverse,C=solve(ids);res=float((C@(g[ids][:,ids]+ridge*torch.eye(len(ids),dtype=G.dtype,device=G.device))-x[:,ids]).norm()/x[:,ids].norm().clamp_min(1e-30));assert res<1e-7
            snapshots[len(ids)]=dict(indices=ids.clone(),coefficients=C/normalizer[ids],normal_residual=res)
        if len(ids)<=min(budgets):break
        costs=C.square().sum(0)/inverse.diag();remove=int(costs.argmin());keep=torch.arange(len(ids),device=G.device)!=remove
        col=inverse[keep,remove];pivot=inverse[remove,remove];before=float((C*x[:,ids]).sum());prediction=float(costs[remove])
        C=C[:,keep]-C[:,remove,None]*(col/pivot)[None,:]
        inverse=inverse[keep][:,keep]-col[:,None]*col[None,:]/pivot;ids=ids[keep]
        after=float((C*x[:,ids]).sum());err=abs((before-after)-prediction)/(1+abs(before));assert err<1e-7
        history.append(dict(remaining=len(ids),predicted_loss_increase=prediction,formula_error=err))
        if len(history)%16==0:inverse,C=solve(ids);refits+=1
    return snapshots,history


def controls():
    rows=[]
    for seed in range(5):
        torch.manual_seed(12700+seed);A=torch.randn(17,9,dtype=torch.float64);G=A.T@A;X=torch.randn(3,9,dtype=torch.float64)@G;ridge=1e-10
        snapshots,history=prune(G,X,[9,8,4],ridge)
        norm=G.diag().sqrt();g=G/(norm[:,None]*norm[None,:]);x=X/norm
        scores=[]
        for dropped in range(9):
            ids=torch.tensor([i for i in range(9) if i!=dropped]);C=torch.linalg.solve(g[ids][:,ids]+ridge*torch.eye(8,dtype=G.dtype),x[:,ids].T).T;scores.append(float((C*x[:,ids]).sum()))
        actual=set(snapshots[8]['indices'].tolist());expected=set(range(9))-{int(torch.tensor(scores).argmax())};assert actual==expected
        rows.append(dict(seed=seed,first_deletion_exhaustive_match=True,max_formula_error=max(r['formula_error'] for r in history)))
    return rows
