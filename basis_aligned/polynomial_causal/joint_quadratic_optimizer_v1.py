"""Variable-projection optimization of shared bilinear products, full-output metric.
W is solved exactly at each step; envelope gradients treat its optimum as fixed.
"""
import torch
import torch.nn.functional as F
from joint_quadratic_fit_v1 import product_cross

def solve(l,r,d,a,b):
    cross=product_cross(l,r,a,b);gram=product_cross(a,b,a,b)
    w=torch.linalg.solve(gram,(d@cross).T).T
    return w,cross,gram

def fit(l,r,d,metric,a0,b0,total,steps=240,lr=.03):
    pa=torch.nn.Parameter(a0.clone());pb=torch.nn.Parameter(b0.clone());opt=torch.optim.Adam([pa,pb],lr=lr)
    history=[];best=None;bestloss=float('inf');dtm=d.T@metric
    for step in range(steps+1):
        a=F.normalize(pa,dim=1);b=F.normalize(pb,dim=1);cross=product_cross(l,r,a,b);gram=product_cross(a,b,a,b)
        with torch.no_grad():w=torch.linalg.solve(gram,(d@cross).T).T
        loss=(total+((w.T@metric@w)*gram).sum()-2*((dtm@w)*cross).sum())/total
        value=float(loss.detach());assert torch.isfinite(loss) and value>=-1e-8
        if value<bestloss:
            bestloss=value;best=dict(a=a.detach().clone(),b=b.detach().clone(),w=w.detach().clone(),step=step)
        if step%30==0 or step==steps:history.append(dict(step=step,squared_relative_error=value))
        if step==steps:break
        opt.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in [pa,pb])
        opt.step()
    with torch.no_grad():
        gram=product_cross(best['a'],best['b'],best['a'],best['b']);condition=float(torch.linalg.cond(gram))
    return best,dict(history=history,best_squared_relative_error=bestloss,condition=condition,steps=steps,lr=lr)
