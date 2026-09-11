"""Chunked full folded tensor objective with physical whole-group penalty.

A: groups x input; V: groups x rank x input; W: groups x output x rank.
The computation is sum_j (A_j x) W_j V_j x. No normalization assumed here.
"""
import json
from pathlib import Path
import torch
from chunked_bilinear_coefficient_v1 import value_gradient as cp_value_gradient, dense


def cp(a, v, w):
    m,r,d=v.shape
    return a[:,None,:].expand(m,r,d).reshape(m*r,d),v.reshape(m*r,d),w.permute(1,0,2).reshape(w.shape[1],m*r)


@torch.no_grad()
def value_gradient(target, a, v, w, total, penalty=0., chunk=256):
    m,r,d=v.shape
    loss,(ga,gv,gw),details=cp_value_gradient(*target,*cp(a,v,w),total,penalty=0.,chunk=chunk)
    ga=ga.reshape(m,r,d).sum(1);gv=gv.reshape_as(v)
    gw=gw.reshape(w.shape[1],m,r).permute(1,0,2)
    gramw=w.transpose(1,2)@w;gramv=v@v.transpose(1,2)
    va=(v@a[:,:,None]).squeeze(-1)
    gva=(gramw@va[:,:,None]).squeeze(-1)
    normm=(gramw*gramv).sum((1,2));norma=a.square().sum(1)
    energy=.5*(norma*normm+(va*gva).sum(1))
    scale=penalty/total
    ga+=scale*(normm[:,None]*a+(v.transpose(1,2)@gva[:,:,None]).squeeze(-1))
    gv+=scale*(norma[:,None,None]*(gramw@v)+gva[:,:,None]*a[:,None,:])
    gw+=scale*(norma[:,None,None]*(w@gramv)+(w@va[:,:,None])*va[:,None,:])
    details['group_energy']=float(energy.sum()/total)
    details['group_energy_by_group']=(energy/total).tolist()
    return loss+penalty*energy.sum()/total,(ga,gv,gw),details


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1437)
    target=(torch.randn(11,9),torch.randn(11,9),torch.randn(7,11))
    truth=dense(*target);total=truth.square().sum()
    a=torch.randn(3,9,requires_grad=True)
    v=torch.randn(3,2,9,requires_grad=True)
    w=torch.randn(3,7,2,requires_grad=True)
    rows=[]
    for penalty in [0.,.01]:
        forms=[dense(*cp(a[j:j+1],v[j:j+1],w[j:j+1])) for j in range(3)]
        objective=((sum(forms)-truth).square().sum()+penalty*sum(f.square().sum() for f in forms))/total
        automatic=torch.autograd.grad(objective,(a,v,w))
        value,gradient,details=value_gradient(target,a,v,w,total,penalty,chunk=2)
        errors=[float((g-exact).norm()/exact.norm()) for g,exact in zip(gradient,automatic)]
        h=torch.tensor([[2.,.7],[-.3,.5]]).expand(3,-1,-1)
        newv=h@v.detach();neww=w.detach()@torch.linalg.inv(h)
        transformed,_,newdetails=value_gradient(target,a.detach(),newv,neww,total,penalty,chunk=2)
        rows.append(dict(penalty=penalty,value_error=float(abs(value-objective.detach())),gradient_errors=errors,
                         partner_gauge_loss_error=float(abs(value-transformed)),
                         partner_gauge_energy_error=abs(details['group_energy']-newdetails['group_energy']),
                         partner_gauge_function_error=float((dense(*cp(a,v,w))-dense(*cp(a,newv,neww))).norm()/truth.norm())))
    error=max(max(row['value_error'],*row['gradient_errors'],row['partner_gauge_loss_error'],row['partner_gauge_energy_error'],row['partner_gauge_function_error']) for row in rows)
    result=dict(predictions=dict(pred_a_dense_gradient=error<=1e-9,pred_b_partner_gauge=error<=1e-9),rows=rows,maximum_error=error,
                scope='Exact group-coupled objective and whole-group cancellation penalty. No optimization or native recovery claim.')
    Path(__file__).with_name('SHARED_READER_GROUP_OBJECTIVE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert error<=1e-9


if __name__=='__main__':control()
