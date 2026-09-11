"""Exact reduced curvature, including the conditional writer's response.

The minimized loss is 1-<M K, K R^-1>/total, with R=G+lambda diag(G).
Differentiating its solve is essential at second order. A detached conditional
writer supplies the first envelope gradient but not the reduced Hessian.
"""
import json
from pathlib import Path
import torch
from orthogonal_multioutput_pymanopt_v2 import View,evaluate
from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective
from joint_quadratic_fit_v1 import product_cross


def reduced_loss(objective,bank,packed,outputs):
    cross,gram=objective.cross_gram(View(bank,packed,outputs))
    reg=gram+objective.penalty*torch.diag(gram.diag())
    writer=torch.linalg.solve(reg,cross.T).T
    return 1-((objective.metric@cross)*writer).sum()/objective.total


def hessian_vector(objective,bank,packed,outputs,direction,detached_writer=False):
    point=[bank.detach().requires_grad_(),packed.detach().requires_grad_()]
    if detached_writer:loss=objective.terms(View(*point,outputs))[0]
    else:loss=reduced_loss(objective,*point,outputs)
    grad=torch.autograd.grad(loss,point,create_graph=True)
    product=sum((g*v).sum() for g,v in zip(grad,direction))
    hv=torch.autograd.grad(product,point)
    return [x.detach() for x in hv]


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1523)
    bank=torch.linalg.qr(torch.randn(2,7,3)).Q;packed=torch.randn(6,8);packed/=packed.norm(dim=0)
    l,r=torch.randn(2,11,7);d=torch.randn(7,11);u=torch.randn(13,7);metric=u.T@u
    total=((d.T@metric@d)*product_cross(l,r,l,r)).sum()
    obj=MultioutputWeightObjective(metric,total,l=l,r=r,d=d,penalty=.01)
    old,oldgrad,_=evaluate(obj,bank,packed,4)
    point=[bank.clone().requires_grad_(),packed.clone().requires_grad_()]
    loss=reduced_loss(obj,*point,4);grad=torch.autograd.grad(loss,point)
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(grad,oldgrad))
    directions=[]
    for _ in range(2):
        ds=[torch.randn_like(x) for x in point];norm=sum(x.square().sum() for x in ds).sqrt();directions.append([x/norm for x in ds])
    v,w=directions;hv=hessian_vector(obj,bank,packed,4,v);hw=hessian_vector(obj,bank,packed,4,w)
    eps=1e-5
    plus=evaluate(obj,bank+eps*v[0],packed+eps*v[1],4)[1]
    minus=evaluate(obj,bank-eps*v[0],packed-eps*v[1],4)[1]
    fd=[(a-b)/(2*eps) for a,b in zip(plus,minus)]
    norm=lambda xs:sum(x.square().sum() for x in xs).sqrt()
    fd_error=float(norm([a-b for a,b in zip(hv,fd)])/norm(fd))
    symmetry=abs(sum(float((a*b).sum()) for a,b in zip(w,hv))-sum(float((a*b).sum()) for a,b in zip(v,hw)))
    frozen=hessian_vector(obj,bank,packed,4,v,detached_writer=True)
    wrong_error=float(norm([a-b for a,b in zip(frozen,fd)])/norm(fd))
    result=dict(instrument_passed=max(abs(float(loss.detach())-old),gradient_error,symmetry)<1e-8 and fd_error<1e-6 and wrong_error>1e-3,
        minimized_loss_identity_error=abs(float(loss.detach())-old),envelope_gradient_error=gradient_error,
        hessian_vector_finite_difference_relative_error=fd_error,hessian_symmetry_error=symmetry,
        detached_writer_hessian_relative_error=wrong_error,
        scope='Exact reduced Euclidean Hessian-vector tool and live incorrect-Hessian control; native curvature and trust-region optimization untested.')
    with Path(__file__).with_name('BLOCK_VARIABLE_PROJECTION_CURVATURE_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']

if __name__=='__main__':control()
