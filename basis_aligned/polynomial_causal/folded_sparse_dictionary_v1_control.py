"""Dense/autograd controls and an analytic reader-proxy cancellation example."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from chunked_bilinear_coefficient_v1 import dense
from folded_sparse_dictionary_v1 import decode,loss_gradient


def control():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(921)
    native=(torch.randn(7,5),torch.randn(7,5),torch.randn(4,7));target=dense(*native);total=target.square().sum()
    raw=torch.randn(8,5,requires_grad=True);values=torch.randn(12,3,requires_grad=True)
    ids=torch.stack([torch.randperm(8)[:3] for _ in range(12)]);scale=torch.rand(12)+.5;writer=torch.randn(4,6)
    readers,basis,code,_=decode(raw,ids,values,scale)
    direct=(dense(*readers.chunk(2),writer)-target).square().sum()/total
    expected=torch.autograd.grad(direct,(raw,values));rows=[]
    for chunk in (1,4,16):
        loss,grad,details=loss_gradient(native,writer,raw,ids,values,scale,total,chunk)
        rows.append(dict(chunk=chunk,loss_error=abs(float(loss-direct.detach())),
            gradient_errors=[float((a-b).norm()/b.norm()) for a,b in zip(grad,expected)]))
    direction=[torch.randn_like(raw),torch.randn_like(values)]
    length=sum(x.square().sum() for x in direction).sqrt();direction=[x/length for x in direction]
    h=1e-5
    upper=loss_gradient(native,writer,raw+h*direction[0],ids,values+h*direction[1],scale,total)[0]
    lower=loss_gradient(native,writer,raw-h*direction[0],ids,values-h*direction[1],scale,total)[0]
    exact=sum((g*d).sum() for g,d in zip(expected,direction))
    finite_error=float(abs((upper-lower)/(2*h)-exact)/exact.abs().clamp_min(1.))
    rescaled=raw.detach()*torch.linspace(.5,2.,8)[:,None]
    same=decode(rescaled,ids,values.detach(),scale)[0]
    gauge_error=float((same-readers.detach()).norm()/readers.norm())
    # q=x1*x2-(x1+x2)*x1=-x1^2. Independent LS under these supports gives -x1*x2.
    na=torch.tensor([[1.,0.],[1.,1.]]);nb=torch.tensor([[0.,1.],[1.,0.]])
    w=torch.tensor([[1.,-1.]]);native2=(na,nb,w);truth=dense(*native2);energy=truth.square().sum()
    identity=torch.eye(2);support=torch.tensor([[0],[1],[0],[0]])
    initial=torch.tensor([[1.],[1.],[0.],[1.]]);scale2=torch.ones(4)
    def objective(flat):
        v=torch.from_numpy(flat).reshape(4,1)
        loss,(_,g),_=loss_gradient(native2,w,identity,support,v,scale2,energy,chunk=2)
        return float(loss),g.flatten().numpy()
    initial_loss=objective(initial.flatten().numpy())[0]
    fitted=minimize(objective,initial.flatten().numpy(),jac=True,method='L-BFGS-B',
        options=dict(maxiter=1000,ftol=1e-15,gtol=1e-11,maxls=30))
    final=torch.from_numpy(fitted.x).reshape(4,1)
    original_readers=torch.cat((na,nb))
    before=decode(identity,support,initial,scale2)[0];after=decode(identity,support,final,scale2)[0]
    before_proxy=float((before-original_readers).square().sum())
    after_proxy=float((after-original_readers).square().sum())
    cancellation=dict(initial_function_error=initial_loss,final_function_error=fitted.fun,
        initial_reader_squared_error=before_proxy,final_reader_squared_error=after_proxy,
        optimized_values=fitted.x.tolist(),solver_success=bool(fitted.success),iterations=int(fitted.nit),
        maximum_code_gradient=float(np.max(np.abs(fitted.jac))),
        dense_final_error=float((dense(*after.chunk(2),w)-truth).square().sum()/energy))
    predictions=dict(pred_a_dense_gradient=max(max(r['loss_error'],*r['gradient_errors']) for r in rows)<=1e-9,
        pred_b_finite_and_gauge=finite_error<=1e-7 and gauge_error<=1e-10,
        pred_c_functional_recovery=cancellation['dense_final_error']<=1e-10 and cancellation['maximum_code_gradient']<=1e-8,
        pred_d_proxy_conflict=abs(initial_loss-1.5)<=1e-10 and after_proxy>before_proxy+1e-6)
    return dict(predictions=predictions,gradient_controls=rows,directional_fd_error=finite_error,
        gauge_replay_error=gauge_error,cancellation=cancellation,
        scope='Known small weight polynomial, fixed supports. Demonstrates reader-proxy conflict and exact chain rule; no native performance, convergence or circuit claim.')


if __name__=='__main__':
    result=control()
    with Path(__file__).with_name('FOLDED_SPARSE_DICTIONARY_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert all(result['predictions'].values())
