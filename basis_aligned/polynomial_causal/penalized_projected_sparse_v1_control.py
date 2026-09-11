"""Dense end-to-end differentiation control for the penalized sparse kernel."""
import json
from pathlib import Path
import torch
from penalized_projected_sparse_v1 import value_gradient,output_solve
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(481)
    native=(torch.randn(9,7),torch.randn(9,7),torch.randn(4,9))
    ids=torch.stack([torch.randperm(8)[:3] for _ in range(12)])
    rb=torch.randn(8,7);rv=torch.randn(12,3);rows=[]
    for singular in (False,True):
        wh=torch.eye(4);wh[-1,-1]=0. if singular else 1.
        target=dense(native[0],native[1],wh@native[2]);total=target.square().sum()
        for penalty in (1e-4,.01):
            point=[rb.clone().requires_grad_(),rv.clone().requires_grad_()]
            basis=point[0]/point[0].norm(dim=1,keepdim=True)
            vals=point[1]/point[1].norm(dim=1,keepdim=True)
            codes=torch.zeros(12,8).scatter(1,ids,vals)
            a,b=(codes@basis).chunk(2)
            writer,_=output_solve(native[2],native[0],native[1],a,b,penalty)
            predicted=dense(a,b,wh@writer)
            component=(torch.einsum('ok,ki,kj->koij',wh@writer,a,b)+torch.einsum('ok,ki,kj->koji',wh@writer,a,b))/2
            loss=((predicted-target).square().sum()+penalty*component.square().sum())/total
            exact=torch.autograd.grad(loss,point)
            actual,gradient,_,details=value_gradient(native,wh,rb,ids,rv,total,penalty,chunk=2)
            gradient_error=max(float((g-e).norm()/e.norm().clamp_min(1e-30)) for g,e in zip(gradient,exact))
            direction=[torch.randn_like(x) for x in point];norm=sum(x.square().sum() for x in direction).sqrt();direction=[x/norm for x in direction]
            h=1e-5
            plus=value_gradient(native,wh,rb+h*direction[0],ids,rv+h*direction[1],total,penalty)[0]
            minus=value_gradient(native,wh,rb-h*direction[0],ids,rv-h*direction[1],total,penalty)[0]
            fd=abs(float((plus-minus)/(2*h)-sum((g*d).sum() for g,d in zip(gradient,direction))))
            scales=torch.logspace(-2,2,12)[:,None]
            changed,_,_,other=value_gradient(native,wh,rb,ids,rv*scales,total,penalty)
            station_error=max(abs(details[k]-other[k]) for k in ('dictionary_relative_stationarity','codes_relative_stationarity'))
            rows.append(dict(singular_metric=singular,penalty=penalty,loss_error=abs(float(actual-loss.detach())),
                gradient_error=gradient_error,fd_error=fd,row_scale_loss_error=abs(float(changed-actual)),
                row_scale_stationarity_error=station_error,normal_residual=details['output_solve']['normal_residual']))
    passed=dict(pred_a_dense=max(max(r['loss_error'],r['gradient_error'],r['normal_residual']) for r in rows)<=1e-8,
                pred_b_fd=max(r['fd_error'] for r in rows)<=1e-6,
                pred_c_scale=max(max(r['row_scale_loss_error'],r['row_scale_stationarity_error']) for r in rows)<=1e-8)
    result=dict(predictions=passed,rows=rows,scope='Small exact integration and scale checks; no native convergence or global recovery.')
    with Path(__file__).with_name('PENALIZED_PROJECTED_SPARSE_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert all(passed.values())


if __name__=='__main__':main()
