"""Dense differentiation through the writer solve versus the envelope chain."""
import json
from pathlib import Path
import torch
from projected_sparse_dictionary_v1 import value_gradient
from folded_sparse_dictionary_v1 import decode
from joint_quadratic_fit_v1 import product_cross
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(923)
    native=(torch.randn(7,5),torch.randn(7,5),torch.randn(4,7))
    raw=torch.randn(8,5,requires_grad=True);values=torch.randn(12,3,requires_grad=True)
    ids=torch.stack([torch.randperm(8)[:3] for _ in range(12)]);scale=torch.rand(12)+.5
    rows=[]
    for output_dimension in (7,3):
        whitener=torch.randn(output_dimension,4)
        target=dense(native[0],native[1],whitener@native[2]);total=target.square().sum()
        readers,_,_,_=decode(raw,ids,values,scale);a,b=readers.chunk(2)
        gram=product_cross(a,b,a,b);cross=product_cross(native[0],native[1],a,b)
        w=torch.linalg.solve(gram,(native[2]@cross).T).T
        direct=(dense(a,b,whitener@w)-target).square().sum()/total
        gradients=torch.autograd.grad(direct,(raw,values))
        found,g,writer,details=value_gradient(native,whitener,raw,ids,values,scale,total,chunk=4)
        errors=[float((actual-expected).norm()/expected.norm()) for actual,expected in zip(g,gradients)]
        directions=[torch.randn_like(raw),torch.randn_like(values)]
        length=sum(d.square().sum() for d in directions).sqrt();directions=[d/length for d in directions]
        h=1e-5
        up=value_gradient(native,whitener,raw+h*directions[0],ids,values+h*directions[1],scale,total)[0]
        dn=value_gradient(native,whitener,raw-h*directions[0],ids,values-h*directions[1],scale,total)[0]
        slope=sum((actual*d).sum() for actual,d in zip(g,directions))
        fd=float(abs((up-dn)/(2*h)-slope)/slope.abs().clamp_min(1.))
        rows.append(dict(output_dimension=output_dimension,output_metric_rank=int(torch.linalg.matrix_rank(whitener)),
            candidate_rank=details['writer_solve']['rank'],condition=details['writer_solve']['retained_condition'],
            writer_normal_residual=details['writer_solve']['normal_residual'],
            loss_error=abs(float(found-direct.detach())),gradient_errors=errors,directional_fd_error=fd))
    predictions=dict(pred_a_dense_gradient=all(max(r['loss_error'],*r['gradient_errors'])<=1e-8 for r in rows),
        pred_b_writer_solve=all(r['candidate_rank']==6 and r['writer_normal_residual']<=1e-9 for r in rows),
        pred_c_finite_difference=all(r['directional_fd_error']<=1e-6 for r in rows))
    result=dict(predictions=predictions,arms=rows,
        scope='Integration of prior variable projection and sparse shared-feature chain rule at fixed full product rank. No native cost, global convergence, rank-change smoothness or circuit claim.')
    with Path(__file__).with_name('PROJECTED_SPARSE_DICTIONARY_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert all(predictions.values())


if __name__=='__main__':main()
