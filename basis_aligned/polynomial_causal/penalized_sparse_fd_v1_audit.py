"""Repeat the native initial-direction FD with one tenth the step size.

Original PENALIZED_SPARSE_STEP_V1 numerical prediction remains failed.
A initial replay<=1e-10; B FD<=1e-6; C >=25x reduction versus h1e-5.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK
from penalized_projected_sparse_v1 import value_gradient


def main():
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'PENALIZED_SPARSE_FD_V1_AUDIT.json';assert not out.exists();started=time.perf_counter()
    original=json.loads((P/'PENALIZED_SPARSE_STEP_V1_AUDIT.json').read_text());source=original['source']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu')
    basis=saved['analysis_basis'].double();ids=saved['code_indices'].long();values=saved['code_values'].double()
    values/=values.norm(dim=1,keepdim=True)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    native=tuple(sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down'))
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    penalty=original['initial_details']['penalty']
    value,gradient,_,_=value_gradient(native,wh,basis,ids,values,total,penalty)
    directions=[-g/g.norm()*len(x)**.5 for g,x in zip(gradient,(basis,values))]
    slope=sum(float((g*d).sum()) for g,d in zip(gradient,directions));h=1e-6
    plus=value_gradient(native,wh,basis+h*directions[0],ids,values+h*directions[1],total,penalty)[0]
    minus=value_gradient(native,wh,basis-h*directions[0],ids,values-h*directions[1],total,penalty)[0]
    finite=float((plus-minus)/(2*h));error=abs(finite-slope)/max(1.,abs(slope))
    replay=abs(float(value)-original['initial_objective']);reduction=original['fd_error']/max(error,1e-30)
    result=dict(predictions=dict(pred_a_initial=replay<=1e-10,pred_b_fd=error<=1e-6,pred_c_truncation=reduction>=25),
        h=h,finite_difference=finite,analytic_slope=slope,fd_error=error,coarse_error=original['fd_error'],
        error_reduction=reduction,initial_replay=replay,seconds=time.perf_counter()-started,source=source,
        scope='Same native initial point/direction at smaller difference step. Original instrument miss preserved; '
              'no optimizer convergence or independent circuit evidence.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
