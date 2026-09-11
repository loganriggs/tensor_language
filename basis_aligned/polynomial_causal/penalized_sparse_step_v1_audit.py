"""CPU native kernel/one-step check; no live GPU fit mutation.

A initial/CP/penalty replay<=1e-8, FD<=1e-6; B Armijo gain>=1e-5;
C initial gradient<=60CPU seconds. No native convergence claim.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK
from penalized_projected_sparse_v1 import value_gradient
from folded_sparse_dictionary_v1 import decode
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'PENALIZED_SPARSE_STEP_V1_AUDIT.json';assert not out.exists()
    reference=json.loads((P/'OUTPUT_COMPONENT_BUDGET_V1_AUDIT.json').read_text());source=reference['source']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu')
    basis=saved['analysis_basis'].double();ids=saved['code_indices'].long()
    values=saved['code_values'].double();values/=values.norm(dim=1,keepdim=True)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    penalty=reference['penalty'];started=time.perf_counter()
    initial,gradient,writer,details=value_gradient((l,r,d),wh,basis,ids,values,total,penalty)
    gradient_seconds=time.perf_counter()-started
    initial_replay=max(abs(1-details['residual']-reference['capture']),abs(details['component_energy']-reference['component_energy']))
    directions=[-g/g.norm()*len(x)**.5 for g,x in zip(gradient,(basis,values))]
    slope=sum(float((g*v).sum()) for g,v in zip(gradient,directions));h=1e-5
    print(json.dumps(dict(initial=float(initial),gradient_seconds=gradient_seconds,initial_replay=initial_replay)),flush=True)
    plus=value_gradient((l,r,d),wh,basis+h*directions[0],ids,values+h*directions[1],total,penalty)[0]
    minus=value_gradient((l,r,d),wh,basis-h*directions[0],ids,values-h*directions[1],total,penalty)[0]
    fd=abs(float((plus-minus)/(2*h))-slope)/max(1.,abs(slope))
    step=.001;accepted=False;trials=[]
    for _ in range(10):
        nb=basis+step*directions[0];nv=values+step*directions[1]
        loss,g,writer,final=value_gradient((l,r,d),wh,nb,ids,nv,total,penalty)
        trials.append(dict(step=step,loss=float(loss)))
        print(json.dumps(trials[-1]),flush=True)
        if float(loss)<=float(initial)+1e-4*step*slope:accepted=True;break
        step*=.5
    assert accepted,'No Armijo step; no silent same-point success'
    nv=nv/nv.norm(dim=1,keepdim=True);readers,nb,_,_=decode(nb,ids,nv,torch.ones(len(ids)))
    a,b=readers.chunk(2);proposal=(a,b,wh@writer);native=(l,r,wh@d)
    capture=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
    replay=max(abs(capture-(1-final['residual'])),abs(float(loss)-(1-capture+penalty*final['component_energy'])))
    cache=Path('/dev/shm/bilin18_penalized_sparse_step_v1.pt');assert not cache.exists()
    torch.save(dict(analysis_basis=nb,code_indices=ids.to(torch.int16),code_values=nv,down=writer,
        retained_bias_key=saved['retained_bias_key'],source=source,penalty=penalty),cache)
    result=dict(predictions=dict(pred_a_instrument=max(initial_replay,replay)<=1e-8 and fd<=1e-6,
        pred_b_descent=float(initial-loss)>=1e-5,pred_c_cost=gradient_seconds<=60),
        initial_objective=float(initial),final_objective=float(loss),objective_gain=float(initial-loss),
        initial_details=details,final_details=final,final_capture=capture,step=step,trials=trials,
        initial_replay=initial_replay,fd_error=fd,cp_replay=replay,gradient_seconds=gradient_seconds,
        seconds=time.perf_counter()-started,source=source,
        cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
        scope='One CPU Armijo step in penalized fixed-support unit-code coordinates; no joint convergence '
              'or stable/native behavioral circuit evidence, and no live run changed.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
