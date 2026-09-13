"""Actual-weight synthetic-context algebra/cost check; frozen bars on board."""
import json,time,statistics
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis
from fixed_writer_products_v1 import prepare_global
from polynomial_tail_mlp10_v1 import prepare as old_prepare,evaluate as old_evaluate
from shared_polynomial_tail_v1 import prepare,evaluate,omitted

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(7131212)
    p=Path(__file__).resolve().parent
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(p/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    def maps(layer):return [sd[f'transformer.h.{layer}.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    l9,r9,d9=maps(9);l,r,d=maps(10);bias=sd['transformer.h.10.mlp.Down_bias'].double()
    eps=torch.finfo(torch.float32).eps;rows=[];g=None
    for batch in (1,16,64):
        z9=torch.randn(batch,1152,dtype=torch.float64)
        base=((z9@l9.T)*(z9@r9.T))@d9.T/(z9.square().mean(-1,keepdim=True)+eps)
        response,basis=prepare_basis(z9,base,program,float(sd['transformer.h.10.lambdas'][0]))
        context=dict(response=response,basis=basis)
        if g is None:
            start=time.perf_counter();g=prepare_global(basis[0,0],l,r,d)
            setup=time.perf_counter()-start
        old=old_prepare(context,l,r,d,bias);new=prepare(context,l,r,d,bias,g)
        full_errors=[];tail_errors=[]
        for strength in (-2.,-.5,0.,.5,1.,2.):
            a=torch.full((batch,1),strength,dtype=torch.float64)
            z=torch.randn(batch,1152,dtype=torch.float64)
            ref=old_evaluate(z,a,old);fit=evaluate(z,a,new)
            full_errors.append(float((ref-fit).norm()/ref.norm()))
            t=a/old['scale'];rho=response['perpendicular_rms']+response['writer_rms']*(a-response['parallel']).square()
            tail=a.square()/(2*rho.square())*t.pow(3)*(old['tail'][...,0,:]+t*old['tail'][...,1,:])
            tail_errors.append(float((tail-omitted(a,new)).norm()/tail.norm().clamp_min(1e-30)))
        funcs={'old':lambda:old_prepare(context,l,r,d,bias),'shared':lambda:prepare(context,l,r,d,bias,g)}
        samples={k:[] for k in funcs}
        for fn in funcs.values():fn();fn()
        for trial in range(9):
            for name in (list(funcs) if trial%2 else list(funcs)[::-1]):
                start=time.perf_counter();funcs[name]();samples[name].append(time.perf_counter()-start)
        med={k:statistics.median(v) for k,v in samples.items()}
        rows.append(dict(batch=batch,max_full_error=max(full_errors),max_tail_error=max(tail_errors),median_seconds=med,speedup=med['old']/med['shared']))
    bad=dict(g,writer=g['writer']+1)
    try:prepare(context,l,r,d,bias,bad);rejected=False
    except ValueError:rejected=True
    result=dict(pred_a=all(x['max_full_error']<1e-10 and x['max_tail_error']<1e-10 for x in rows),
        pred_b=all(x['speedup']>=1.1 for x in rows),pred_c=rejected,rows=rows,
        global_setup_seconds=setup,global_cache_scalars=sum(v.numel() for v in g.values()),
        private_scalars_per_context=1152,old_private_scalars_per_context=2305,
        scope='FP64 CPU, synthetic pristine contexts, actual weights, same writer across contexts. Global setup excluded from reused preparation timing and reported separately. Native dense maps and response state remain. No fresh/native/OOD behavior claim.')
    (p/'SHARED_POLYNOMIAL_TAIL_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
