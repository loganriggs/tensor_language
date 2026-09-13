"""Matched direct/bank full branch, same FP64 inputs and batch sizes."""
import json,time,statistics
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis,coefficients
from diagonal_mlp10_branch_v1 import prepare,evaluate
P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(61591)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    z9=torch.randn(1,1152,dtype=torch.float64);eps=torch.finfo(torch.float32).eps
    l9,r9,d9=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    base=((z9@l9.T)*(z9@r9.T))@d9.T/(z9.square().mean(-1,keepdim=True)+eps)
    response,basis=prepare_basis(z9,base,program,float(sd['transformer.h.10.lambdas'][0]));context=dict(response=response,basis=basis)
    l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')];bias=sd['transformer.h.10.mlp.Down_bias'].double()
    prepared=prepare(context,l,r,d,bias);rows=[]
    for batch in (1,3,16,64):
        a=torch.linspace(-1,1,batch,dtype=z9.dtype)[:,None];u=coefficients(a,response)
        z=torch.randn(batch,1152,dtype=z9.dtype)+torch.einsum('bk,bkd->bd',u,basis.expand(batch,-1,-1))
        def direct():return z+((z@l.T)*(z@r.T))@d.T/(z.square().mean(-1,keepdim=True)+eps)+bias
        def cold():return evaluate(z,a,prepare(context,l,r,d,bias))
        def warm():return evaluate(z,a,prepared)
        ref=direct();error=float((warm()-ref).norm()/ref.norm());assert error<1e-10
        functions={'direct':direct,'bank_preparation_included':cold,'bank_reused':warm};samples={k:[] for k in functions}
        for fn in functions.values():fn();fn()
        for trial in range(7):
            order=list(functions)
            if trial%2:order.reverse()
            for name in order:
                start=time.perf_counter();functions[name]();samples[name].append(time.perf_counter()-start)
        med={k:statistics.median(v) for k,v in samples.items()}
        rows.append(dict(batch=batch,relative_error=error,median_seconds=med,
            speedup_with_preparation=med['direct']/med['bank_preparation_included'],speedup_reused=med['direct']/med['bank_reused']))
    result=dict(pred_a=True,pred_b=all(x['speedup_with_preparation']>=1.1 for x in rows),pred_c=all(x['speedup_reused']>=1.1 for x in rows),rows=rows,
        additional_prepared_scalars=6*4608+5*1152,additional_prepared_fp64_bytes=8*(6*4608+5*1152),
        direct_main_mac_per_input=3*1152*4608,bank_main_mac_per_input=3*1152*4608,
        scope='Local fullMLPbranch FP64 two-thread CPU, same batched z inputs. Cold includes fivebank/readers preparation, excludes shared upstreamresponsecontext and attentioninputgeneration onbothsides. Warm assumesbankfreeafterprioruse. Both retain same dense L/R/Downmaps; bankaddsworkandstate. No GPU/wholemodeltiming.')
    (P/'DIAGONAL_MLP10_COST_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
