"""Tail rewrite countercheck: same inputs, native, old executor, new executor.
Pred_a: replay <1e-10; pred_tail_speed: >=1.1x old warm at every batch.
Inherited pred_b/c still refer to the old three-bank vs native.
Tail scalars exclude shared response context and native weights, as old count does.
"""
import json,time,statistics,sys
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis,coefficients
from diagonal_mlp10_branch_v1 import prepare,evaluate
from polynomial_mlp10_branch_v1 import prepare as prepare_three,evaluate as evaluate_three
from polynomial_tail_mlp10_v1 import prepare as prepare_tail,evaluate as evaluate_tail
P=Path(__file__).resolve().parent
FP32='--fp32' in sys.argv
DTYPE=torch.float32 if FP32 else torch.float64
TOL=2e-5 if FP32 else 1e-10

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(61591)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    z9=torch.randn(1,1152,dtype=DTYPE);eps=torch.finfo(torch.float32).eps
    l9,r9,d9=[sd['transformer.h.9.mlp.'+k+'.weight'].to(DTYPE) for k in ('Left','Right','Down')]
    base=((z9@l9.T)*(z9@r9.T))@d9.T/(z9.square().mean(-1,keepdim=True)+eps)
    response,basis=prepare_basis(z9,base,program,float(sd['transformer.h.10.lambdas'][0]));response={k:v.to(DTYPE) if isinstance(v,torch.Tensor) else v for k,v in response.items()};basis=basis.to(DTYPE);context=dict(response=response,basis=basis)
    l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].to(DTYPE) for k in ('Left','Right','Down')];bias=sd['transformer.h.10.mlp.Down_bias'].to(DTYPE)
    prepared=prepare(context,l,r,d,bias);three=prepare_three(context,l,r,d,bias);tail=prepare_tail(context,l,r,d,bias);rows=[]
    for batch in (1,3,16,64):
        a=torch.linspace(-1,1,batch,dtype=z9.dtype)[:,None];u=coefficients(a,response)
        z=torch.randn(batch,1152,dtype=z9.dtype)+torch.einsum('bk,bkd->bd',u,basis.expand(batch,-1,-1))
        def direct():return z+((z@l.T)*(z@r.T))@d.T/(z.square().mean(-1,keepdim=True)+eps)+bias
        def cold():return evaluate(z,a,prepare(context,l,r,d,bias))
        def warm():return evaluate(z,a,prepared)
        ref=direct();error=float((warm()-ref).norm()/ref.norm());assert error<TOL
        functions={'tail_reused':lambda:evaluate_tail(z,a,tail),'tail_preparation_included':lambda:evaluate_tail(z,a,prepare_tail(context,l,r,d,bias)),'direct':direct,'bank_preparation_included':cold,'bank_reused':warm,'three_preparation_included':lambda:evaluate_three(z,a,prepare_three(context,l,r,d,bias)),'three_reused':lambda:evaluate_three(z,a,three)};samples={k:[] for k in functions}
        for fn in functions.values():fn();fn()
        for trial in range(7):
            order=list(functions)
            if trial%2:order.reverse()
            for name in order:
                start=time.perf_counter();functions[name]();samples[name].append(time.perf_counter()-start)
        med={k:statistics.median(v) for k,v in samples.items()}
        rows.append(dict(tail_replay_error=float((evaluate_tail(z,a,tail)-evaluate_three(z,a,three)).norm()/evaluate_three(z,a,three).norm()),tail_speedup=med['three_reused']/med['tail_reused'],batch=batch,relative_error=error,three_relative_error=float((evaluate_three(z,a,three)-ref).norm()/ref.norm()),median_seconds=med,
            speedup_with_preparation=med['direct']/med['three_preparation_included'],speedup_reused=med['direct']/med['three_reused']))
    result=dict(dtype=str(DTYPE),replay_tolerance=TOL,pred_a=all(x['tail_replay_error']<TOL for x in rows),pred_tail_speed=all(x['tail_speedup']>=1.1 for x in rows),tail_bank_scalars=2*1152+1,pred_b=all(x['speedup_with_preparation']>=1.1 for x in rows),pred_c=all(x['speedup_reused']>=1.1 for x in rows),rows=rows,
        additional_prepared_scalars=6*4608+3*1152+1,additional_prepared_fp64_bytes=8*(6*4608+3*1152+1),
        direct_main_mac_per_input=3*1152*4608,bank_main_mac_per_input=3*1152*4608,
        scope='Local fullMLPbranch declared-dtype two-thread CPU, same batched z inputs. Cold includes each bank/readers preparation, excludes shared upstreamresponsecontext and attentioninputgeneration onbothsides. Warm assumes bank prepared after prior use; three-term approximation errors are reported separately. Both retain same dense L/R/Downmaps; bankaddsworkandstate. No GPU/wholemodeltiming.')
    (P/('POLYNOMIAL_TAIL_MLP10_V1_FP32_RESULT.json' if FP32 else 'POLYNOMIAL_TAIL_MLP10_V1_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
