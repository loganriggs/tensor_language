"""Registered full conditional three-branch price comparison.
A all branch output replay relative errors <=1e-10.
B shared >=10%faster than direct at12pairs, including preparation.
C shared >=10%faster than response-only folding at12pairs.
Report1/4pair outcomes regardless; no whole-model speed claim.
Seven rotated-order timing repeats, identical retained outputs; CPU2threads.
"""
import json
import time
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from composed_mlp10_context_v1 import prepare
import composed_joint_response_v1 as joint
import raw_attention_response_v1 as attn

P = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(61373)
    start = time.perf_counter()
    sd = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    program = torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt', weights_only=True)
    z9 = torch.randn(1, 17, 1152, dtype=torch.float64)
    raw10 = torch.randn_like(z9)
    first = torch.randn_like(z9)
    def mlp_weights(index):
        prefix = f'transformer.h.{index}.mlp.'
        return [sd[prefix+k+'.weight'].double() for k in ['Left','Right','Down']]
    l9,r9,d9 = mlp_weights(9)
    l10,r10,d10 = mlp_weights(10)
    bias10 = sd['transformer.h.10.mlp.Down_bias'].double()
    def h9(z):
        return z+((z@l9.T)*(z@r9.T))@d9.T/(z.square().mean(-1,keepdim=True)+attn.EPS)
    h0 = h9(z9)
    baseline_mlp9 = h0-z9
    matrices = [sd['transformer.h.10.attn.'+k+'.weight'].double()
                for k in ['c_q','c_k','c_q2','c_k2','c_v']]
    output = sd['transformer.h.10.attn.c_proj.weight'].double()
    mixture = float(sd['transformer.h.10.attn.lamb'])
    scale = float(sd['transformer.h.10.lambdas'][0])
    def attend(raw):
        ports,rho = attn.prepare(raw,matrices)
        return attn.execute(ports,rho,first,mixture,output)
    z10 = raw10+attend(raw10)
    context = prepare(z9,baseline_mlp9,raw10,z10,first,program,scale,matrices,mixture,output)
    writer = context['response']['direction']
    def direct(amplitude):
        raw = raw10+scale*(h9(z9-amplitude*writer)-h0)
        z = raw+attend(raw)
        return joint.block_output(z,l10,r10,d10,bias10)
    a0 = torch.randn(1,17,1,dtype=z9.dtype)*.3
    b0 = torch.randn_like(a0)*.3
    import statistics
    import directional_mlp_response_context_v1 as response
    pairs=[(a0*(i+1)/12,b0*(13-i)/12) for i in range(12)]
    def full_direct(count):
        return [direct(amplitude) for a,b in pairs[:count] for amplitude in (a,b,a+b)]
    def response_only(count):
        rc=response.prepare(z9,baseline_mlp9,program)
        outputs=[]
        for a,b in pairs[:count]:
            for amplitude in (a,b,a+b):
                raw=raw10+scale*response.evaluate(amplitude,rc)
                z=raw+attend(raw)
                outputs.append(joint.block_output(z,l10,r10,d10,bias10))
        return outputs
    def full_shared(count):
        ctx=prepare(z9,baseline_mlp9,raw10,z10,first,program,scale,matrices,mixture,output)
        return [joint.block_output(joint.branch(amplitude,ctx),l10,r10,d10,bias10)
                for a,b in pairs[:count] for amplitude in (a,b,a+b)]
    variants={'direct':full_direct,'response_only':response_only,'shared':full_shared}
    cells=[]
    for count in (1,4,12):
        reference=full_direct(count)
        errors={}
        for name,fn in variants.items():
            candidate=fn(count)
            errors[name]=max(float((x-y).norm()/y.norm()) for x,y in zip(candidate,reference))
            assert errors[name]<=1e-10
        # Rotate measurement order to reduce thermal/order confounding.
        times={name:[] for name in variants}
        names=list(variants)
        for trial in range(7):
            for name in names[trial%3:]+names[:trial%3]:
                tic=time.perf_counter();outputs=variants[name](count)
                times[name].append(time.perf_counter()-tic)
                del outputs
        med={name:statistics.median(v) for name,v in times.items()}
        cells.append(dict(pairs=count,branch_replay_errors=errors,median_seconds=med,
                          samples_seconds=times,shared_speedup_over_direct=med['direct']/med['shared'],
                          shared_speedup_over_response_only=med['response_only']/med['shared']))
    common=sum(x.numel() for x in [*matrices,output,l10,r10,d10,bias10])
    direct_specific=sum(x.numel() for x in [l9,r9,d9,writer])
    folded_specific=sum(x.numel() for x in program.values() if isinstance(x,torch.Tensor))
    result=dict(pred_a=all(max(c['branch_replay_errors'].values())<=1e-10 for c in cells),
                pred_b=cells[-1]['shared_speedup_over_direct']>=1/0.9,
                pred_c=cells[-1]['shared_speedup_over_response_only']>=1/0.9,
                cells=cells,local_weight_scalars=dict(common=common,direct_specific=direct_specific,
                    folded_specific=folded_specific,direct_total=common+direct_specific,folded_total=common+folded_specific),
                seconds=time.perf_counter()-start,
                scope='Actual weights, synthetic B1,T17,d1152,FP64,CPU2threads. Three changed branches per pair. Identical output list retention. Shared includes all global/context preparation each measured call. Pristine z9/MLP9/raw10/z10/first-values common supplied; no prefix or suffix cost. Local weight inventory counts objects needed by conditional operator only; native weights remain necessary for pristine background and other edits. Not whole-model compression/speed.')
    (P/'COMPOSED_JOINT_COST_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
    for c in cells:print(c['pairs'],c['median_seconds'],c['shared_speedup_over_direct'],c['shared_speedup_over_response_only'])


if __name__=='__main__':
    main()
