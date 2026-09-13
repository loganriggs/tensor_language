"""Registered full conditional-operator control and1/4/12pair CPU price.
A new/old generators and full products<=1e-10 relative; zero edit product zero.
B >=10% lower total time at12pairs; include all preparation and products.
No requirement at1pair; every workload reported. No whole-model/OOD claim.
"""
import json,time,statistics
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from composed_mlp10_inputs_v1 import execute as old_execute
import composed_mlp10_context_v1 as shared
import directional_mlp_response_context_v1 as response
import raw_attention_response_v1 as raw_attn
P=Path(__file__).resolve().parent

def timed(fn):
    fn();samples=[]
    for _ in range(5):
        start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
    return statistics.median(samples)

@torch.no_grad()
def main(output_name='COMPOSED_MLP10_CONTEXT_V1_CONTROL.json'):
    torch.set_num_threads(2);torch.manual_seed(61355);start=time.perf_counter()
    sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True);program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    z=torch.randn(1,17,1152,dtype=torch.float64);raw=torch.randn_like(z);z10=torch.randn_like(z);first=torch.randn_like(z)
    l9,r9,d9=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    base=((z@l9.T)*(z@r9.T))@d9.T/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
    mats=[sd['transformer.h.10.attn.'+k+'.weight'].double() for k in ['c_q','c_k','c_q2','c_k2','c_v']]
    out=sd['transformer.h.10.attn.c_proj.weight'].double();mix=float(sd['transformer.h.10.attn.lamb']);scale=float(sd['transformer.h.10.lambdas'][0])
    l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    arguments=(z,base,raw,z10,first,program,scale,mats,mix,out)
    context=shared.prepare(*arguments)
    pairs=[(torch.randn(1,17,1,dtype=z.dtype)*a,torch.randn(1,17,1,dtype=z.dtype)*b) for a,b in zip(torch.linspace(-.5,.5,12),torch.linspace(.3,-.3,12))]
    maxgen=0.;maxprod=0.
    for a,b in pairs:
        old=old_execute(z,base,raw,z10,first,a,b,program,scale,mats,mix,out);new=shared.evaluate(a,b,context)
        maxgen=max(maxgen,max(float((old[k]-new[k]).norm()/old[k].norm().clamp_min(1e-30)) for k in old))
        x=shared.product(old,l,r,d);y=shared.product(new,l,r,d);maxprod=max(maxprod,float((x-y).norm()/x.norm()))
    zero=float(shared.product(shared.evaluate(torch.zeros_like(pairs[0][0]),pairs[0][1],context),l,r,d).abs().max())
    assert maxgen<=1e-10 and maxprod<=1e-10 and zero==0
    cells=[]
    for count in [1,4,12]:
        def baseline():
            for a,b in pairs[:count]:
                parts=old_execute(z,base,raw,z10,first,a,b,program,scale,mats,mix,out);shared.product(parts,l,r,d)
        def cached():
            c=shared.prepare(*arguments)
            for a,b in pairs[:count]:shared.product(shared.evaluate(a,b,c),l,r,d)
        def simple_cache():
            rc=response.prepare(z,base,program);p0,rho0=raw_attn.prepare(raw,mats)
            baseline_attn=raw_attn.execute(p0,rho0,first,mix,out)
            outputs=[]
            for a,b in pairs[:count]:
                changes=[]
                for amplitude in [a,b]:
                    delta=scale*response.evaluate(amplitude,rc)
                    ports,norm=raw_attn.changed(raw,delta,p0,mats)
                    partner=raw_attn.execute(ports,norm,first,mix,out)-baseline_attn
                    changes.append(delta+partner)
                cc,rr=changes;norm=(z10+cc+rr).square().mean(-1,keepdim=True)+raw_attn.EPS
                outputs.append(shared.product(dict(child=cc,remainder=rr,joint_rho=norm),l,r,d))
            return outputs
        simple_outputs=simple_cache()
        for (aa,bb),observed in zip(pairs[:count],simple_outputs):
            expected=shared.product(shared.evaluate(aa,bb,context),l,r,d)
            assert float((observed-expected).norm()/expected.norm())<1e-10
        simpletime=timed(simple_cache)
        oldtime=timed(baseline);newtime=timed(cached);cells.append(dict(pairs=count,original_seconds=oldtime,shared_seconds=newtime,speed_ratio=oldtime/newtime,simple_cache_seconds=simpletime,versus_simple_cache_speed_ratio=simpletime/newtime))
    # Unique tensor storage, excluding supplied common input arrays, program weights and output map.
    supplied=[z,base,raw,z10,first,out,*program.values()]
    excluded={x.untyped_storage().data_ptr() for x in supplied if isinstance(x,torch.Tensor)};seen={};
    def visit(x):
        if isinstance(x,torch.Tensor):
            key=x.untyped_storage().data_ptr()
            if key not in excluded:seen[key]=x.untyped_storage().nbytes()
        elif isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
    visit(context)
    result=dict(pred_a=True,pred_b=cells[-1]['shared_seconds']<=.9*cells[-1]['original_seconds'],generator_error=maxgen,product_error=maxprod,zero_product_maxabs=zero,cells=cells,extra_prepared_tensor_bytes=sum(seen.values()),seconds=time.perf_counter()-start,
        scope='Full conditional input generator plus one complete bilinear cross-product, synthetic17position context and actualweights. Includes global/context preparation; excludes common pristine states generation and native suffix. No six-product bank allocated when only sumRR+mixed+AA is consumed.')
    (P/output_name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
