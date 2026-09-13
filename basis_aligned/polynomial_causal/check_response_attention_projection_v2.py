"""Registered controls: v2/v1 projections and full attention <=1e-10;
actual-weight signed perposition edits. Report2/21query,1/4context projection
sweep timing including global and context cache preparation. Target >=10%
time saving vs v1 in both two-query sizes; not a claim vs direct projection.
"""
import json,time,statistics
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis,coefficients
import response_attention_projection_v1 as v1
import response_attention_projection_v2 as v2
from raw_attention_response_v1 import execute
P=Path(__file__).resolve().parent

def timed(fn):
    fn();times=[]
    for _ in range(5):
        start=time.perf_counter();fn();times.append(time.perf_counter()-start)
    return statistics.median(times)

@torch.no_grad()
def main(output_name='RESPONSE_ATTENTION_PROJECTION_V2_CONTROL.json'):
    torch.set_num_threads(2);torch.manual_seed(61346);start=time.perf_counter()
    sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True);program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    l,r,d=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    mats=[sd['transformer.h.10.attn.'+k+'.weight'].double() for k in ['c_q','c_k','c_q2','c_k2','c_v']]
    out=sd['transformer.h.10.attn.c_proj.weight'].double();mix=float(sd['transformer.h.10.attn.lamb']);scale=float(sd['transformer.h.10.lambdas'][0])
    cells=[];maxproj=0.;maxnorm=0.;maxatt=0.
    for batch in [1,4]:
        z=torch.randn(batch,17,1152,dtype=torch.float64);raw=torch.randn_like(z);first=torch.randn_like(z)
        base=((z@l.T)*(z@r.T))@d.T/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
        response,basis=prepare_basis(z,base,program,scale);global_cache=v2.prepare_global(program,mats,scale)
        c1=v1.prepare(raw,basis,mats);c2=v2.prepare(raw,response,global_cache,mats)
        field=torch.randn(batch,17,1,dtype=torch.float64)
        amplitudes=[float(s)*field for s in torch.linspace(-1,1,21)];coef=[coefficients(a,response) for a in amplitudes]
        for a,u in zip(amplitudes,coef):
            p,rho=v1.changed(u,c1);q,sigma=v2.changed(a,c2)
            maxproj=max(maxproj,max(float((x-y).norm()/x.norm()) for x,y in zip(p,q)))
            maxnorm=max(maxnorm,float((rho-sigma).norm()/rho.norm()))
            x=execute(p,rho,first,mix,out);y=execute(q,sigma,first,mix,out);maxatt=max(maxatt,float((x-y).norm()/x.norm()))
        for count in [2,21]:
            def run1():
                context=v1.prepare(raw,basis,mats)
                return [v1.changed(u,context) for u in coef[:count]]
            def run2():
                g=v2.prepare_global(program,mats,scale);context=v2.prepare(raw,response,g,mats)
                return [v2.changed(a,context) for a in amplitudes[:count]]
            def run2warm():
                context=v2.prepare(raw,response,global_cache,mats)
                return [v2.changed(a,context) for a in amplitudes[:count]]
            a=timed(run1);b=timed(run2);warm=timed(run2warm);setup=timed(lambda:v2.prepare_global(program,mats,scale))
            cells.append(dict(contexts=batch,queries=count,v1_seconds=a,v2_seconds=b,speed_ratio=a/b,v2_warm_seconds=warm,warm_speed_ratio=a/warm,global_setup_seconds=setup))
    assert max(maxproj,maxnorm,maxatt)<1e-10
    result=dict(pred_a=True,pred_b=all(c['v2_seconds']<=.9*c['v1_seconds'] for c in cells if c['queries']==2),projection_error=maxproj,norm_error=maxnorm,attention_error=maxatt,cells=cells,
        v1_extra_projection_scalars_per_position=17280,v2_extra_projection_scalars_per_position=11520,v2_global_projection_scalars=11520,
        seconds=time.perf_counter()-start,scope='Actual weights, synthetic contexts; exact two-level projection reuse. Timings include global projection preparation but exclude common response-context construction, attention routing and suffix; response fields and coefficients supplied to v1.')
    (P/output_name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
