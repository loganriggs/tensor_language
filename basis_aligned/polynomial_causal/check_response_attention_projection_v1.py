"""Actual-weight controls: projection/full attention <=1e-10 relative,
zero edit exact. Descriptive two-thread timing across21 strengths, including
projection-bank preparation. Native FP32/full-circuit validation not claimed.
"""
import json,time,statistics
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis,coefficients
import response_attention_projection_v1 as shared
import raw_attention_response_v1 as original
P=Path(__file__).resolve().parent

def timing(fn):
    fn();times=[]
    for _ in range(5):
        start=time.perf_counter();fn();times.append(time.perf_counter()-start)
    return statistics.median(times)

@torch.no_grad()
def main(timing_sizes=(21,), output_name='RESPONSE_ATTENTION_PROJECTION_V1_CONTROL.json'):
    torch.set_num_threads(2);torch.manual_seed(61342);start=time.perf_counter()
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    z=torch.randn(1,17,1152,dtype=torch.float64);l,r,d=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    base=((z@l.T)*(z@r.T))@d.T/(z.square().mean(-1,keepdim=True)+original.EPS)
    ctx,basis=prepare_basis(z,base,program,float(sd['transformer.h.10.lambdas'][0]))
    raw=torch.randn_like(z);matrices=[sd['transformer.h.10.attn.'+k+'.weight'].double() for k in ['c_q','c_k','c_q2','c_k2','c_v']]
    output=sd['transformer.h.10.attn.c_proj.weight'].double();mix=float(sd['transformer.h.10.attn.lamb']);first=torch.randn_like(z)
    cache=shared.prepare(raw,basis,matrices);p0,rho0=original.prepare(raw,matrices)
    field=torch.randn(1,17,1,dtype=torch.float64);inputs=[coefficients(field*float(a),ctx) for a in torch.linspace(-1,1,21)]
    maxproj=0.;maxoutput=0.;maxrho=0.;zero=0.
    for i,u in enumerate(inputs):
        delta=torch.einsum('...k,...kd->...d',u,basis);p,rho=original.changed(raw,delta,p0,matrices);q,sigma=shared.changed(u,cache)
        maxproj=max(maxproj,max(float((a-b).norm()/a.norm()) for a,b in zip(p,q)));maxrho=max(maxrho,float((rho-sigma).abs().max()))
        a=original.execute(p,rho,first,mix,output);b=original.execute(q,sigma,first,mix,output)
        maxoutput=max(maxoutput,float((a-b).norm()/a.norm()))
    u=coefficients(torch.zeros_like(field),ctx);q,sigma=shared.changed(u,cache);zero=max(float((a-b).abs().max()) for a,b in zip(q,cache['baseline']))
    assert maxproj<1e-10 and maxoutput<1e-10 and maxrho==0 and zero==0
    deltas=[torch.einsum('...k,...kd->...d',u,basis) for u in inputs]
    def sweep_old(items):
        baseline,_=original.prepare(raw,matrices)
        return [original.changed(raw,delta,baseline,matrices) for delta in items]
    def sweep_shared(items):
        c=shared.prepare(raw,basis,matrices)
        return [shared.changed(u,c) for u in items]
    timing_cells=[]
    for count in timing_sizes:
        oldtime=timing(lambda:sweep_old(deltas[:count]));newtime=timing(lambda:sweep_shared(inputs[:count]))
        timing_cells.append(dict(count=count,old_seconds=oldtime,shared_seconds=newtime,speed_ratio=oldtime/newtime))
    out=dict(timing_cells=timing_cells,projection_error=maxproj,full_attention_error=maxoutput,rho_maxabs=maxrho,zero_projection_change=zero,
        strengths=21,positions=17,heads=9,old_seconds=oldtime,shared_seconds=newtime,speed_ratio=oldtime/newtime,
        old_projection_matrix_applications=5*(1+21),shared_projection_vector_applications=5*(1+3),
        added_projected_bank_scalars_per_position=5*3*1152,seconds=time.perf_counter()-start,
        scope='Same synthetic fixed context, actual weights, position-dependent signed edits. Both QK factors and V vary; raw state norms recomputed. Timing only projection updates including bank preparation, not complete attention/model. Existing deltas supplied to baseline; shared method forms them for norms.')
    (P/output_name).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
