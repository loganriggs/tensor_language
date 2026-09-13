"""Precision, serialization and warmed CPU timing for the frozen sparse operator."""
import json,time,statistics
from pathlib import Path
import torch
from sparse_interaction_executor_v1 import compile_program,Executor
from head17_output_block_objective_v1 import build
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent

def relative(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
def timed(fn,z,a):
    for _ in range(2):fn(z,a)
    samples=[]
    for _ in range(7):
        start=time.perf_counter();fn(z,a);samples.append(time.perf_counter()-start)
    return dict(median_seconds=statistics.median(samples),samples_seconds=samples)

@torch.no_grad()
def main():
    torch.set_num_threads(2);start=time.perf_counter();t,ids=build()
    program,fit=compile_program(t)
    path=P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt';assert not path.exists();torch.save(program,path)
    decode=time.perf_counter();execute=Executor(torch.load(path,weights_only=True));decode=time.perf_counter()-decode
    sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    l,r,d=[sd['transformer.h.17.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    c=sd['lm_head.weight'][ids].double()@d
    w=torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt',weights_only=True)['output_matrix'].double()
    lw,rw=l@w,r@w
    data=torch.load(P/'COMPOSED_LAST_BLOCK_STATES_V1_ARTIFACT.pt',weights_only=True)
    z=data['linear_parts'][:,3].double();v=data['linear_parts'][:,2].double()-z
    a=torch.linalg.lstsq(w,v.T).solution.T
    gen=torch.Generator().manual_seed(61323)
    panels={'regional':(z,a),'independent':(torch.randn(120,1152,generator=gen,dtype=torch.float64),torch.randn(120,128,generator=gen,dtype=torch.float64))}
    checks=[]
    for name,(zz,aa) in panels.items():
        native=((zz@l.T)*(aa@rw.T)+(zz@r.T)*(aa@lw.T))@c.T
        exact=torch.einsum('oia,ni,na->no',t,zz,aa)
        approx=torch.einsum('oia,ni,na->no',fit,zz,aa)
        pred=execute(zz.float(),aa.float()).double()
        checks.append(dict(panel=name,exact_formula_error=relative(exact,native),sparse_precision_error=relative(pred,approx),compressed_function_error=relative(pred,exact)))
    tf=t.float();ff=fit.float();lf,rf,cf,lwf,rwf=[x.float() for x in [l,r,c,lw,rw]]
    dense=lambda zz,aa:torch.einsum('oia,ni,na->no',tf,zz,aa)
    dense_pruned=lambda zz,aa:torch.einsum('oia,ni,na->no',ff,zz,aa)
    native=lambda zz,aa:((zz@lf.T)*(aa@rwf.T)+(zz@rf.T)*(aa@lwf.T))@cf.T
    timings=[]
    for batch in [1,120]:
        zz=z[:batch].float();aa=a[:batch].float()
        cells={name:timed(fn,zz,aa) for name,fn in [('sparse_csr',execute),('dense',dense),('dense_pruned',dense_pruned),('native_factored',native)]}
        timings.append(dict(batch=batch,methods=cells))
    dense_bytes=4*t.numel();native_bytes=sum(x.numel()*4 for x in [l,r,c,lw,rw])
    out=dict(checks=checks,timings=timings,decode_seconds=decode,serialized_bytes=path.stat().st_size,
        dense_bytes=dense_bytes,native_factored_bytes=native_bytes,csr_resident_bytes=execute.resident_bytes(),
        packed_tensor_payload_bytes=sum(v.numel()*v.element_size() for v in program.values() if isinstance(v,torch.Tensor)),
        coefficient_error=relative(fit,t),retained_entries=program['values'].numel(),seconds=time.perf_counter()-start,
        pred_a=all(c['exact_formula_error']<=1e-5 and c['sparse_precision_error']<=1e-5 for c in checks),
        pred_b=path.stat().st_size/dense_bytes<.75,
        pred_c=all(x['methods']['sparse_csr']['median_seconds']<min(x['methods'][name]['median_seconds'] for name in ['dense','native_factored']) for x in timings),
        scope='Two-thread CPU raw-port mixed operator only; immutable packed artifact. CSR working indexes counted; input generators, normalization and other routes not included.')
    (P/'SPARSE_INTERACTION_EXECUTOR_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
