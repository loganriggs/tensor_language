"""Actual sparse shared-source execution against a hoisted dense baseline.
Pred_a FP64 relative replay<=1e-10 and FP32<=1e-5.
Pred_b >=1.1x speed at all1/19/128/512 rows in both precisions.
Pred_c prepared resident storage smaller than dense in both precisions.
CPU two threads; packed payload and execution memory are different prices.
"""
import json,time,statistics,warnings
from pathlib import Path
import torch
import torch.nn.functional as F
from sparse_parent_reader_v1 import unpack


def prepare(s,q,keys,dtype):
    # S^T CSR uses64 rows; both QK readers consume one shared source projection.
    st=s.T.to(dtype).contiguous().to_sparse_csr()
    st=torch.sparse_csr_tensor(st.crow_indices().int(),st.col_indices().int(),st.values(),size=st.shape)
    inverse=torch.linalg.inv(s.T@s)
    sk=(keys@s@inverse).T.to(dtype).contiguous()
    dq=q.to(dtype).contiguous();dk=(keys@q).T.to(dtype).contiguous()
    return (st,sk),(dq,dk)


def execute_sparse(x,program):
    st,sk=program
    return torch.sparse.mm(st,x.T).T@sk


def execute_dense(x,program):
    q,k=program
    return (x@q)@k


def byte_count(program):
    total=0
    for x in program:
        parts=(x.values(),x.crow_indices(),x.col_indices()) if x.layout==torch.sparse_csr else (x,)
        total+=sum(t.numel()*t.element_size() for t in parts)
    return total


def main():
    torch.set_num_threads(2);warnings.filterwarnings('ignore',message='Sparse CSR tensor support')
    p=Path(__file__).resolve().parent;gen=torch.Generator().manual_seed(7131401)
    item=torch.load(p/'SPARSE_COMPLETE_EVEN_FIT_V1_PROGRAM.pt',weights_only=True)['0.25']
    s,_,q=unpack(item)
    n=torch.load(p/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
    keys=torch.cat([n[k][1].double() for k in ('k1','k2')])
    cache=torch.load(p/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    native=F.rms_norm(cache['raw9'].float(),(1152,),eps=torch.finfo(torch.float32).eps).reshape(-1,1152)
    rows=[];controls=[];prices=[]
    for dtype in (torch.float64,torch.float32):
        before=time.perf_counter();sp,dp=prepare(s,q,keys,dtype);setup=time.perf_counter()-before
        label=str(dtype)
        for name,x in [('native',native.to(dtype)),('independent',torch.randn(37,1152,generator=gen,dtype=dtype))]:
            exact=execute_dense(x.double(),(q,(keys@q).T))
            y=execute_sparse(x,sp).double();dense=execute_dense(x,dp).double()
            controls.append(dict(dtype=label,inputs=name,sparse_relative_error=float((y-exact).norm()/exact.norm()),
                                 dense_relative_error=float((dense-exact).norm()/exact.norm())))
        prices.append(dict(dtype=label,sparse_bytes=byte_count(sp),dense_bytes=byte_count(dp),combined_prepare_seconds=setup,
            scope='Prepared source matrix+both folded key adapters. Shared native Q/K/RMS/value/prefix not counted on either side. Correction absorbed into adapters, not retained.'))
        for count in (1,19,128,512):
            x=torch.randn(count,1152,generator=gen,dtype=dtype)
            for _ in range(7):execute_dense(x,dp);execute_sparse(x,sp)
            samples={'dense':[],'sparse':[]}
            for repeat in range(31):
                names=('dense','sparse') if repeat%2 else ('sparse','dense')
                for name in names:
                    t=time.perf_counter()
                    if name=='dense':execute_dense(x,dp)
                    else:execute_sparse(x,sp)
                    samples[name].append(time.perf_counter()-t)
            dm,sm=[statistics.median(samples[k]) for k in ('dense','sparse')]
            rows.append(dict(dtype=label,token_rows=count,dense_median_seconds=dm,sparse_median_seconds=sm,speedup=dm/sm))
    out=dict(pred_a=all(c['sparse_relative_error']<=(1e-10 if c['dtype']=='torch.float64' else 1e-5) for c in controls),
        pred_b=all(x['speedup']>=1.1 for x in rows),pred_c=all(x['sparse_bytes']<x['dense_bytes'] for x in prices),
        controls=controls,prices=prices,timings=rows,scope=__doc__)
    (p/'SPARSE_PARENT_EXECUTOR_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
