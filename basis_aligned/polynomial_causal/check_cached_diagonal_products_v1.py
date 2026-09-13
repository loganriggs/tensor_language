"""Compare cached five-bank directly with existing five-bank preparation."""
import json,time,statistics
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis
from response_diagonal_conic_v1 import prepare_direct
from fixed_writer_products_v1 import prepare_global
from cached_diagonal_products_v1 import prepare
P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(61611)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    lam=float(sd['transformer.h.10.lambdas'][0]);writer=lam*program['direction'].double()
    l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    l9,r9,d9=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    start=time.perf_counter();global_state=prepare_global(writer,l,r,d);setup=time.perf_counter()-start;rows=[]
    for batch in (1,16,128):
        z=torch.randn(batch,1152,dtype=torch.float64);base=((z@l9.T)*(z@r9.T))@d9.T/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
        ctx,basis=prepare_basis(z,base,program,lam)
        baseline=lambda:prepare_direct(basis,l,r,d,ctx)
        candidate=lambda:prepare(basis,l,r,d,ctx,global_state)
        ref=baseline();fit=candidate();errors=((ref-fit).norm(dim=(0,2))/ref.norm(dim=(0,2))).tolist();assert max(errors)<1e-10
        baseline();candidate();timings=[[],[]]
        for trial in range(7):
            for index in ([0,1] if trial%2==0 else [1,0]):
                start=time.perf_counter();[baseline,candidate][index]();timings[index].append(time.perf_counter()-start)
        old,new=[statistics.median(x) for x in timings];gain=old-new
        rows.append(dict(batch=batch,per_vector_relative_errors=errors,baseline_seconds=old,cached_seconds=new,
            reused_speedup=old/new,including_first_setup_speedup=old/(new+setup),break_even_contexts=None if gain<=0 else setup/gain*batch))
    result=dict(pred_a=True,pred_b=all(r['reused_speedup']>=1.1 for r in rows),rows=rows,setup_seconds=setup,
        cache_scalars=sum(x.numel() for x in global_state.values()),
        scope='Actualweights/syntheticcontexts,FP64two-threadCPU; optimizedfivebaseline vsfixedwritercachedfive. Four varyingDownwrites plusglobalP00. Outputstillmaterializesfivevectors/context; onlypreparationreuse, notextraoutputstorageorwholebranchgain.')
    (P/'CACHED_DIAGONAL_PRODUCTS_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
