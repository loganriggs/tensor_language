"""Exact actual-weight and matched compilation/amortization cost control."""
import json,time,statistics
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from response_product_basis_v2 import prepare_basis,prepare_products
from fixed_writer_products_v1 import prepare_global,products
P=Path(__file__).resolve().parent

@torch.no_grad()
def main(contiguous_private=False,result_prefix='FIXED_WRITER_PRODUCTS_V1'):
    torch.set_num_threads(2);torch.manual_seed(61601)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    lam=float(sd['transformer.h.10.lambdas'][0]);writer=lam*program['direction'].double()
    l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    l9,r9,d9=[sd['transformer.h.9.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    start=time.perf_counter();cached=prepare_global(writer,l,r,d);cheap_seconds=time.perf_counter()-start
    start=time.perf_counter();compiled=prepare_global(writer,l,r,d,True);compile_seconds=time.perf_counter()-start
    rows=[]
    for batch in (1,16,128):
        z=torch.randn(batch,1152,dtype=torch.float64)
        base=((z@l9.T)*(z@r9.T))@d9.T/(z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)
        _,basis=prepare_basis(z,base,program,lam)
        functions={'original':lambda:prepare_products(basis,l,r,d)[1],
            'cached':lambda:products(basis,l,r,d,cached,contiguous_private),'compiled':lambda:products(basis,l,r,d,compiled,contiguous_private)}
        ref=functions['original']();errors={k:float((fn()-ref).norm()/ref.norm()) for k,fn in functions.items()};assert max(errors.values())<1e-10
        samples={k:[] for k in functions}
        for fn in functions.values():fn();fn()
        for trial in range(7):
            order=list(functions)
            if trial%2:order.reverse()
            for name in order:
                start=time.perf_counter();functions[name]();samples[name].append(time.perf_counter()-start)
        med={k:statistics.median(v) for k,v in samples.items()};gain=med['cached']-med['compiled']
        rows.append(dict(batch=batch,relative_errors=errors,median_seconds=med,
            cached_speedup=med['original']/med['cached'],compiled_speedup_over_cached=med['cached']/med['compiled'],
            compiled_break_even_contexts=None if gain<=0 else max(0,compile_seconds-cheap_seconds)/gain*batch))
    result=dict(pred_a=True,pred_b=all(r['cached_speedup']>=1.1 for r in rows),pred_c=all(r['compiled_speedup_over_cached']>=1.1 for r in rows),
        rows=rows,contiguous_private=contiguous_private,cheap_global_preparation_seconds=cheap_seconds,dense_global_preparation_seconds=compile_seconds,
        cached_global_scalars=sum(x.numel() for x in cached.values()),compiled_global_scalars=sum(x.numel() for x in compiled.values()),
        scope='Actual weights,synthetic pristinecontexts,FP64two-threadCPU; same six-product outputs, fixed writer reused acrosscontexts. Globalsetup andstoragecharged, nativeL/R/Down retained. Productbankpreparation only, not fullbranch/modeladoption. One-shot global compiletime makes break-even descriptive.')
    (P/(result_prefix+'_CONTROL.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
