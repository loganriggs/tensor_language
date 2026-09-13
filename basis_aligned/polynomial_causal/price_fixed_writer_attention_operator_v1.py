"""Price an exact mixed interaction compiler, including a precompiled baseline."""
from pathlib import Path
import json,time,statistics,torch
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(9131000)
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
 program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
 w=program['direction'].double()*float(sd['transformer.h.10.lambdas'][0])
 l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']];o=sd['transformer.h.10.attn.c_proj.weight'].double()
 tic=time.perf_counter();lw=l@w;rw=r@w;j=(d*rw[None,:])@l+(d*lw[None,:])@r;jtime=time.perf_counter()-tic
 tic=time.perf_counter();m=j@o;mtime=time.perf_counter()-tic;rows=[]
 for batch in [1,16,128]:
  u=torch.randn(batch,1152,dtype=torch.float64)
  def direct():
   x=u@o.T
   return ((x@l.T)*rw+(x@r.T)*lw)@d.T
  functions={'native_maps':direct,'precompiled_J_then_O':lambda:(u@o.T)@j.T,'folded':lambda:u@m.T}
  ref=direct();err=float((functions['folded']()-ref).norm()/ref.norm());assert err<1e-10
  samples={k:[] for k in functions}
  for fn in functions.values():fn();fn()
  for trial in range(9):
   keys=list(functions)
   if trial%2:keys.reverse()
   for key in keys:
    tic=time.perf_counter();functions[key]();samples[key].append(time.perf_counter()-tic)
  times={k:statistics.median(v) for k,v in samples.items()};saving=times['precompiled_J_then_O']-times['folded']
  rows.append(dict(batch=batch,relative_error=err,seconds=times,folded_vs_precompiled_speedup=times['precompiled_J_then_O']/times['folded'],break_even_inputs_after_J=None if saving<=0 else mtime/saving*batch))
 result=dict(rows=rows,compile_J_seconds=jtime,additional_fold_seconds=mtime,compiled_scalars=m.numel(),unshared_J_and_O_scalars=j.numel()+o.numel(),
  pred_a=all(x['folded_vs_precompiled_speedup']>=1.1 for x in rows),
  scope='Isolated K(lambda*w,O*u), exact FP64 CPU. Shared global M reusable at all contexts/positions; constructing head-write u, other two residual modes, normalization, background and suffix excluded. If O/J still needed elsewhere, M is additional storage rather than a replacement. No native semantic intervention or whole-branch speed claim.')
 (P/'FIXED_WRITER_ATTENTION_OPERATOR_V1_PRICE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
