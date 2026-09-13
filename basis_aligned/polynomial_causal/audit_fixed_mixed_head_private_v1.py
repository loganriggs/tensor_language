"""Postfit cancellation, stationarity and literal execution counterchecks."""
from pathlib import Path
import json,time,statistics,torch
from head17_source_interface_v1 import CHECKPOINT
from head_private_matrix_v1 import reconstruct,cycle,lowrank
from head_private_matrix_executor_v1 import execute
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(71011)
 result=json.loads((P/'FIXED_MIXED_HEAD_PRIVATE_V1_RESULT.json').read_text())
 f=torch.load(P/'FIXED_MIXED_HEAD_PRIVATE_V1_PROGRAM.pt',weights_only=True);fit=reconstruct(f)
 s=f['shared_left']@f['shared_right'];p=(f['private_left']@f['private_right']).permute(1,0,2).reshape_as(s)
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu');program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
 w=program['direction'].double()*float(sd['transformer.h.10.lambdas'][0]);l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']];o=sd['transformer.h.10.attn.c_proj.weight'].double()
 x=((d*(r@w)[None,:])@l+(d*(l@w)[None,:])@r)@o;n=x.norm();err=(fit-x).norm()/n
 s2,p2,_=cycle(x,p,9,128,16);nextloss=float((x-s2-p2).square().sum()/n.square());base,bl,br=lowrank(x,208)
 # Equal private widths are an assumption. Optimize their allocation at fixed
 # shared matrix and identical total private rank/parameter count.
 residual=(x-s).reshape(1152,9,128).permute(1,0,2)
 pu,ps,pv=torch.linalg.svd(residual,full_matrices=False)
 indices=ps.square().flatten().topk(144).indices
 counts=torch.bincount(indices//128,minlength=9)
 allocated=torch.stack([(pu[h,:,:int(k)]*ps[h,:int(k)])@pv[h,:int(k)] for h,k in enumerate(counts)])
 allocated_loss=float((residual-allocated).square().sum()/n.square())
 timings=[]
 for batch in [1,16,128]:
  u=torch.randn(batch,1152,dtype=torch.float64);ref=u@fit.T;pred=execute(u,f);replay=float((ref-pred).norm()/ref.norm());assert replay<1e-12
  funcs={'structured':lambda:execute(u,f),'global208':lambda:(u@br.T)@bl.T,'dense_exact':lambda:u@x.T};samples={k:[] for k in funcs}
  for fn in funcs.values():fn();fn()
  for trial in range(7):
   keys=list(funcs)
   if trial%2:keys.reverse()
   for key in keys:
    tic=time.perf_counter();funcs[key]();samples[key].append(time.perf_counter()-tic)
  timings.append(dict(batch=batch,replay_error=replay,median_seconds={k:statistics.median(v) for k,v in samples.items()}))
 audit=dict(shared_norm_over_target=float(s.norm()/n),private_norm_over_target=float(p.norm()/n),shared_private_cosine=float((s*p).sum()/(s.norm()*p.norm())),
  serialized_relative_error=float(err),conditional_cycle_relative_gain=(float(err.square())-nextloss)/float(err.square()),timings=timings,
  private_reallocated_ranks=counts.tolist(),fixed_shared_reallocation_squared_error=allocated_loss,fixed_shared_reallocation_relative_gain=1-allocated_loss/float(err.square()),
  scope='CPU postfit diagnostic on serialized best weights. Same scalar count for global208/structured, different approximation errors. One-cycle gain and component norms are not uniqueness/global-optimality certificates. No new fit adopted.')
 (P/'FIXED_MIXED_HEAD_PRIVATE_V1_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
