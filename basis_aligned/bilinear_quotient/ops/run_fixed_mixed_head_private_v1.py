#!/usr/bin/env python3
# BQGATE:10starts6cycles;top3x80cycles;420seconds.
"""pred_a monotone conditional updates and serialized replay<=1e-5.
pred_b at least two promoted fits plateau for three relative gains<=1e-7.
pred_c >=5% squared error improvement vs global rank208 at479232scalars.
Null: head-private structure does not beat equally priced global factors.
Price10starts6cycles,top3x80cycles,420seconds;weights only, no text fit.
"""
from pathlib import Path
from hashlib import sha256
import torch,json,sys,os,time,signal
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from head17_source_interface_v1 import CHECKPOINT
from head_private_matrix_v1 import lowrank,private_fit,cycle,reconstruct
@torch.no_grad()
def main():
 files=json.loads((P/'FIXED_MIXED_HEAD_PRIVATE_V1_BINDING.json').read_text())['files']
 assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('10starts6cycles;top3x80cycles;479232scalars;420seconds');return
 assert not (P/'FIXED_MIXED_HEAD_PRIVATE_V1_RESULT.json').exists()
 signal.alarm(420);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
 program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
 w=program['direction'].double().cuda()*float(sd['transformer.h.10.lambdas'][0])
 l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double().cuda() for k in ['Left','Right','Down']];o=sd['transformer.h.10.attn.c_proj.weight'].double().cuda()
 x=((d*(r@w)[None,:])@l+(d*(l@w)[None,:])@r)@o;norm=float(x.square().sum())
 baseline,_,_=lowrank(x,208);baseline_loss=float((x-baseline).square().sum()/norm)
 screened=[];states=[];monotone=True
 for seed in range(10):
  torch.manual_seed(9131007+seed)
  if seed==0:private=torch.zeros_like(x)
  elif seed==1:private=private_fit(x,9,16)[0]
  else:
   private=torch.randn_like(x);private*=x.norm()/private.norm()*(.1 if seed<6 else .5)
   private=private_fit(private,9,16)[0]
  history=[]
  for step in range(6):
   shared,private,factors=cycle(x,private,9,128,16);loss=float((x-shared-private).square().sum()/norm);history.append(loss)
  monotone &= all(b<=a+1e-10 for a,b in zip(history,history[1:]));screened.append(dict(seed=seed,loss=loss,history=history));states.append(private)
 promoted=[];best=None
 for index in sorted(range(10),key=lambda i:screened[i]['loss'])[:3]:
  private=states[index];old=screened[index]['loss'];history=[];stable=0;reason='step_limit'
  for step in range(80):
   if time.perf_counter()-tic>380:reason='time_limit';break
   shared,private,factors=cycle(x,private,9,128,16);loss=float((x-shared-private).square().sum()/norm)
   gain=(old-loss)/max(old,1e-300);monotone &= loss<=old+1e-10;history.append(dict(loss=loss,relative_gain=gain));stable=stable+1 if 0<=gain<=1e-7 else 0;old=loss
   if best is None or loss<best[0]:best=(loss,{k:v.cpu() for k,v in factors.items()})
   if stable>=3:reason='objective_plateau';break
  promoted.append(dict(seed=index,loss=old,stop=reason,history=history))
 assert best is not None
 factors=best[1];assert sum(v.numel() for v in factors.values())==479232
 path=P/'FIXED_MIXED_HEAD_PRIVATE_V1_PROGRAM.pt';torch.save(factors,path)
 replay=reconstruct(torch.load(path,weights_only=True)).cuda();replayloss=float((x-replay).square().sum()/norm)
 predictions={'pred_a':monotone and abs(replayloss-best[0])<=1e-5,'pred_b':sum(r['stop']=='objective_plateau' for r in promoted)>=2,'pred_c':best[0]<=.95*baseline_loss}
 result=dict(**predictions,baseline_rank208_squared_error=baseline_loss,best_squared_error=best[0],squared_error_gain=1-best[0]/baseline_loss,
  screened=screened,promoted=promoted,serialized_loss=replayloss,scalars=479232,serialized_bytes=path.stat().st_size,program_sha256=sha256(path.read_bytes()).hexdigest(),seconds=time.perf_counter()-tic,
  scope='Weights-only sharedrank128 plus9private rank16 blocks, matchedglobal208. Exact conditional SVD, objective plateau not global optimality. No native behavior, OOD, selective manipulation, or runtime claim.')
 (P/'FIXED_MIXED_HEAD_PRIVATE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['screened','promoted']},indent=2))
if __name__=='__main__':main()
