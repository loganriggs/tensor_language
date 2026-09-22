"""Warm eager CPU microbenchmark of exported16-output polynomial programs."""
import json,statistics,time
from pathlib import Path
import torch
from conditional_quartic_cp import evaluate as full
from lean_conditional_cp import evaluate as lean
P=Path(__file__).resolve().parent

def cp(p,x):
 z=x@p['factors'][0].T
 for a in p['factors'][1:]:z=z*(x@a.T)
 return z@p['coefficients'].T

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);x=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True)['rows'].float();programs=[]
 for seed in [1001,1002]:
  variants=[('CP512',f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',cp),('conditional256',f'CONDITIONAL_CP_SEED{seed}_RANK256_V1.pt',full)]+[(f'lean{r}',f'LEAN_CONDITIONAL_CP_SEED{seed}_RANK{r}_V1.pt',lean) for r in [64,128,256]]
  for name,file,fn in variants:programs.append((seed,name,torch.load(P/file,weights_only=True),fn))
 timings={(s,n,b):[] for s,n,p,f in programs for b in [1,64,2048]}
 for batch in [1,64,2048]:
  xx=x[:batch];repeat={1:100,64:10,2048:1}[batch]
  for seed,name,p,fn in programs:assert torch.isfinite(fn(p,xx)).all()
  for round_index in range(7):
   # Rotate and reverse order to distribute transient shared-machine effects.
   order=programs[round_index:]+programs[:round_index]
   if round_index%2:order=order[::-1]
   for seed,name,p,fn in order:
    start=time.perf_counter()
    for _ in range(repeat):fn(p,xx)
    timings[seed,name,batch].append((time.perf_counter()-start)/repeat)
 rows=[]
 for (seed,name,batch),t in timings.items():
  rows.append(dict(seed=seed,program=name,batch=batch,median_seconds=statistics.median(t),min_seconds=min(t),max_seconds=max(t),median_microseconds_per_state=statistics.median(t)/batch*1e6,timings_seconds=t))
 baseline={(r['seed'],r['batch']):r['median_seconds'] for r in rows if r['program']=='CP512'}
 for r in rows:r['speedup_over_CP512']=baseline[r['seed'],r['batch']]/r['median_seconds']
 cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
 out=dict(rows=rows,cpu=cpu,torch_version=torch.__version__,threads=2,dtype='float32',scope='Warm eager CPU,7rounds, rotating/reversedorder,1/64/2048statebatches; disk/modelcapture/fixedwriter/vocabulary/normsoftcap excluded equally. Sharedinstance timing, not GPU or end-to-end transformer speedup. Frozen exported programs, no compilation or fitting.')
 (P/'CONDITIONAL_CP_CPU_BENCHMARK_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in rows:
  if r['seed']==1001:print(r['program'],r['batch'],round(r['median_microseconds_per_state'],3),round(r['speedup_over_CP512'],3))
if __name__=='__main__':main()
