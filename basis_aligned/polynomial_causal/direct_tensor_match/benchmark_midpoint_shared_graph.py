from pathlib import Path
import torch,json,time,statistics
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
original={k:v.float() for k,v in torch.load(p/'MIDPOINT_OUTPUT_CORRECTION_GRAPHS_V1.pt',weights_only=True)['rank8'].items()};shared={k:v.float() for k,v in torch.load(p/'MIDPOINT_GRAPH_INPUT_MODE_GRAPHS_V1.pt',weights_only=True)['rank256'].items()};rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1);m=rows['m'].flatten(0,1)
def execute(e,n,m):
 left=(n@e['Pn'])@e['Tn'] if 'Pn' in e else n@e['A'];right=(m@e['Pm'])@e['Tm'] if 'Pm' in e else m@e['B'];phi=left*right-e['product_mean'];return phi.reshape(len(n),256,4).sum(-1)@e['group_writers'].T+(phi@e['correction_left'])@e['correction_writers'].T+e['full_mean']
records=[]
for size in [128,2048]:
 for label,e in [('original',original),('shared256',shared)]:
  for _ in range(3):execute(e,n[:size],m[:size])
  times=[]
  for _ in range(15):
   start=time.perf_counter();execute(e,n[:size],m[:size]);times.append(time.perf_counter()-start)
  records.append(dict(program=label,rows=size,median_seconds=statistics.median(times),min_seconds=min(times),max_seconds=max(times)))
out=p/'MIDPOINT_SHARED_GRAPH_CPU_BENCH_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,threads=2,dtype='float32',scope='LocalCPU microbenchmark of actual standalonegraphs on cachednormalizedinputs. Excludes upstreammodel, normalization, output coordinate conversion, all compile/fitting work. Sharedhost timing, notGPU orwholemodel speed evidence.'),indent=2)+'\n');print(out.read_text())
