"""Cold CPU portable replay; no checkpoint, tokenizer, prefix or suffix access."""
from pathlib import Path
import json,torch
from attention_mixed_edge_core import mixed
from mixed_edge_writer_features import execute,writers
p=Path(__file__).resolve().parent;a=p.parent/'bilinear_quotient/circuits/followups';torch.set_num_threads(2)
bundle=torch.load(a/'v4_compiled_edge_join_v1_cores.pt',weights_only=False,map_location='cpu');records=[]
for g in bundle['groups']:
 core=g['core'];native=float((mixed(core,1.,1.)-g['edge']).abs().max());factor=0.;dead=0.
 for u,v in [(-.25,.5),(.25,.75),(.5,.5),(1.,1.),(0.,1.),(1.,0.)]:
  y=execute(core,u,v);factor=max(factor,float((y-mixed(core,u,v)).abs().max()))
  if u==0 or v==0:dead=max(dead,float(y.abs().max()))
 sv=torch.linalg.svdvals(writers(core));ranks=(sv>sv[:,:1]*1e-10).sum(-1)
 assert native<1e-10 and factor<1e-10 and dead<1e-12
 records.append(dict(panel=g['panel'],template=g['template'],native_saved_replay=native,writer_factor_replay=factor,zero_axis=dead,writer_columns=writers(core).shape[-1],numerical_writer_rank=ranks.tolist(),coefficients_per_input=g['coefficients_per_input']))
result=dict(records=records,scope='CPU extracted conditional edge core and27writer factorization only; native coefficient/background/suffix production remains external; numerical rank not semantic minimality')
f=p/'COMPILED_EDGE_CPU_REPLAY_V1.json';assert not f.exists();f.write_text(json.dumps(result,indent=2)+'\n');print(records)
