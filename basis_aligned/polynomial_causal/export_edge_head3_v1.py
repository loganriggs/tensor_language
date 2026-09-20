"""Extract one conditional head core and independently compare its writer sum."""
from pathlib import Path
import json,sys,torch
from attention_mixed_edge_core import mixed
from mixed_edge_writer_features import writers,amplitudes
p=Path(__file__).resolve().parent;a=p.parent/'bilinear_quotient/circuits/followups';sys.path.insert(0,str(a.parent.parent/'ops'))
from disk_guard import guard_torch_save
torch.set_num_threads(2)
b=torch.load(a/'v4_compiled_edge_join_v1_cores.pt',map_location='cpu',weights_only=False);groups=[];records=[]
for g in b['groups']:
 c=g['core'];h={}
 for k,v in c.items():
  h[k]=(v[:,3:4] if k in ['product','qnorm','knorm'] else v[:,:,3:4] if k in ['value_base','value_linear','cached_value'] else v).clone()
 errors=[]
 for u,v in [(1,1),(.5,1),(1,.5),(-1,1),(1,-1),(2,1),(1,2),(0,1),(1,0)]:
  reference=(writers(c)*amplitudes(c,u,v)[:,None,:]).reshape(len(c['product']),1152,3,9).sum(2)[:,:,3]
  errors.append(float((mixed(h,u,v)-reference).abs().max()))
 assert max(errors)<1e-10
 values=sum(v.numel() for v in h.values())//len(h['product']);assert values==3480
 groups.append(dict(panel=g['panel'],template=g['template'],families=g['families'],core=h))
 records.append(dict(panel=g['panel'],template=g['template'],max_replay=max(errors),values_per_context=values))
out=a/'v4_edge_head3_program_v1.pt';assert not out.exists();guard_torch_save(dict(groups=groups,head=3,scope='Conditional mixed interaction only; native source/background/suffix preparation external'),str(out))
f=p/'EDGE_HEAD3_EXPORT_CPU_V1.json';assert not f.exists();f.write_text(json.dumps(dict(records=records,scope='Model-free execution of prepared head3 cores, no behavioral/OOD evidence from this CPU replay'),indent=2)+'\n');print(f.read_text())
