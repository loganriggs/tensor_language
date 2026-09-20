"""Cross-head grouping of value branches; no downstream behavioral claim."""
import json,torch
from pathlib import Path
from mixed_edge_writer_features import writers,amplitudes
from attention_mixed_edge_core import mixed
p=Path(__file__).resolve().parent;a=p.parent/'bilinear_quotient/circuits/followups';torch.set_num_threads(2)
b=torch.load(a/'v4_compiled_edge_join_v1_cores.pt',map_location='cpu',weights_only=False);records=[]
for g in b['groups']:
 c=g['core'];w=writers(c)
 for u,v in [(1.,1.),(2.,1.),(1.,2.),(-1.,1.)]:
  z=amplitudes(c,u,v);branches=(w*z[:,None,:]).reshape(len(w),1152,3,9).sum(-1);total=branches.sum(-1);ref=mixed(c,u,v)
  error=float((total-ref).abs().max());assert error<1e-10
  norms=branches.flatten(0,1).norm(dim=0);gram=branches.flatten(0,1).T@branches.flatten(0,1)
  cosine=gram/(norms[:,None]*norms[None,:]).clamp_min(1e-30)
  records.append(dict(panel=g['panel'],template=g['template'],query_amplitude=u,key_amplitude=v,replay=error,relative_branch_norms=(norms/total.norm()).tolist(),branch_cosines=cosine.tolist()))
out=p/'EDGE_VALUE_BRANCH_SCREEN_V1.json';assert not out.exists();out.write_text(json.dumps(dict(branch_order=['baseline_current_value','edited_current_value','cached_first_value'],records=records,scope='Algebraic cross-head branch screen; query/key amplitudes not necessarily subject/attractor; norms/cosines do not prove causal importance'),indent=2)+'\n')
print({'max_replay':max(r['replay'] for r in records),'range_relative_norm':[(min(r['relative_branch_norms'][i] for r in records),max(r['relative_branch_norms'][i] for r in records)) for i in range(3)]})
