"""Conditional writer-bank screen; state norms do not establish causal importance."""
from pathlib import Path
import json,torch
from mixed_edge_writer_features import writers,amplitudes
p=Path(__file__).resolve().parent;a=p.parent/'bilinear_quotient/circuits/followups';torch.set_num_threads(2)
b=torch.load(a/'v4_compiled_edge_join_v1_cores.pt',map_location='cpu',weights_only=False)
rows=[]
for g in b['groups']:
 w=writers(g['core']);z=amplitudes(g['core'],1.,1.);atoms=w*z[:,None,:]
 heads=atoms.reshape(len(w),1152,3,9).sum(2);full=heads.sum(-1)
 denom=full.norm().clamp_min(1e-30)
 errors=[float(heads[:,:,i].norm()/denom) for i in range(9)]
 replay=float((full-g['edge']).abs().max());assert replay<1e-10
 rows.append(dict(panel=g['panel'],template=g['template'],relative_write_change_if_head_removed=errors,shared_writer_replay=replay,scope='Local residual vector, no downstream reader or behavioral inference'))
result=dict(records=rows,scope='Weight/core-driven native-head grouping screen only; a native ablation must test which groups matter under registered output controls')
out=p/'EDGE_HEAD_WRITE_SCREEN_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
