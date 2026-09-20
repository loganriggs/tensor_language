"""Exact signed partition of reader context change, not causal attribution."""
from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;a=p.parent/'bilinear_quotient/circuits/followups';torch.set_num_threads(2)
groups=torch.load(a/'attention_reader_fold_v1_tensors.pt',weights_only=False,map_location='cpu')['groups'];sources=torch.load(a/'residual_reader_transfer_v1_tensors.pt',weights_only=False,map_location='cpu')['groups'];rows=[]
for g in groups:
 if g['panel']!='congruent':continue
 donor=next(v for v in groups if v['panel']=='opposite' and v['template']==g['template'] and v['role']==g['role']);ds=next(v['sources'] for v in sources if v['panel']=='congruent' and v['template']==g['template'] and v['role']==g['role'])
 ids=torch.arange(len(ds));ids=ids^1 if g['role']=='attractor' else ids
 delta=g['parts']-donor['parts'][ids];contract=torch.einsum('bfod,bkd->bfok',delta,ds);total=contract.sum(1)
 shares=(contract*total[:,None]).sum((0,3))/total.square().sum((0,2)).clamp_min(1e-30)[None]
 rows.append(dict(role=g['role'],template=g['template'],signed_shares=shares.tolist(),closure=float((shares.sum(0)-1).abs().max())))
out=dict(branches=['residual','Q1','K1','Q2','K2','V'],records=rows,scope='Canonical number reader, source-number-matched donor, recipient source directions; exact algebraic partition, not independent causal effects')
assert max(r['closure'] for r in rows)<1e-10
path=p/'ATTENTION_READER_CONTEXT_V1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['role'],r['template'],[round(v[0],3) for v in r['signed_shares']])
