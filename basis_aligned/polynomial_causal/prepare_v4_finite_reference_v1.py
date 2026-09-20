"""CPU finite-reference diagnostic; no gradients or native outcomes refitted."""
from pathlib import Path
import json,numpy as np,torch
from shared_selective_source_lp import choose
from refined_native_sources import PARENTS
p=Path(__file__).resolve().parent;a=p.parent/'bilinear_quotient/circuits/followups';torch.set_num_threads(2)
g=np.array([[-1.,0.],[0.,1.]])
base,cb=choose(g,[1,0]);explicit,ce=choose(g,[1,0],target_effect=1.);double,cd=choose(g,[1,0],target_effect=2.);negative,cn=choose(g,[1,0],target_effect=-1.)
assert np.array_equal(base,explicit) and cb==ce and abs(cd['retention']-.5)<1e-12 and abs(cn['retention']-1)<1e-12 and negative[0]==-1
prior=json.loads((a/'fixed_query_fresh_v4_result.json').read_text());groups=torch.load(a/'v4_source_oracle_audit_v1_tensors.pt',weights_only=False,map_location='cpu')['groups'];ref=np.array([0.,0.,1.,1.,1.,0.])[PARENTS];records=[]
for group in groups:
 grad=torch.einsum('bod,bkd->bok',group['reader'],group['sources']);grad[:,0]*=group['orientation'][:,None];grad=grad.numpy();families=group['families'];used={f:0 for f in families};targets=[]
 for family in families:
  old=next(r for r in prior['records'] if r['dataset']=='v4' and r['panel']==group['panel'] and r['role']==group['role'] and r['family']==family);targets.append(old['reference_effect'][used[family]][0]);used[family]+=1
 weights=[];checks=[]
 for row,target in zip(grad,targets):
  aa,check=choose(row,ref,target_effect=target);weights.append(aa.tolist());checks.append(check)
 records.append(dict(panel=group['panel'],template=group['template'],role=group['role'],families=families,targets=targets,amplitudes=weights,checks=checks,prediction=(-np.einsum('bok,bk->bo',grad,np.array(weights))).tolist()))
result=dict(planted=dict(default_explicit_equal=True,doubled_target_retention=cd['retention'],negative_target_coefficient=negative[0]),records=records,scope='Same bounded linear optimization, measured native finite target supplied; still per-input derivative and reference dependent')
target=p/'V4_FINITE_REFERENCE_CHOICES_V1.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print('prepared',len(records),'groups; min gamma',min(c['retention'] for r in records for c in r['checks']))
