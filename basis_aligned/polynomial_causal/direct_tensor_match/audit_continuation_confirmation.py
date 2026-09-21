"""Paired document bootstrap for frozen continuation-group confirmation."""
from pathlib import Path
import json,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2)
r=json.loads((p/'MIDPOINT_CONTINUATION_GROUP_NATIVE_V1.json').read_text());rows=r['records'];dtype=torch.float64
counts=torch.zeros(32,2,dtype=dtype);ce=torch.zeros(32,2,2,dtype=dtype);odds=torch.zeros_like(ce)
for x in rows:
 doc=x['document']-112;arm=['real','sham'].index(x['arm'])
 for j,field in enumerate(['continuation','spaced_word']):
  if x[field]:
   if arm==0:counts[doc,j]+=1
   ce[doc,arm,j]+=x['ce_added'];odds[doc,arm,j]+=x['continuation_logodds_decrease']
gen=torch.Generator().manual_seed(261344);draw=torch.randint(32,(10000,32),generator=gen);weights=torch.zeros(10000,32,dtype=dtype);weights.scatter_add_(1,draw,torch.ones_like(draw,dtype=dtype))
C=weights@counts;CE=(weights@ce.flatten(1)).reshape(-1,2,2)/C[:,None,:];O=(weights@odds.flatten(1)).reshape(-1,2,2)/C[:,None,:]
interval=lambda t:torch.quantile(t,torch.tensor([.025,.975],dtype=dtype)).tolist()
summary=dict(real_continuation_ce_interval=interval(CE[:,0,0]),real_spaced_ce_interval=interval(CE[:,0,1]),real_minus_sham_continuation_ce_interval=interval(CE[:,0,0]-CE[:,1,0]),real_minus_sham_continuation_logodds_interval=interval(O[:,0,0]-O[:,1,0]))
keys=[x['input_token'] for x in r['same_token_secondary']['strata']];lookup={k:j for j,k in enumerate(keys)};N=torch.zeros(32,len(keys),2,dtype=dtype);Y=torch.zeros_like(N)
for x in rows:
 if x['arm']!='real' or x['input_token'] not in lookup:continue
 for j,field in enumerate(['continuation','spaced_word']):
  if x[field]:N[x['document']-112,lookup[x['input_token']],j]+=1;Y[x['document']-112,lookup[x['input_token']],j]+=x['continuation_logodds_decrease']
NN=(weights@N.flatten(1)).reshape(-1,len(keys),2);YY=(weights@Y.flatten(1)).reshape_as(NN);means=YY/NN.clamp_min(1);matchedweight=NN.min(2).values
contrast=((means[:,:,0]-means[:,:,1])*matchedweight).sum(1)/matchedweight.sum(1)
summary['same_token_logodds_contrast_interval']=interval(contrast)
out=p/'MIDPOINT_CONTINUATION_GROUP_AUDIT_V1.json';assert not out.exists();out.write_text(json.dumps(dict(summary=summary,draws=10000,scope='Paired document bootstrap over32frozen confirmation rows. Recomputes same-current-token stratum means and min-count weights within each resample, allowing strata to lose coverage. Descriptive intervals conditional on panel and discovered hypothesis; not familywise guarantees or randomized context treatment.'),indent=2)+'\n');print(out.read_text())
