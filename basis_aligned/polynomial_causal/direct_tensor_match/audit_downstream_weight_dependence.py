"""Audit correlations omitted by independent source-gradient objectives."""
from pathlib import Path
import torch,json
from source_graph_metrics import matrices
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());parent=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[meta['winner']];newmeta=json.loads((P/'SOURCE_SOBOLEV_REFIT_V1.json').read_text());new=torch.load(P/'SOURCE_SOBOLEV_REFIT_PROGRAMS_V1.pt',weights_only=True)
z=d['z'];h=d['h'];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();delta=z-d['mu']
Q=torch.stack([q for p in d['pairs'][:2] for q in p['Qs']]);q=torch.einsum('ni,oij,nj->no',z,Q,z);records=[]
for name,p in [('parent',parent),('lambda1',new[newmeta['winners']['1']]),('lambda10',new[newmeta['winners']['10']])]:
 D=matrices(p['shared_mixed'])-Q;responses=torch.einsum('oij,nj->noi',D,delta)
 for j,pair in enumerate(d['pairs'][:2]):
  u=(h@pair['a']-.5*q[:,2*j])/s-pair['alpha'];v=q[:,2*j+1]/s-pair['beta'];a=-v/s;b=2*u/s
  for panel,ids in [('train',torch.arange(24*64)),('opened',d['indices'])]:
   c=a[ids];e=b[ids];A=responses[ids,2*j];B=responses[ids,2*j+1]
   energy=(c[:,None]*A+e[:,None]*B).square().sum(-1)
   exact=energy.mean();independent=c.square().mean()*A.square().sum(-1).mean()+e.square().mean()*B.square().sum(-1).mean()+2*(c*e).mean()*(A*B).sum(-1).mean()
   ess=[float(x.square().sum().square()/x.pow(4).sum()) for x in [c,e]]
   unique=torch.unique(ids//64);chunks=torch.stack([energy[(ids//64)==k].sum() for k in unique])
   record=dict(candidate=name,component=j+1,panel=panel,sites=len(ids),independent_to_exact_squared_error_ratio=float(independent/exact),amplitude_effective_sites=ess,max_chunk_error_energy_share=float(chunks.max()/chunks.sum()),top_one_percent_site_error_energy_share=float(energy.topk(max(1,round(.01*len(ids)))).values.sum()/energy.sum()))
   records.append(record);print(record,flush=True)
out=dict(records=records,predictions=dict(pred_a_independence=all(abs(r['independent_to_exact_squared_error_ratio']-1)<=.1 for r in records if r['candidate']=='parent')),scope='Correlation diagnostic on cached chunks; no document independence. ESS applies to squared amplitude weights only, not a general confidence interval. Includes signed crossmoment in independence surrogate; exact metric positive. No fitting or fresh native evidence.')
(P/'DOWNSTREAM_WEIGHT_DEPENDENCE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'])
