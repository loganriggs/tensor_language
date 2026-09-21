"""Exact error decomposition for fixed-h conditional component Jacobians."""
from pathlib import Path
import torch,json
from source_graph_metrics import matrices
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);fit=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());old=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[fit['winner']];newfit=json.loads((P/'SOURCE_SOBOLEV_REFIT_V1.json').read_text());ps=torch.load(P/'SOURCE_SOBOLEV_REFIT_PROGRAMS_V1.pt',weights_only=True)
ids=d['indices'];z=d['z'][ids];h=d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
Q=torch.stack([q for p in d['pairs'][:2] for q in p['Qs']]);true=torch.einsum('ni,oij,nj->no',z,Q,z);gtrue=2*torch.einsum('oij,nj->noi',Q,z)
records=[]
for name,p in [('parent',old),('lambda1',ps[newfit['winners']['1']]),('lambda10',ps[newfit['winners']['10']])]:
 part=p['shared_mixed'];hat=matrices(part);q=torch.einsum('ni,oij,nj->no',z,hat,z)+z@part['source_linear']+part['source_bias'];gq=2*torch.einsum('oij,nj->noi',hat,z)+part['source_linear'].T[None]
 for j,pair in enumerate(d['pairs'][:2]):
  a,b=2*j,2*j+1;u=(h@pair['a']-.5*true[:,a])/s-pair['alpha'];v=true[:,b]/s-pair['beta'];du=-.5*(q[:,a]-true[:,a])/s;dv=(q[:,b]-true[:,b])/s
  ga,gb=gtrue[:,a],gtrue[:,b];da,db=gq[:,a]-ga,gq[:,b]-gb
  A=-.5*(v/s)[:,None]*da+(u/s)[:,None]*db
  B=-.5*(dv/s)[:,None]*ga+(du/s)[:,None]*gb
  C=-.5*(dv/s)[:,None]*da+(du/s)[:,None]*db
  J=-.5*(v/s)[:,None]*ga+(u/s)[:,None]*gb
  Jh=-.5*((v+dv)/s)[:,None]*gq[:,a]+((u+du)/s)[:,None]*gq[:,b]
  delta=Jh-J;replay=float((A+B+C-delta).norm()/delta.norm());assert replay<1e-8
  rec=dict(candidate=name,component=j+1,replay=replay,total_relative_error=float(delta.norm()/J.norm()),gradient_term_over_total=float(A.norm()/delta.norm()),value_plus_cross_over_total=float((B+C).norm()/delta.norm()),gradient_term_cosine_with_total=float((A*delta).sum()/(A.norm()*delta.norm())))
  records.append(rec);print(rec)
out=dict(records=records,predictions=dict(pred_a_replay=max(r['replay'] for r in records)<1e-8,pred_b_gradient_dominates=all(r['value_plus_cross_over_total']<=.1 for r in records)),scope='Exact algebraic decomposition on448opened states, h fixed. Norm fractions need not sum to1 because errors can align or cancel. No independent causal or fresh-data evidence.')
(P/'JACOBIAN_ERROR_TERMS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
