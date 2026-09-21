"""Independent native decode, restart identity and centered private-branch removal."""
from pathlib import Path
import json,torch
from global_mixed_source_graph import source_reads
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'SHARED_PRIVATE_DIRECTIONS_V1.json').read_text());programs=torch.load(P/'SHARED_PRIVATE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);T=d['teacher'];F=T.flatten(1);e,U=torch.linalg.eigh(F@F.T);truthcores=U.T@F
ids=d['indices'];z=d['z'][ids];h=d['h'][ids];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[:,None]
hread=torch.stack([pair['a'] for pair in d['pairs']],1);alpha=torch.stack([pair['alpha'] for pair in d['pairs']]);beta=torch.stack([pair['beta'] for pair in d['pairs']]);truth=torch.stack([pair['truth'][ids] for pair in d['pairs']],1);den=(truth-truth.mean(0)).norm(dim=0)
def phi(q):return ((h@hread-.5*q[:,::2])/scale-alpha)*(q[:,1::2]/scale-beta)
rows=[];private_forms={}
for rec in meta['records']:
 p=programs[rec['key']];L=p['left_reader'];R=p['right_reader'];W=p['product_weights'];V=p['square_reader'];v=p['square_weights']
 raw=torch.einsum('ir,ro,jr->oij',L,W,R);Q=(raw+raw.transpose(-1,-2))/2;private=(V*v)@V.T;Q[5]+=private
 native=torch.einsum('ni,oij,nj->no',z,Q,z)+z@p['source_linear']+p['source_bias'];direct=source_reads(z,p);direct[:,5]+=(z@V).square()@v;execution=float((native-direct).norm()/native.norm())
 hat=torch.einsum('ai,oij,jb->oab',root,Q,root)/d['scales'][:,None,None];coef=float((hat-T).norm()/T.norm());cores=U.T@hat.flatten(1);cos=((cores*truthcores).sum(1)/(cores.norm(dim=1)*truthcores.norm(dim=1))).tolist();coefficient=max(abs(coef-rec['original_coefficient_error']),max(abs(a-b) for a,b in zip(cos,rec['original_core_cosines'])))
 values=((phi(native)-truth).norm(dim=0)/den).tolist();score_replay=max(abs(a-b) for a,b in zip(values,rec['per_mode_errors']));assert max(execution,coefficient,score_replay)<1e-8
 row=dict(key=rec['key'],execution_replay=execution,coefficient_replay=coefficient,scalar_replay=score_replay,per_mode_errors=values)
 if rec['removed']:
  delta=z-d['mu'];centered=(delta@V).square()@v-torch.trace(d['old_covariance']@private)
  removed=native.clone();removed[:,5]-=centered
  without=((phi(removed)-truth).norm(dim=0)/den).tolist();other=float((phi(removed)[:,:2]-phi(native)[:,:2]).abs().max());gain=1-values[2]/without[2]
  row.update(without_centered_private_errors=without,third_error_reduction_from_private=gain,other_components_max_change=other,private_coefficient_norm=float((root@private@root/d['scales'][5]).norm()))
  private_forms[rec['key']]=(root@private@root/d['scales'][5]).flatten()
 rows.append(row)
a,b=list(private_forms.values());cosine=float((a@b)/(a.norm()*b.norm()));difference=float((a-b).norm()/a.norm())
selected=next(r for r in rows if r['key']==meta['winners']['16'])
out=dict(checks=rows,private_form_cosine_between_restarts=cosine,private_form_relative_difference=difference,predictions=dict(pred_a_execution=max(max(r['execution_replay'],r['coefficient_replay'],r['scalar_replay']) for r in rows)<1e-8,pred_b_private_use=selected['third_error_reduction_from_private']>=.2 and selected['other_components_max_change']<1e-10,pred_c_private_identity=cosine>=.99),scope='Centered private quadratic removal on opened448sites, others fixed by graph topology; algebraic output selectivity, not semantic selectivity or fresh/OOD identity.')
(P/'SHARED_PRIVATE_DIRECTIONS_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
