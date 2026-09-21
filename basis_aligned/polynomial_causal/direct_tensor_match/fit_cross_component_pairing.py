"""Enumerate15 perfect matchings of six unchanged source outputs at fixed storage.
Subspace and matching selection use only the specified coefficient metric.
"""
from pathlib import Path
import json,time,torch,itertools
from quadratic_pair_blocks import compile_pair,products
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def matchings(nodes):
 if not nodes:return [()]
 first=nodes[0];out=[]
 for i in range(1,len(nodes)):
  for rest in matchings(nodes[1:i]+nodes[i+1:]):out.append(((first,nodes[i]),)+rest)
 return out

def main():
 start=time.perf_counter();plan=json.loads((P/'CROSS_COMPONENT_PAIRING_PLAN_V1.json').read_text());allmatch=matchings(tuple(range(6)));assert len(allmatch)==15 and len(set(allmatch))==15
 for seed in range(5):
  planted=allmatch[seed*3];cost={e:(0 if e in planted else 1) for e in itertools.combinations(range(6),2)};assert min(allmatch,key=lambda m:sum(cost[e] for e in m))==planted
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);native_scale=Q.square().sum((-1,-2)).reshape(3,2).sum(1).sqrt().repeat_interleave(2);baseline=json.loads((P/'COST_MATCHED_PAIR_BASELINES_V1.json').read_text());original=((0,1),(2,3),(4,5));rows=[];programs={};width=plan['width']
 for metric in ['native_isotropic','calibration_shaped']:
  scales=native_scale if metric=='native_isotropic' else d['scales'];inv=torch.eye(1152,dtype=Q.dtype) if metric=='native_isotropic' else d['inverse_root'];T=Q/scales[:,None,None] if metric=='native_isotropic' else d['teacher'];total=T.square().sum();edges={}
  for pair in itertools.combinations(range(6),2):
   matrices=T[list(pair)];_,U=torch.linalg.eigh(sum(t@t for t in matrices));U=U[:,-width:];cores=[U.T@t@U for t in matrices];loss=float((matrices.square().sum()-sum(c.square().sum() for c in cores))/total)
   edges[pair]=dict(loss=loss,basis=inv@U,cores=cores)
  objectives=[dict(pairs=[list(e) for e in m],squared_error=sum(edges[e]['loss'] for e in m)) for m in allmatch];selected=min(allmatch,key=lambda m:sum(edges[e]['loss'] for e in m));old=sum(edges[e]['loss'] for e in original);new=sum(edges[e]['loss'] for e in selected)
  reference=next(r for r in baseline['records'] if r['key']==metric+'_384');referror=reference['native_isotropic_equal_pair_error'] if metric=='native_isotropic' else reference['calibration_shaped_error'];assert abs(old-referror**2)<1e-8
  graph=dict(banks=[],h_readers=torch.stack([pair['a'] for pair in d['pairs']],1),alpha=torch.stack([pair['alpha'] for pair in d['pairs']]),beta=torch.stack([pair['beta'] for pair in d['pairs']]),residual_writer=d['residual_writer'].clone());hats=torch.zeros_like(Q);failure=None
  for pair in selected:
   edge=edges[pair];B=edge['basis'];cores=edge['cores']
   try:c=compile_pair(*cores)
   except (ValueError,RuntimeError) as error:failure=str(error);break
   bank=dict(shared_reader=B@c['input_transform'],product_indices=c['product_indices'],product_weights=c['product_weights']*scales[list(pair)],source_indices=torch.tensor(pair),source_linear=[],source_bias=[])
   for slot,core in zip(pair,cores):
    hats[slot]=B@core@B.T*scales[slot];delta=Q[slot]-hats[slot];bank['source_linear'].append(2*delta@d['mu']);bank['source_bias'].append(torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu'])
   bank['source_linear']=torch.stack(bank['source_linear'],1);bank['source_bias']=torch.stack(bank['source_bias']);graph['banks'].append(bank)
  row=dict(metric=metric,all_matchings=objectives,selected_pairs=[list(e) for e in selected],original_error=old**.5,selected_error=new**.5,relative_error_gain=1-(new/old)**.5,compiler_failure=failure)
  if failure is None:
   ids=d['indices'];z=d['z'][ids];h=d['h'][ids];reads=torch.zeros(len(z),6,dtype=z.dtype)
   for bank in graph['banks']:reads[:,bank['source_indices']]=products(z@bank['shared_reader'],bank['product_indices'])@bank['product_weights']+z@bank['source_linear']+bank['source_bias']
   delta=Q-hats;lin=2*torch.einsum('oij,j->io',delta,d['mu']);bias=torch.einsum('ij,oji->o',d['old_covariance'],delta)-torch.einsum('i,oij,j->o',d['mu'],delta,d['mu']);dense=torch.einsum('ni,oij,nj->no',z,hats,z)+z@lin+bias;replay=float((reads-dense).norm()/dense.norm());assert replay<1e-8
   s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[:,None];phi=((h@graph['h_readers']-.5*reads[:,::2])/s-graph['alpha'])*(reads[:,1::2]/s-graph['beta']);truth=torch.stack([p['truth'][ids] for p in d['pairs']],1);values=((phi-truth).norm(dim=0)/(truth-truth.mean(0)).norm(dim=0)).tolist()
   price=sum(v.numel() for k,v in graph.items() if k!='banks')+sum(v.numel() for bank in graph['banks'] for v in bank.values() if v.is_floating_point());assert price==1340940
   E=hats-Q;native=float((E.square().sum((-1,-2)).reshape(3,2).sum(1)/Q.square().sum((-1,-2)).reshape(3,2).sum(1)).mean().sqrt());covE=torch.einsum('ai,oij,jb->oab',S,E,S)/d['scales'][:,None,None];cov=float(covE.norm()/d['teacher'].norm());computed=native if metric=='native_isotropic' else cov;assert abs(computed-new**.5)<1e-8
   row.update(execution_replay=replay,source_products=3*width,stored_floats=price,index_integers=9*width+6,per_mode_errors=values,native_isotropic_equal_pair_error=native,calibration_shaped_error=cov,ratios_to_original_pair_baseline=[a/b for a,b in zip(values,reference['per_mode_errors'])]);programs[metric]=graph
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='all_matchings'}),flush=True)
 primary=next(r for r in rows if r['metric']==plan['primary_metric']);valid=primary['compiler_failure'] is None
 predictions=dict(pred_a_instrument=valid and primary['execution_replay']<1e-8,pred_b_coefficient_gain=primary['relative_error_gain']>=.1,pred_c_components=valid and max(primary['per_mode_errors'])<=.15 and max(primary['ratios_to_original_pair_baseline'])<=1.1)
 torch.save(programs,P/'CROSS_COMPONENT_PAIRING_V1.pt');(P/'CROSS_COMPONENT_PAIRING_V1.json').write_text(json.dumps(dict(plan=plan,records=rows,predictions=predictions,seconds=time.perf_counter()-start,scope='Same six target forms and three component definitions. Exact matching over15 candidates formed from pair-specific spectral subspaces, not optimal over all possible subspaces/graphs. No evaluation-based matching choice. Opened448notfresh; nativeports retained.'),indent=2)+'\n');print(predictions)
if __name__=='__main__':main()
