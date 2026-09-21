"""Independent native-coordinate export and restart-group audit for output balancing."""
from pathlib import Path
import itertools,json,torch
from scipy.optimize import linear_sum_assignment
from shared_mixed_source_graph import source_reads
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);T=d['teacher'][:4];root=torch.linalg.inv(d['inverse_root']);e,V=torch.linalg.eigh(T.flatten(1)@T.flatten(1).T);V=V.flip(1);truthcores=V.T@T.flatten(1);parent=torch.load(P/'PARTIAL_MIXED_GRAPH_V1.pt',weights_only=True);all_results={}
for family in ['HALF','FULL']:
 result=json.loads((P/f'SOURCE_OUTPUT_BALANCE_{family}_V1.json').read_text());programs=torch.load(P/f'SOURCE_OUTPUT_BALANCE_{family}_PROGRAMS_V1.pt',weights_only=True);checks=[];groups={}
 for row in result['records']:
  p=programs[row['key']];s=p['shared_mixed'];raw=torch.einsum('ir,ro,jr->oij',s['left_reader'],s['product_weights'],s['right_reader']);Qs=(raw+raw.transpose(-1,-2))/2
  M=torch.stack([root@Q@root for Q in Qs])/d['scales'][:4,None,None];coef=float((M-T).norm()/T.norm());cores=V.T@M.flatten(1);cos=(truthcores*cores).sum(1)/(truthcores.norm(dim=1)*cores.norm(dim=1));replay=max(abs(coef-row['original_coefficient_error']),max(abs(float(a)-b) for a,b in zip(cos,row['original_core_cosines'])))
  F=M.flatten(1);ee,U=torch.linalg.eigh(F@F.T);U=U.flip(1);groups[row['key']]=(U,U.T@F)
  z=d['z'][:32];dense=torch.einsum('ni,oij,nj->no',z,Qs,z)+z@s['source_linear']+s['source_bias'];execution=float((dense-source_reads(z,p)[:,:4]).norm()/dense.norm())
  correction=[]
  for o,Q in enumerate([q for pair in d['pairs'][:2] for q in pair['Qs']]):
   target=2*Q@d['mu'];actual=2*Qs[o]@d['mu']+s['source_linear'][:,o];correction.append(float((target-actual).norm()/target.norm()))
   targetmean=d['mu']@Q@d['mu']+torch.trace(d['old_covariance']@Q);actualmean=d['mu']@Qs[o]@d['mu']+torch.trace(d['old_covariance']@Qs[o])+d['mu']@s['source_linear'][:,o]+s['source_bias'][o];correction.append(float(abs(targetmean-actualmean)/(abs(targetmean)+1e-30)))
  unchanged=all(torch.equal(v,parent['private_pair'][k]) for k,v in p['private_pair'].items());assert unchanged and max(replay,execution,max(correction))<1e-8
  checks.append(dict(key=row['key'],original_feature_relative_errors=((truthcores-cores).norm(dim=1)/truthcores.norm(dim=1)).tolist(),coefficient_and_core_replay=replay,execution_replay=execution,centered_affine_mean_replay=max(correction),private_unchanged=unchanged))
 pairs=[]
 for first,second in itertools.combinations(groups,2):
  U,A=groups[first];W,B=groups[second];output=U.T@W;i,j=linear_sum_assignment(-output.abs().numpy());core=A@B.T/(A.norm(dim=1)[:,None]*B.norm(dim=1)[None,:]);pairs.append(dict(first=first,second=second,output_cosines=output[i,j].abs().tolist(),core_cosines=core[i,j].abs().tolist()))
 summary=dict(checks=checks,pairs=pairs,minimum_restart_core_cosine=min(min(r['core_cosines']) for r in pairs),minimum_restart_output_cosine=min(min(r['output_cosines']) for r in pairs));all_results[family]=summary;print(family,{k:v for k,v in summary.items() if k not in ['checks','pairs']})
(P/'OUTPUT_BALANCE_PROGRAM_AUDIT_V1.json').write_text(json.dumps(dict(families=all_results,all_replay_checks_pass=True,scope='Restart stability within each objective, distinct from original-feature fidelity and native behavior. Allfourgroups retained, same teacher/covariance and two independent seeds. Native-coordinate and exact affine/mean checks do not imply causal adoption.'),indent=2)+'\n')
