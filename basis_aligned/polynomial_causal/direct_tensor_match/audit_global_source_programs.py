"""Independent decode and within-objective restart comparison of global source graphs."""
from pathlib import Path
import json,torch
from scipy.optimize import linear_sum_assignment
from global_mixed_source_graph import source_reads
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
result=json.loads((P/'GLOBAL_SOURCE_BALANCE_V1.json').read_text());programs=torch.load(P/'GLOBAL_SOURCE_BALANCE_PROGRAMS_V1.pt',weights_only=True)
root=torch.linalg.inv(d['inverse_root']);true=d['teacher'];e,U=torch.linalg.eigh(true.flatten(1)@true.flatten(1).T);U=U.flip(1)
truthcores=torch.einsum('oa,oij->aij',U,true);checks=[];groups={}
for rec in result['records']:
 p=programs[rec['key']];raw=torch.einsum('ir,ro,jr->oij',p['left_reader'],p['product_weights'],p['right_reader']);Q=(raw+raw.transpose(-1,-2))/2
 T=torch.einsum('ai,oij,jb->oab',root,Q,root)/d['scales'][:,None,None]
 err=float((T-true).norm()/true.norm());core=torch.einsum('oa,oij->aij',U,T)
 cos=((core*truthcores).sum((-1,-2))/(core.norm(dim=(-1,-2))*truthcores.norm(dim=(-1,-2)))).tolist()
 replay=max(abs(err-rec['original_coefficient_error']),max(abs(a-b) for a,b in zip(cos,rec['original_core_cosines'])))
 z=d['z'][:32];dense=torch.einsum('ni,oij,nj->no',z,Q,z)+z@p['source_linear']+p['source_bias'];execution=float((dense-source_reads(z,p)).norm()/dense.norm())
 affine=[]
 for o,orig in enumerate([q for pair in d['pairs'] for q in pair['Qs']]):
  delta=orig-Q[o];lin=2*delta@d['mu'];bias=torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu']
  affine.extend([float((lin-p['source_linear'][:,o]).abs().max()),float((bias-p['source_bias'][o]).abs())])
 ev,V=torch.linalg.eigh(T.flatten(1)@T.flatten(1).T);V=V.flip(1);C=torch.einsum('oa,oij->aij',V,T).flatten(1)
 groups[rec['key']]=(V,C)
 checks.append(dict(key=rec['key'],coefficient_core_replay=replay,execution_replay=execution,affine_mean_maxabs=max(affine),stored_floats=sum(v.numel() for v in p.values())))
pairs=[]
for power in result['plan']['powers']:
 keys=[r['key'] for r in result['records'] if r['power']==power];(A,X),(B,Y)=[groups[k] for k in keys]
 row,col=linear_sum_assignment(-(A.T@B).abs().numpy());sign=(A[:,row]*B[:,col]).sum(0).sign()
 corecos=((X[row]*Y[col]).sum(1)*sign/(X[row].norm(dim=1)*Y[col].norm(dim=1))).tolist()
 pairs.append(dict(power=power,keys=keys,core_cosines=corecos,minimum_core_cosine=min(corecos)))
assert all(max(c[k] for k in ['coefficient_core_replay','execution_replay','affine_mean_maxabs'])<1e-8 for c in checks)
out=dict(checks=checks,restart_pairs=pairs,all_replay_checks_pass=True,scope='Independent native-factor decode and whole-group restart agreement; no new model forwards or semantic identity.')
(P/'GLOBAL_SOURCE_PROGRAM_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
