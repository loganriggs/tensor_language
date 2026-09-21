"""Independent original-coordinate audit of every partial-graph fit export."""
from pathlib import Path
import torch,json
from shared_mixed_source_graph import source_reads,component_scalars,residual_write
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);before=torch.load(P/'PARTIAL_MIXED_GRAPH_V1.pt',weights_only=True);fit=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());programs=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);records=[]
for row in fit['records']:
 program=programs[row['key']];p=program['shared_mixed'];private_unchanged=all(torch.equal(v,before['private_pair'][k]) for k,v in program['private_pair'].items());assert private_unchanged
 raw=torch.einsum('ir,ro,jr->oij',p['left_reader'],p['product_weights'],p['right_reader']);Qs=.5*(raw+raw.transpose(-1,-2));metric=torch.stack([root@Q@root for Q in Qs])/d['scales'][:4,None,None];coefficient=float((metric-d['teacher'][:4]).norm()/d['teacher'][:4].norm());assert abs(coefficient-row['coefficient_error_first_two'])<1e-8
 dense=torch.einsum('ni,oij,nj->no',d['z'],Qs,d['z'])+d['z']@p['source_linear']+p['source_bias'];computed=source_reads(d['z'],program);replay=float((computed[:,:4]-dense).norm()/dense.norm());assert replay<1e-8
 affine=[];means=[]
 for j,pair in enumerate(d['pairs'][:2]):
  for k,Q in enumerate(pair['Qs']):
   pos=2*j+k;old=2*Q@d['mu'];new=p['source_linear'][:,pos]+2*Qs[pos]@d['mu'];affine.append(float((new-old).norm()/old.norm()))
   original=d['mu']@Q@d['mu']+torch.trace(d['old_covariance']@Q);new=p['source_bias'][pos]+d['mu']@p['source_linear'][:,pos]+d['mu']@Qs[pos]@d['mu']+torch.trace(d['old_covariance']@Qs[pos]);means.append(float(abs(new-original)/(abs(original)+1e-30)))
 phi=component_scalars(d['z'],d['h'],program);errors=[]
 for j,pair in enumerate(d['pairs']):
  truth=pair['truth'][d['indices']];errors.append(float((phi[d['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
 combined=residual_write(d['z'][:32],d['h'][:32],program);parts=sum(residual_write(d['z'][:32],d['h'][:32],program,[i]) for i in range(3));composition=float((combined-parts).norm()/combined.norm());assert composition<1e-10
 rec=dict(key=row['key'],coefficient_error_first_two=coefficient,coefficient_replay=abs(coefficient-row['coefficient_error_first_two']),dense_source_replay=replay,centered_linear_replay=max(affine),centered_mean_replay=max(means),per_mode_errors=errors,scalar_error_replay=max(abs(a-b) for a,b in zip(errors,row['per_mode_errors'])),private_pair_bitwise_unchanged=private_unchanged,residual_composition_replay=composition)
 assert max(rec[k] for k in ['centered_linear_replay','centered_mean_replay','scalar_error_replay'])<1e-8;records.append(rec)
out=dict(records=records,all_checks_pass=True,scope='Dense original-coordinate replay and exact residual-write composition; same opened rows. Not fresh native nonlinear intervention or semantic composition evidence.')
(P/'PROFILED_PARTIAL_GRAPH_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
