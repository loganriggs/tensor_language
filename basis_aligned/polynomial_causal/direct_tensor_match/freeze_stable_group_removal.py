"""Freeze aligned split-fit groups and unused documents before native evaluation."""
from pathlib import Path
import torch,json,hashlib
p=Path(__file__).resolve().parent;root=p.parents[2];torch.set_num_threads(2)
out=p/'MIDPOINT_STABLE_GROUP_REMOVAL_PROGRAMS_V1.pt';assert not out.exists()
s=torch.load(p/'MIDPOINT_JOINT_METRIC_STABILITY_V1.pt',weights_only=True);q=torch.load(p/'MIDPOINT_PRIVATE_FROZEN_V1.pt',weights_only=True)['private512'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
mean_n=rows['n'].double().mean((0,1));mean_m=rows['m'].double().mean((0,1));ids=s['ids'];audit=json.loads((p/'MIDPOINT_METRIC_STABLE_GROUPS_V1.json').read_text());exports={}
for label in ['half0','half1']:
 a,b,w=[v[:,ids]@c for v,c in zip([q['A'],q['B'],q['reduced_writers']],s['programs'][label]['coefficients'])]
 if label=='half1':
  perm=audit['alignment_permutation'];a=a[:,perm];b=b[:,perm];w=w[:,perm]
 exports[label]=dict(A=a,B=b,W=w,left_mean=mean_n@a,right_mean=mean_m@b)
torch.save(exports,out)
cache=root/'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt';tokens=torch.load(cache,weights_only=True)[96:112,:257].clone()
prior=[torch.load(x,weights_only=True) for x in p.glob('*CONFIRMATION_FINEWEB_V1.pt')];prior.append(rows['tokens'])
for row in tokens:
 assert not any(torch.equal(row[:min(len(row),len(other))],other[:min(len(row),len(other))]) for panel in prior for other in panel)
torch.save(tokens,p/'MIDPOINT_STABLE_GROUP_REMOVAL_TOKENS_V1.pt')
meta=dict(groups={f'group{i}':v['components'] for i,v in enumerate(audit['maximum_stable_partition'])},individual_controls={'individual1':[1],'individual4':[4]},program_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),source_cache=str(cache.relative_to(root)),documents=list(range(96,112)),context=256,positions='16:256',discovery_threshold=.1,registered_native_threshold=.1,scope='Frozen five groups from conditional split stability. New documents for this decomposition study, no pretraining-overlap claim. Compare removal effects across two decompositions, not against a uniquely defined native group or semantic task.')
(p/'MIDPOINT_STABLE_GROUP_REMOVAL_PLAN_V1.json').write_text(json.dumps(meta,indent=2)+'\n')
