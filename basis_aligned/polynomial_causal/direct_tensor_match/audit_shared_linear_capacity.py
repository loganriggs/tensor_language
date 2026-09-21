"""Necessary span rank versus literal cost for the shared-linear-node edit.
Uses target forms, not fitted reader SVD. Not a bound on sparse/general DAGs.
"""
from pathlib import Path
import json,torch,math
from shared_linear_source_graph import expand,source_reads,arithmetic
from compact_source_graph import source_reads as dense_reads
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);meta=json.loads((P/'SHARED_LINEAR_GRAPH_V1.json').read_text());programs=torch.load(P/'SHARED_LINEAR_GRAPH_PROGRAMS_V1.pt',weights_only=True)
# Every rank-r common-reader program has all quadratic ranges in an r-dimensional span.
# One-mode unfolding tail gives a necessary (not sufficient) squared coefficient error.
bounds={}
for label,forms in [('native',Q),('covariance',S@Q@S)]:
 energy=forms.square().sum((-1,-2)).reshape(3,2).sum(1);T=forms/energy.repeat_interleave(2).sqrt()[:,None,None]
 eig=torch.linalg.eigvalsh(torch.einsum('oij,okj->ik',T,T)).clamp_min(0)
 tail=torch.cat([eig.cumsum(0).flip(0),eig.new_zeros(1)]) / T.square().sum()
 bounds[label]=tail.sqrt()
assert all(p['left_map'].shape[1]==560 and p['right_map'].shape[1]==560 and p['square_map'].shape[1]==32 and p['input_basis'].shape[0]==1152 for p in programs.values())
rows=[]
for label,base in meta['parents'].items():
 minimums={key:int(torch.where(b<=1.1*base[key+'_error'])[0][0]) for key,b in bounds.items()}
 required=max(minimums.values());maxrank=math.floor((.8*base['source_total_multiplications']-3984)/2304)
 rows.append(dict(parent=label,minimum_necessary_rank_by_metric=minimums,minimum_necessary_joint_rank=required,maximum_rank_for_20percent_multiplication_saving=maxrank,necessary_minimum_source_multiplications=2304*required+3984,twenty_percent_target_impossible_from_span_bound=required>maxrank,bounds_at_primary_rank448={k:float(b[448]) for k,b in bounds.items()},bounds_at_maximum_saving_rank={k:float(b[maxrank]) for k,b in bounds.items()}))
checks=[]
for row in meta['records']:
 p=programs[row['key']];dense=expand(p);ids=d['indices'];z=d['z'][ids];a=source_reads(z,p);b=dense_reads(z,dense);error=float((a-b).norm()/b.norm());assert error<1e-8
 for key,bound in bounds.items():assert row[key+'_error']>=float(bound[row['rank']])-1e-10
 assert arithmetic(p)['source_total_multiplications']==row['source_total_multiplications']
 checks.append(dict(key=row['key'],execution_replay=error,metric_lower_bound_consistent=True))
out=dict(rows=rows,independent_saved_graph_checks=checks,lower_bound_at_ranks={str(r):{k:float(b[r]) for k,b in bounds.items()} for r in [384,448,460,512,576,766,1024]},scope='Necessary one-mode unfolding bound for any rank-r shared linear input bottleneck, even allowing an arbitrary unpriced dense quadratic core. Total multiplication comparison assumes the current 560mixed+32square graph with dense two-stage reader matrices. Not a general circuit lower bound and not a functional-error bound. No new tests or calibration inputs.')
(P/'SHARED_LINEAR_CAPACITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
