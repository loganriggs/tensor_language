"""Paired-document uncertainty and exact strength/direction error partition."""
from pathlib import Path
import json,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2)
r=json.loads((p/'MIDPOINT_STABLE_GROUP_REMOVAL_NATIVE_V1.json').read_text());records=[];gen=torch.Generator().manual_seed(261331);draw=torch.randint(16,(10000,16),generator=gen)
for group in r['summary']:
 rows=[v for v in r['records'] if v['group']==group];assert [v['document'] for v in rows]==list(range(96,112))
 a,b,err=[torch.tensor([v[k] for v in rows],dtype=torch.float64) for k in ['effect_energy_half0','effect_energy_half1','effect_discrepancy_energy']]
 A=float(a.sum());B=float(b.sum());E=float(err.sum());dot=(A+B-E)/2;cos=dot/(A*B)**.5
 strength=(A**.5-B**.5)**2;direction=2*(A*B)**.5*(1-cos);assert abs(strength+direction-E)/E<1e-10
 sample=(err[draw].sum(1)/torch.minimum(a[draw].sum(1),b[draw].sum(1))).sqrt()
 records.append(dict(group=group,relative_effect_discrepancy=(E/min(A,B))**.5,descriptive95interval=torch.quantile(sample,torch.tensor([.025,.975],dtype=torch.float64)).tolist(),aggregate_effect_cosine=cos,effect_norm_half1_over_half0=(B/A)**.5,strength_fraction_of_squared_discrepancy=strength/E,direction_fraction_of_squared_discrepancy=direction/E))
out=p/'MIDPOINT_STABLE_GROUP_REMOVAL_AUDIT_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,bootstrap_draws=10000,scope='Paired bootstrap over16newdocuments, fixed programs/groups; descriptive conditional intervals, no familywise guarantee. Algebraic norm/direction partition fits no rescaling and leaves registered failures unchanged.'),indent=2)+'\n');print(out.read_text())
