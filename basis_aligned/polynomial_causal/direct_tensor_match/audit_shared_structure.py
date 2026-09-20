"""Check whether shared computation topology/factors are more stable than individual atoms."""
import itertools,json
from pathlib import Path
import torch
from core import metric
P=Path(__file__).resolve().parent;torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
sweep=json.load(open(P/'QUARTIC_BASIS_SWEEP_V1.json'));pairs=list(itertools.combinations_with_replacement(range(3),2));centers=[]
for r in sweep['records']:
 if r['target']!='planted_shared':continue
 refit=next(x for x in r['refits'] if x['products']==3);edges=[pairs[i] for i in refit['support']];common=set(edges[0])
 for e in edges:common&=set(e)
 assert len(common)==1
 i=next(iter(common));centers.append(torch.tensor(r['bank'][i]))
M=metric(6,2);correlations=[float(abs(a@M@b)/((a@M@a)*(b@M@b)).sqrt()) for a,b in itertools.combinations(centers,2)]
x=json.load(open(P/'QUARTIC_CONTEXT_BASIS_V1.json'));pairs=list(itertools.combinations_with_replacement(range(4),2));topologies=[]
for c in x['records']:
 edges=[{pairs[i] for i in r['refits'][1]['support']} for r in c['runs']]
 isomorphic=any({tuple(sorted((p[a],p[b]))) for a,b in edges[0]}==edges[1] for p in itertools.permutations(range(4)))
 topologies.append(dict(case_index=c['case_index'],same_support_graph_up_to_permutation=isomorphic))
out=dict(planted_centers=len(centers),minimum_common_factor_correlation=min(correlations),native_topology_matches=topologies,scope='Post-hoc structural diagnostic: stable planted common factor across12 basis runs, native graph isomorphism across two same-span runs. Graph agreement does not establish matching feature functions or semantic circuit identity.')
(P/'SHARED_STRUCTURE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print('common factor min correlation',min(correlations));print('native graph matches',sum(r['same_support_graph_up_to_permutation'] for r in topologies),'of',len(topologies))
