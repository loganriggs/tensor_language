import json,itertools
from pathlib import Path
import torch
from quartic_cp import cp_gram
P=Path(__file__).resolve().parent;torch.set_num_threads(1);data=torch.load(P/'NATIVE_EXACT_CP_GREEDY_V1.pt',weights_only=True);student=data['students'][8];f=[a.double() for a in student['factors']];C=student['C'].double();K=cp_gram(f,f);corr=K/K.diag().sqrt()[:,None]/K.diag().sqrt()[None,:];within=[]
for atom in range(8):
 v=torch.stack([a[atom] for a in f]);g=v@v.T;within.append(dict(atom=atom,slot_correlations=[float(g[i,j]) for i,j in itertools.combinations(range(4),2)],coefficient_atom_squared_norm=float(K[atom,atom])))
v=torch.cat(f);g=v@v.T;pairs=[dict(slot_a=i//8,atom_a=i%8,slot_b=j//8,atom_b=j%8,abs_correlation=abs(float(g[i,j]))) for i,j in itertools.combinations(range(32),2) if i%8!=j%8];pairs=sorted(pairs,key=lambda r:-r['abs_correlation']);out=dict(within_atoms=within,top_cross_atom_factor_pairs=pairs[:12],factor_pair_count_above_099=sum(r['abs_correlation']>.99 for r in pairs),quartic_atom_correlation=corr.tolist(),feature_gram_condition=float(torch.linalg.cond(K)),writer_singular_values=torch.linalg.svdvals(C).tolist(),scope='Post-fit descriptive structure for8 native CP atoms. Correlated factors do not establish shared computation, stable identity or useful approximation.')
(P/'NATIVE_CP_STRUCTURE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print('condition',out['feature_gram_condition'],'max cross atomfactor',pairs[0],'>.99',out['factor_pair_count_above_099']);print('within',[(r['atom'],r['coefficient_atom_squared_norm'],r['slot_correlations']) for r in within])
