"""Necessary native-coefficient rank bounds; not achievability or DAG lower bounds."""
from pathlib import Path
import json,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);Q=torch.stack([q for p in d['pairs'] for q in p['Qs']]);energy=Q.square().sum((-1,-2)).reshape(3,2).sum(1);T=Q/energy.repeat_interleave(2).sqrt()[:,None,None]
eigen=torch.linalg.eigvalsh(T).square().sort(descending=True).values
rows=[]
for products,private in [(383,0),(367,32),(512,0),(560,32)]:
 ranks=[min(1152,2*products)]*5+[min(1152,2*products+private)];individual=sum(eigen[o,ranks[o]:].sum() for o in range(6))/T.square().sum()
 width=min(1152,2*products+private);S=torch.einsum('oij,okj->ik',T,T);vals=torch.linalg.eigvalsh(S).clamp_min(0).sort(descending=True).values
 shared=vals[width:].sum()/T.square().sum()
 rows.append(dict(shared_mixed_products=products,private_squares=private,common_input_span_max_width=width,individual_output_rank_caps=ranks,native_equal_pair_error_lower_bound_individual_ranks=float(individual.sqrt()),native_equal_pair_error_lower_bound_shared_span=float(shared.sqrt())))
out=dict(records=rows,scope='Necessary spectral bounds for arbitrary dictionaries at stated topology. Each mixed product has symmetric-matrix rank<=2 and each private square rank<=1. Every output column space lies in the common reader span. Relaxations ignore simultaneous atom sharing constraints and do not prove attainability, general polynomial DAG limits, or behavioral fidelity.')
(P/'SOURCE_CAPACITY_BOUNDS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
