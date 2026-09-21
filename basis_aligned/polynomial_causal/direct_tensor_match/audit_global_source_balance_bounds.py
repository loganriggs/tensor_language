"""All-six-output common-input rank bounds and literal global graph price."""
from pathlib import Path
import torch,json
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);T=d['teacher'];e,U=torch.linalg.eigh(T.flatten(1)@T.flatten(1).T);rows=[]
for power in [0,.5,1]:
 S=torch.einsum('oa,oij->aij',U,T)*e.pow(-power/2)[:,None,None];K=sum(M@M for M in S);values=torch.linalg.eigvalsh(K).clamp_min(0).flip(0)
 for rank in [512,766,768]:rows.append(dict(output_whitening_power=power,maximum_input_rank=rank,relative_coefficient_lower_bound=float((values[rank:].sum()/values.sum()).sqrt())))
out=dict(records=rows,proposed_shared_products=383,proposed_input_directions=766,proposed_stored_float_coefficients=2*1152*383+6*383+1152*6+6+1152*3+6+1152,reference_partial_graph_floats=897804,scope='Necessary right-unfolding rank bound for all6sourceoutputs. Fullglobalmixed383 wouldfitallthreecomponentswithoutprivatebranch. Bounds differ in metric and need not be attainable; no fit or behavioral claim.')
(P/'GLOBAL_SOURCE_BALANCE_BOUNDS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
