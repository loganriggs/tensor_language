"""Physically remove products changed by the selective rotation."""
from pathlib import Path
import torch,json
from source_graph_metrics import export,score
from shared_quadratic_products import materialize_mixed
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());parent=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[meta['winner']];pairs=torch.load(P/'SELECTIVE_PRODUCT_ROTATION_PAIRS_V1.pt',weights_only=True)['selected_pairs'];mask=torch.ones(256,dtype=torch.bool)
for i,j in pairs:mask[i]=False;mask[j]=False
root=torch.linalg.inv(d['inverse_root']);s=parent['shared_mixed'];L=root@s['left_reader'];R=root@s['right_reader'];W=s['product_weights'].T/d['scales'][:4,None];original=materialize_mixed(L,R,W)
new=export(L[:,mask],R[:,mask],W[:,mask],d,parent);hat=materialize_mixed(L[:,mask],R[:,mask],W[:,mask]);change=float((hat-original).norm()/original.norm());rotation=json.loads((P/'SELECTIVE_PRODUCT_ROTATIONS_V1.json').read_text())['result']['relative_function_change'];result=score(new,d);base=json.loads((P/'MULTIMODE_PROJECTION_SHARING_V1.json').read_text())['baseline_errors']
floats=new['residual_writer'].numel()+sum(v.numel() for key in ['shared_mixed','private_pair'] for v in new[key].values() if v.is_floating_point())
out=dict(shared_products=int(mask.sum()),total_source_products=int(mask.sum())+256,stored_floats=floats,relative_function_change=change,ratio_to_rotation=change/rotation,**result,predictions=dict(pred_a_necessity=change>10*rotation,pred_b_fidelity=all(a<=.15 and a<=1.1*b for a,b in zip(result['per_mode_errors'],base))),scope='Opened-state deletion control selected from coefficient-space rotation mask. Exact original centered affine/mean corrections retained. Physical column slicing counts literal products and coefficients; no refit and no fresh behavioral claim.')
(P/'ROTATION_DELETION_CONTROL_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
