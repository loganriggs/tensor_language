"""Exact fixed-span projection and executable pair compilation as a structural control."""
import json,time
from pathlib import Path
import torch
from pairwise_reader_graph import GROUPS,expand
from quadratic_pair_blocks import compile_pair
from local_shared_reader_graph import decode
from pairwise_graph_assessment import Assessment,global_form
from global_mixed_source_graph import score
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
audit=Assessment(d)
params=torch.load(P/'PENCIL_JOINT_REFIT_STATES_V1.pt',weights_only=True)
graphs=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)
base=json.loads((P/'PRIVATE_CAPACITY_PLAN_V1.json').read_text())['baseline']
rows=[]
for geometry in ('calibration_shaped','native_isotropic'):
 key=geometry+'_inherited';raw=params[key];bundle=expand(graphs[key]);A,inv=(audit.S,d['inverse_root']) if geometry=='calibration_shaped' else (audit.I,audit.I)
 target=A@audit.Q@A;forms=[];diagnostics=[];failures=[];widths=[]
 for j,(a,b) in enumerate(GROUPS):
  V=torch.cat([raw[a],raw[b],raw[3+j]],1);U=torch.linalg.qr(V,mode='reduced').Q
  core=U.T@target[2*j:2*j+2]@U;H=inv@U@core@U.T@inv.T;forms.extend(H);widths.append(U.shape[1])
  try:
   compiled=compile_pair(core[0],core[1]);item=bundle[str(j)];item['shared_reader']=inv@U@compiled['input_transform'];item['product_indices']=compiled['product_indices'];item['product_weights']=compiled['product_weights']
   for name,true,hat in zip(('a','b'),audit.Q[2*j:2*j+2],H):
    delta=true-hat;item[name+'_linear']=2*delta@d['mu'];item[name+'_bias']=torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu']
   assert float((decode(item)-H).norm()/H.norm())<1e-8
   diagnostics.append(compiled['diagnostics'])
  except (AssertionError,ValueError,RuntimeError) as failure:
   failures.append(dict(pair=j,message=str(failure)))
 H=torch.stack(forms);errors=[]
 for metric in (audit.I,audit.S):
  T=metric@audit.Q@metric;E=metric@(H-audit.Q)@metric;errors.append(float((E.square().sum((-1,-2)).reshape(3,2).sum(1)/T.square().sum((-1,-2)).reshape(3,2).sum(1)).mean().sqrt()))
 row=dict(geometry=geometry,native_error=errors[0],covariance_error=errors[1],widths=widths,compiler=diagnostics,failures=failures)
 if not failures:
  g=global_form(bundle);row.update(score(g,d));products=sum(v['product_weights'].shape[0] for v in bundle.values());projections=sum(v['shared_reader'].numel() for v in bundle.values());row.update(activation_products=products,projection_multiplications=projections,source_total_multiplications=projections+3*products)
  row['coefficient_pass']=all(row[k]<=1.1*base[k] for k in ('native_error','covariance_error'))
  row['component_pass']=all(a<=min(.15,1.1*b) for a,b in zip(row['per_mode_errors'],base['per_mode_errors']))
  row['jacobian_pass']=all(a<=1.1*b for a,b in zip(row['euclidean_jacobian_errors'],base['euclidean_jacobian_errors']))
  row['original_arithmetic_pass']=row['source_total_multiplications']<=.8*1330560
 rows.append(row)
out=dict(records=rows,seconds=time.monotonic()-start,scope='Diagnostic unrestricted core in final inherited input spans, followed by independent pair compilation. Shared across-pair projections are intentionally not retained in this concrete executor; all dense projections are charged. No fitting, new data or adoption. Mean/affine corrections match prior assessment. Same component and dual coefficient requirements; original20% source-saving requirement retained.')
(P/'FULL_CORE_CLOSURE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
