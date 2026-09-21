"""Original-weight two-source quadratic baseline without approximation fitting."""
from pathlib import Path
import torch,json,time
from quadratic_pair_blocks import compile_pair,products
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter();fold=torch.load(p/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.pt',weights_only=True);mode=torch.load(p/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True);data=torch.load(p/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);A=fold['a']['matrix'];B=fold['b']['matrix'];out=p/'MIDPOINT_ORIGINAL_SOURCE_BLOCK_V1.json'
attempts=[]
try:
 c=compile_pair(A,B);attempts.append(dict(method='raw_pencil',status='PASS'))
except ValueError as error:
 attempts.append(dict(method='raw_pencil',status='REJECTED',reason=str(error)))
 # Exact signed whitening changes coordinates, not either target form.
 ev,U=torch.linalg.eigh(A+B);conditioning=(U*ev.abs().rsqrt())
 try:
  inner=compile_pair(conditioning.T@A@conditioning,conditioning.T@B@conditioning)
  transform=torch.linalg.solve(conditioning.T,inner['input_transform']);indices=inner['product_indices'];weights=inner['product_weights'];errors=[]
  for j,Q in enumerate([A,B]):
   core=torch.zeros_like(Q)
   for position in range(weights.shape[0]):
    i,k,kind=indices[:,position].tolist();w=weights[position,j]
    if kind==0:core[i,i]+=w
    elif kind==1:core[i,i]+=w;core[k,k]-=w
    else:core[i,k]+=w/2;core[k,i]+=w/2
   errors.append(float((Q-transform@core@transform.T).norm()/Q.norm()))
  if max(errors)>1e-10:raise ValueError(f'Original-coordinate reconstruction failed: {errors}')
  c={**inner,'input_transform':transform,'diagnostics':{**inner['diagnostics'],'whitening_condition':float(torch.linalg.cond(conditioning)),'matrix_replay':errors}}
  attempts.append(dict(method='signed_whitening',status='PASS'))
 except ValueError as error:
  attempts.append(dict(method='signed_whitening',status='REJECTED',reason=str(error)));result=dict(status='REJECTED',attempts=attempts,seconds=time.perf_counter()-start);out.write_text(json.dumps(result,indent=2)+'\n');print(result);raise SystemExit(0)
z=data['z'].flatten(0,1).double();pred=products(z@c['input_transform'],c['product_indices'])@c['product_weights'];truth=torch.stack([((z@Q)*z).sum(1) for Q in [A,B]],1);replay=float((pred-truth).norm()/truth.norm());assert replay<1e-10
program=dict(shared_reader=c['input_transform'],product_indices=c['product_indices'],product_weights=c['product_weights'],h_reader=mode['A'][:,0].clone(),residual_writer=torch.linalg.solve(mode['R_U'],mode['writer']),alpha=mode['mean_n']@mode['A'][:,0],beta=mode['mean_m']@mode['B'][:,0])
# The homogeneous source forms require no affine parameters.
floats=sum(t.numel() for t in program.values() if t.is_floating_point());indices=program['product_indices'].numel();torch.save(program,p/'MIDPOINT_ORIGINAL_SOURCE_BLOCK_V1.pt');result=dict(status='COMPILED',attempts=attempts,predictions=dict(pred_a_exact=max(c['diagnostics']['matrix_replay']+[replay])<1e-10,pred_b_products=len(c['product_weights'])<=1152,pred_c_price=floats<10628354),diagnostics=c['diagnostics'],calibration_source_replay=replay,source_products=len(c['product_weights']),native_source_products=4608,stored_float_scalars=floats,stored_integer_indices=indices,native_two_source_scalar_program_weights=10628354,real_blocks=sum(len(b)==1 for b in c['blocks']),complex_blocks=sum(len(b)==2 for b in c['blocks']),seconds=time.perf_counter()-start,scope='OriginalQa/Qb weight forms, no fit or covariance approximation. Source-only native comparison uses L16/R16 shared4608products plus2channel readers, h-reader/writer and2centers. Zero affine fields omitted explicitly. Native z/h producers and normalization remain outside cost; not wholemodel saving.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
