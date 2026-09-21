"""Shared-input Tucker source pair, then exact shared-product core compilation.
Width128/256, plain versus affine-weighted subspace proposal. Fixed old affine
branches. Primary affine256, compared with independent128 at256 source products.
"""
from pathlib import Path
import torch,json,time
from quadratic_pair_blocks import compile_pair,products
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
data=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);c=data['common']
geometry=torch.load(P/'EXPANDED_SOURCE_METRIC_GEOMETRY_V1.pt',weights_only=True);root=geometry['root'];inv=geometry['inverse_root']
oldroot=torch.linalg.inv(c['source_inverse_root']);oldcov=oldroot@oldroot
cal=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);extra=torch.load(P/'EXPANDED_COVARIANCE_STATES_V1.pt',weights_only=True)
z=cal['z'].flatten(0,1).double();delta=z-c['mu'];newz=extra['z'].flatten(0,1).double()
train_delta=torch.cat([delta[:24*64],newz-c['mu']]);t=torch.cat([c['t'][:24*64],extra['t'].flatten()]);s=torch.cat([c['scale'][:24*64],extra['scale'].flatten()])
la=-.5*(train_delta@c['sources']['a']['linear'])+t-c['t'][:24*64].mean()-c['alpha']*(s-c['scale'][:24*64].mean())
lb=train_delta@c['sources']['b']['linear']-c['beta']*(s-c['scale'][:24*64].mean())
K=torch.tensor([[.25*lb.square().mean(),-.5*(la*lb).mean()],[-.5*(la*lb).mean(),la.square().mean()]],dtype=torch.float64)
mode=torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True)
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();Down=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'][0].double()
Ms=[]
for reader in [c['h_reader'],mode['B'][:,2]]:
 raw=L.T@((lam*(Down.T@reader))[:,None]*R);Q=(raw+raw.T)/2;Ms.append(root@Q@root)
A,B=Ms;records=[];exports={}
indices=torch.tensor([j for i in [24,25,26,27,29,30,31] for j in range(i*64,(i+1)*64)])
for kind,weights in [('plain',torch.eye(2,dtype=torch.float64)),('affine',K/K.trace())]:
 gram=weights[0,0]*(A@A)+weights[1,1]*(B@B)+weights[0,1]*(A@B+B@A);gram=(gram+gram.T)/2
 ev,V=torch.linalg.eigh(gram);assert ev.min()>-1e-10*ev.max();V=V[:,ev.argsort(descending=True)]
 for width in [128,256]:
  basis=V[:,:width];shared=inv@basis;cores=[basis.T@M@basis for M in Ms]
  try:
   compiled=compile_pair(*cores)
  except ValueError as error:
   records.append(dict(kind=kind,width=width,status='COMPILER_REJECTED',reason=str(error)));continue
  program={k:c[k].clone() for k in ['h_reader','residual_writer','alpha','beta']}
  program.update(shared_reader=shared@compiled['input_transform'],product_indices=compiled['product_indices'],product_weights=compiled['product_weights'])
  mean=c['mu']@shared;oldcorecov=shared.T@oldcov@shared
  direct=[]
  for k,core in zip(['a','b'],cores):
   source=c['sources'][k];program[k+'_linear']=source['linear']-2*shared@(core@mean)
   program[k+'_bias']=source['constant']-c['mu']@source['linear']+mean@core@mean-torch.trace(oldcorecov@core)
   coords=delta@shared
   direct.append(source['constant']+delta@source['linear']+((coords@core)*coords).sum(1)-torch.trace(oldcorecov@core))
  values=products(z@program['shared_reader'],program['product_indices'])@program['product_weights']
  reads=[program[k+'_bias']+z@program[k+'_linear']+values[:,j] for j,k in enumerate(['a','b'])]
  replay=max(float((a-b).norm()/b.norm()) for a,b in zip(reads,direct));assert replay<1e-8
  phi=((c['t']-.5*reads[0])/c['scale']-c['alpha'])*(reads[1]/c['scale']-c['beta'])
  errors={}
  for name,sl in [('old_training',slice(0,24*64)),('opened_distinct',indices)]:
   truth=c['true_phi'][sl];errors[name+'_normalized_error']=float((phi[sl]-truth).norm()/(truth-truth.mean()).norm())
  row=dict(kind=kind,width=width,status='COMPILED',source_products=len(compiled['product_weights']),stored_float_scalars=sum(v.numel() for v in program.values() if v.is_floating_point()),stored_integer_indices=program['product_indices'].numel(),matrix_replay=compiled['diagnostics']['matrix_replay'],source_program_replay=replay,**errors)
  records.append(row);exports[f'{kind}_{width}']=program;print(row,flush=True)
primary=next(r for r in records if r['kind']=='affine' and r['width']==256)
predictions=dict(pred_a_compiler=all(r['status']=='COMPILED' and r['source_program_replay']<1e-8 for r in records),pred_b_primary=primary['status']=='COMPILED' and primary['opened_distinct_normalized_error']<=min(.15,1.1*.16343149968132947),pred_c_products=primary['status']=='COMPILED' and primary['source_products']<=256)
out=dict(predictions=predictions,records=records,affine_source_gram=K.tolist(),independent128_error=.16343149968132947,independent128_products=256,independent128_scalars=299780,seconds=time.perf_counter()-start,scope='Two-stage fixed-weight construction: commoninput Tucker projection of originalsourceforms, exact shared-product core compilation. Sameoldaffinebranches and expandedtraininggeometry. Seven opened evaluation prefixes, no outcome optimization, fresh/nativeintervention evidence pending.')
(P/'SHARED_EXPANDED_SOURCE_FIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(exports,P/'SHARED_EXPANDED_SOURCE_PROGRAMS_V1.pt')
print(predictions)
