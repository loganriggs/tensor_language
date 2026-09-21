"""Change only source weighting sample size; keep old center/affine terms.
New weighting is second moment about the fixed old training mean, not covariance
about a newly fitted mean. Primary rank64, ranks16/128 diagnostic controls.
"""
import torch,json,time
from pathlib import Path
from gaussian_quartic_lowrank import prepare_teacher,squared_error_by_degree
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
receipt=json.loads((P/'EXPANDED_COVARIANCE_CAPTURE_V1.json').read_text());assert all(receipt['predictions'].values())
data=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);c=data['common'];f=data['frames']['covariance']
extra=torch.load(P/'EXPANDED_COVARIANCE_STATES_V1.pt',weights_only=True)
old=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True)
z=old['z'].flatten(0,1).double();newz=extra['z'].flatten(0,1).double();delta=z-c['mu'];newdelta=newz-c['mu']
train=torch.cat([delta[:24*64],newdelta])
weight=train.T@train/len(train);ev,Q=torch.linalg.eigh(weight);assert ev.min()>0
root=(Q*ev.sqrt())@Q.T;inv=(Q*ev.rsqrt())@Q.T
oldroot=torch.linalg.inv(c['source_inverse_root'])
indices=torch.tensor([j for i in [24,25,26,27,29,30,31] for j in range(i*64,(i+1)*64)])
mode=torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True)
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();Down=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'][0].double()
spectra={}
for k,reader in [('a',c['h_reader']),('b',mode['B'][:,2])]:
 w=lam*(Down.T@reader);raw=L.T@(w[:,None]*R);matrix=(raw+raw.T)/2
 values,vectors=torch.linalg.eigh(root@matrix@root);ids=values.abs().argsort(descending=True)
 spectra[k]=(values[ids],vectors[:,ids])
teacher=prepare_teacher(f['A'],f['B']);rows=[];programs={}
for rank in [16,64,128]:
 reads=[];newreads=[];factors=[]
 program={k:c[k].clone() for k in ['h_reader','residual_writer','alpha','beta']}
 for j,(k,factor) in enumerate([('a',-.5),('b',1.)]):
  values,vectors=spectra[k];proj=inv@(vectors[:,:rank]*values[:rank].abs().sqrt());sign=values[:rank].sign()
  U=oldroot@proj;trace=(U.square().sum(0)*sign).sum()
  source=c['sources'][k]
  reads.append(source['constant']+delta@source['linear']+((delta@proj).square()*sign).sum(1)-trace)
  newreads.append(source['constant']+newdelta@source['linear']+((newdelta@proj).square()*sign).sum(1)-trace)
  X=torch.cat([f['J']@U,f[k+'_fixed_vectors'],f['unit'][:,None]],1)
  weights=torch.cat([factor*sign,f[k+'_fixed_values'],(-factor*trace).reshape(1)]);factors.extend([X,weights])
  pm=c['mu']@proj
  program[k+'_reader']=proj;program[k+'_eigenvalues']=sign
  program[k+'_linear']=source['linear']-2*proj@(sign*pm)
  program[k+'_bias']=source['constant']-c['mu']@source['linear']+(sign*pm.square()).sum()-trace
 phi=((c['t']-.5*reads[0])/c['scale']-c['alpha'])*(reads[1]/c['scale']-c['beta'])
 scale=extra['scale'].flatten();t=extra['t'].flatten()
 newphi=((t-.5*newreads[0])/scale-c['alpha'])*(newreads[1]/scale-c['beta'])
 true=c['true_phi'][indices]
 error=float((phi[indices]-true).norm()/(true-true.mean()).norm())
 training_true=torch.cat([c['true_phi'][:24*64],extra['native_phi'].flatten()])
 training_pred=torch.cat([phi[:24*64],newphi])
 training_error=float((training_pred-training_true).norm()/(training_true-training_true.mean()).norm())
 gaussian=float((squared_error_by_degree(teacher,*factors).sum()/teacher['variance']).clamp_min(0).sqrt())
 # Independent exported affine/quadratic execution at all old states.
 from source_interface import source_read
 export_reads=[source_read(z,program,k) for k in ['a','b']]
 replay=max(float((u-v).norm()/v.norm()) for u,v in zip(export_reads,reads));assert replay<1e-8
 row=dict(rank_per_source=rank,source_products=2*rank,stored_scalars=sum(v.numel() for v in program.values()),old24_gaussian_error=gaussian,expanded_training_error=training_error,opened_distinct_normalized_error=error,source_export_replay=replay)
 rows.append(row);programs[str(rank)]=program;print(row,flush=True)
train_maha=float((train@inv).square().sum(1).mean());eval_maha=float((delta[indices]@inv).square().sum(1).mean())
primary=next(r for r in rows if r['rank_per_source']==64)
out=dict(predictions=dict(pred_a_export=max(r['source_export_replay'] for r in rows)<1e-8,pred_b_rank64=primary['opened_distinct_normalized_error']<=min(.15,.8*.3516988017360602)),records=rows,weighting_condition=float(ev.max()/ev.min()),training_mean_squared_mahalanobis=train_maha,evaluation_mean_squared_mahalanobis=eval_maha,evaluation_training_geometry_ratio=eval_maha/train_maha,training_documents=256,seconds=time.perf_counter()-start,scope='Expanded secondmoment about fixed old24mean, all old affine/centering terms preserved. No fitting to evaluation outcomes. Seven previouslyopened evaluation prefixes; no fresh or nativeintervention confirmation.')
(P/'EXPANDED_SOURCE_METRIC_FIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
torch.save(programs,P/'EXPANDED_SOURCE_METRIC_PROGRAMS_V1.pt')
torch.save({'second_moment':weight,'root':root,'inverse_root':inv},P/'EXPANDED_SOURCE_METRIC_GEOMETRY_V1.pt')
print({k:v for k,v in out.items() if k!='records'})
