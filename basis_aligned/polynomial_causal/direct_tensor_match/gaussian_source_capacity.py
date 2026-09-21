"""Original-source capacity curve, no outcome-fitted directions.
Fixed rank64 primary; all native scores are opened diagnostics.
"""
from pathlib import Path
import json,time
import torch
from gaussian_quartic_lowrank import prepare_teacher,squared_error_by_degree
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
data=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);c=data['common'];f=data['frames']['covariance']
mode=torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True)
path='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
state=torch.load(path,weights_only=True,mmap=True,map_location='cpu')
L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();Down=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'].double()[0]
root=torch.linalg.inv(c['source_inverse_root']);spectra={}
for k,reader in [('a',c['h_reader']),('b',mode['B'][:,2])]:
 w=lam*(Down.T@reader);raw=L.T@(w[:,None]*R);Q=(raw+raw.T)/2
 ev,U=torch.linalg.eigh(root@Q@root);ids=ev.abs().argsort(descending=True)
 spectra[k]=(ev[ids],U[:,ids])
teacher=prepare_teacher(f['A'],f['B']);rows=[]
indices=torch.tensor([j for i in [24,25,26,27,29,30,31] for j in range(i*64,(i+1)*64)])
for rank in [8,16,32,64,128]:
 factors=[];reads=[]
 for index,(k,factor) in enumerate([('a',-.5),('b',1.)]):
  ev,V=spectra[k];U=V[:,:rank]*ev[:rank].abs().sqrt();sign=ev[:rank].sign()
  X=torch.cat([f['J']@U,f[k+'_fixed_vectors'],f['unit'][:,None]],1)
  weights=torch.cat([factor*sign,f[k+'_fixed_values'],(-factor*(U.square().sum(0)*sign).sum()).reshape(1)])
  factors.extend([X,weights])
  reads.append(data['affine'][:,index]+(((data['whitened_delta']@U).square()-U.square().sum(0))*sign).sum(1))
 gaussian=float((squared_error_by_degree(teacher,*factors).sum()/teacher['variance']).clamp_min(0).sqrt())
 phi=((c['t']-.5*reads[0])/c['scale']-c['alpha'])*(reads[1]/c['scale']-c['beta'])
 errors={}
 for name,sl in [('train',slice(0,24*64)),('opened_distinct',indices)]:
  truth=c['true_phi'][sl];errors[name+'_normalized_error']=float((phi[sl]-truth).norm()/(truth-truth.mean()).norm())
 row=dict(rank_per_source=rank,source_products=2*rank,stored_scalars=2*1152*rank+2*rank+4*1152+4,gaussian_numerator_variation_error=gaussian,**errors);rows.append(row);print(row,flush=True)
primary=next(r for r in rows if r['rank_per_source']==64)
out=dict(predictions=dict(pred_a_rank16_replay=abs(rows[1]['opened_distinct_normalized_error']-.3986787277891258)<1e-7,pred_b_rank64=primary['opened_distinct_normalized_error']<=.15),records=rows,seconds=time.perf_counter()-start,scope='Spectral source truncation using24traininggeometry; no joint refit. Native evaluation on seven previouslyopened distinct rows. Prices include sourceprogram and fixed downstream readers/writer, exclude native input producers and common RMS computation.')
(P/'GAUSSIAN_SOURCE_CAPACITY_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['predictions'])
