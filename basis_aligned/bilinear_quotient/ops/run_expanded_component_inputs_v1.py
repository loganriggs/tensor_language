#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_inputs pred_b_source pred_c_components
"""Complete the existing expanded calibration cache for all three components.
232 same training prefixes, context64, no new fitting or fresh-panel claims.
pred_a_inputs: existing earlier inputs/later scale/third projection replay <1e-6.
pred_b_source: all6 weight-folded source reads vs native source <1e-4.
pred_c_components: all3 folded component values vs native source values <1e-4.
Null: capture interface is inconsistent with the existing experiment inputs.
Price:232 forwards, zero fits, retains native z/h; no candidate simplicity claim.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 torch.set_num_threads(2)
 plan=json.loads((P/'EXPANDED_COMPONENT_INPUTS_PLAN_V1.json').read_text())
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 old=torch.load(P/'EXPANDED_COVARIANCE_STATES_V1.pt',weights_only=True)
 assert old['tokens'].shape==(232,65)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(plan));return
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 path=P/'EXPANDED_COMPONENT_INPUTS_V1.json';assert not path.exists()
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
 a=torch.stack([p['a'] for p in d['pairs']],1).cuda()
 readers=torch.stack([p[k] for p in d['pairs'] for k in ('a','b')],1).cuda()
 Q=torch.stack([q for p in d['pairs'] for q in p['Qs']]).cuda()
 alpha=torch.stack([p['alpha'] for p in d['pairs']]).cuda();beta=torch.stack([p['beta'] for p in d['pairs']]).cuda()
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17]
 fields={k:[] for k in ['h','t','scale','source_reads','native_phi']};input_checks=[];source_checks=[];num=torch.zeros(3,dtype=torch.float64,device='cuda');den=num.clone()
 for index,row in enumerate(old['tokens']):
  c=capture(model,row[None,:64].cuda());z=c['x16'].flatten(0,1).double();h=c['h17'].flatten(0,1).double()
  source=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).flatten(0,1).double()
  s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();t=h@a;reads=source@readers
  folded=torch.einsum('ni,oij,nj->no',z,Q,z)
  source_checks.append(((folded-reads).norm(dim=0)/reads.norm(dim=0)).cpu().tolist())
  phi=((t-.5*reads[:,::2])/s[:,None]-alpha)*(reads[:,1::2]/s[:,None]-beta)
  replay=((t-.5*folded[:,::2])/s[:,None]-alpha)*(folded[:,1::2]/s[:,None]-beta)
  num+=(replay-phi).square().sum(0);den+=phi.square().sum(0)
  errors=[]
  for new,prior in [(z,old['z'][index]),(s,old['scale'][index]),(t[:,2],old['t'][index]),(phi[:,2],old['native_phi'][index])]:
   prior=prior.cuda().double();errors.append(float((new-prior).norm()/prior.norm().clamp_min(1e-30)))
  input_checks.append(errors)
  for key,value in [('h',h.float()),('t',t),('scale',s),('source_reads',reads),('native_phi',phi)]:fields[key].append(value.cpu())
  if (index+1)%32==0:print('captured',index+1,flush=True)
 component=(num/den).sqrt().cpu().tolist();max_input=max(map(max,input_checks));max_source=max(map(max,source_checks))
 result=dict(plan=plan,predictions=dict(pred_a_inputs=max_input<1e-6,pred_b_source=max_source<1e-4,pred_c_components=max(component)<1e-4),max_prior_input_relative_error=max_input,max_source_relative_error=max_source,component_relative_errors=component,forwards=232,context=64,seconds=time.perf_counter()-start,scope='Same historical calibration prefixes; completes all3 later-state inputs. Native ports remain, no new heldout or OOD evidence.')
 torch.save({**{k:torch.stack(v) for k,v in fields.items()},'tokens':old['tokens']},P/'EXPANDED_COMPONENT_INPUTS_V1.pt')
 path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
