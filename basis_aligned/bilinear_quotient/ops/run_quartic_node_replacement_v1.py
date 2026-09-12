#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;32768exact coefficient contractions,128cached validationvectors.
"""pred_a Schur/direct identity<=1e-8 and nonincreasing objective;
pred_b capture gain>=1%; pred_c native write error reduced>=20%.
Replace least conditional node in frozen LBFGS V1. 64 spectral +32 random
candidates plus re-add original; select by coefficients only. 592704 floats.
Null: these candidate banks may not supply useful residual directions.
"""
import os,sys,json,hashlib,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from coupled_quartic_writer_v1 import gram,target_cross,solve,features
from composed_quartic_contraction_v1 import contract
from quartic_residual_addition_v1 import scores
STEM='QUARTIC_NODE_REPLACEMENT_V1'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 assert json.loads((P/'QUARTIC_RESIDUAL_ADDITION_V1_CONTROL.json').read_text())['pred_a']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,exact_coefficient_contractions=32768,maximum_batch=256,fitted_floats=592704)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists();signal.alarm(300)
 torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 native=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')];scale=float(state['transformer.h.17.lambdas'][0])
 initial=torch.load(P/'COUPLED_QUARTIC_LBFGS_V1_PROGRAM.pt',weights_only=True)
 writer=initial['output_writers'].cuda();native[-1]=(metric@writer).T@native[-1]
 spectral=torch.load(P/'FULLSOURCE_QUARTIC_SPECTRAL_V1_PROGRAMS.pt',weights_only=True)['programs'];assert len(spectral)==2
 banks=[initial['input_readers'].cuda()]+[p['input_readers'].reshape(32,1152,16).cuda() for p in spectral]
 weights=[initial['inner_weights'].cuda()]+[p['inner_weights'].reshape(32,16).cuda() for p in spectral]
 torch.manual_seed(91841);banks.append(torch.linalg.qr(torch.randn(32,1152,16,device='cuda'),mode='reduced')[0]);weights.append(torch.randn(32,16,device='cuda'))
 b=torch.cat(banks);n=torch.cat(weights);n=n/n.norm(dim=-1,keepdim=True)
 k=gram(b,n);c=target_cross(b,n,lambda slots:contract(slots,*native,scale))
 oldmix,diag=solve(k[:32,:32],c[:32]);oldcapture=float((oldmix*c[:32]).sum())
 oldresult=json.loads((P/'COUPLED_QUARTIC_LBFGS_V1_RESULT.json').read_text());replay=abs(oldcapture/initial['divisor']+oldresult['history'][-1]['objective'])
 conditional=oldmix.square().sum(-1)/torch.linalg.inv(k[:32,:32]).diagonal();dropped=int(conditional.argmin())
 retained=torch.tensor([j for j in range(32) if j!=dropped],device='cuda')
 gain,variance,base=scores(k,c,retained);chosen=int(gain.argmax());assert torch.isfinite(gain[chosen])
 selected=torch.arange(32,device='cuda');selected[dropped]=chosen
 mixing,diagnostics=solve(k[selected][:,selected],c[selected]);capture=float((mixing*c[selected]).sum())
 identity=abs(capture-base-float(gain[chosen]))/oldcapture
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
 den=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
 reference=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0].cuda()
 def native_error(bank,weight,mix):return float((features(bank,weight,x)@mix@writer.T/den[:,None]-reference).norm()/reference.norm())
 before=native_error(b[:32],n[:32],oldmix);after=native_error(b[selected],n[selected],mixing)
 torch.save(dict(seed=11511,input_readers=b[selected].cpu(),inner_weights=n[selected].cpu(),mixing=mixing.cpu(),output_writers=writer.cpu(),divisor=initial['divisor']),ap)
 candidates=[dict(index=j,addition_capture_fraction=float(gain[j])/oldcapture,residual_variance=float(variance[j])) for j in range(128) if torch.isfinite(gain[j])]
 result={'pred_a':identity<=1e-8 and replay<=1e-8 and capture>=oldcapture*(1-1e-10) and diagnostics['normal_residual']<=1e-8,
         'pred_b':capture>=1.01*oldcapture,'pred_c':after<=.8*before}
 result.update(dict(dropped_node=dropped,chosen_candidate=chosen,
             candidate_families=['0:32 original','32:64 spectral seed '+str(spectral[0]['seed']),'64:96 spectral seed '+str(spectral[1]['seed']),'96:128 random seed91841'],
             candidates=candidates,coefficient_gain_fraction=capture/oldcapture-1,schur_identity_error=identity,baseline_objective_replay_error=replay,
             initial_native_error=before,final_native_error=after,artifact_sha256=digest(ap),wall_seconds=time.perf_counter()-tic,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
             scope='Frozen old endpoint, one feature replacement and exact output refit; no nonlinear candidate fitting or native data selection, no circuit claim.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='candidates'}),flush=True);assert result['pred_a']
if __name__=='__main__':main()
