#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fidelity pred_c_behavior
"""Original-weight observer and modal removals on reused32FW confirmation rows.
pred_a_instrument scalar weight replay<1e-8 and frozen group2 CE replay<1e-6.
pred_b_fidelity rank3 removal-effect error<=.15 on all/continuation/spaced cohorts.
pred_c_behavior native rank1 continuationCE>.05 and abs(spacedCE)<.02.
All ranks and standalone modes retained. No fitting and no fresh-data claim.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=32,arms=['native','frozen','rank1','rank2','rank3','rank8','mode2','mode3'],fit=False)));return
 import torch
 import torch.nn.functional as F
 import tiktoken
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 from token_boundary_conditions import annotate
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 out=P/'MIDPOINT_NATIVE_OBSERVER_EFFECTS_V1.json';assert not out.exists();artifact=P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt'
 e={k:v.cuda().double() for k,v in torch.load(artifact,weights_only=True).items()};tokens=torch.load(P/'MIDPOINT_CONTINUATION_GROUP_TOKENS_V1.pt',weights_only=True);plan=json.loads((P/'MIDPOINT_CONTINUATION_GROUP_PLAN_V1.json').read_text());assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==plan['token_sha256']
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();channel=b17.mlp.Down.weight.double().T@(e['R_U'].T@e['q']);residual_writer=torch.linalg.solve(e['R_U'],e['writer'])
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 enc=tiktoken.get_encoding('gpt2');records=[];checks=[]
 for doc,row in enumerate(tokens):
  labels=annotate(row.tolist(),enc);c=capture(model,row[None,:256].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=((h-m/2)/scale)[16:]-e['mean_n'];m=(m/scale)[16:]-e['mean_m'];state=c['final'].flatten(0,1)[16:];base=logits(state);target=row[17:257].cuda();ce=F.cross_entropy(base,target,reduction='none')
  native=((n@e['native_matrix'])*m).sum(1)
  if doc==0:
   direct=((n@L.T)*(m@R.T)+(n@R.T)*(m@L.T))@channel;checks.append(float((native-direct).norm()/direct.norm()))
  phi=(n@e['A'])*(m@e['B']);arms=dict(native=native,frozen=(n@e['frozen_a'])*(m@e['frozen_b']),rank1=phi[:,0],rank2=phi[:,:2].sum(1),rank3=phi[:,:3].sum(1),rank8=phi.sum(1),mode2=phi[:,1],mode3=phi[:,2])
  masks=dict(all=torch.ones(240,dtype=torch.bool,device='cuda'),continuation=torch.tensor([v['continuation'] for v in labels[16:256]],device='cuda'),spaced_word=torch.tensor([v['spaced_word'] for v in labels[16:256]],device='cuda'))
  for name,scalar in arms.items():
   changed=logits(state-(scalar[:,None]*residual_writer[None,:]).float());effect=(changed-base).double();effect-=effect.mean(1,keepdim=True);damage=F.cross_entropy(changed,target,reduction='none')-ce
   if name=='native':reference=effect
   for cohort,mask in masks.items():
    records.append(dict(document=int(plan['documents'][doc]),candidate=name,cohort=cohort,sites=int(mask.sum()),ce_added_sum=float(damage[mask].sum()),reference_energy=float(reference[mask].square().sum()),error_energy=float((effect[mask]-reference[mask]).square().sum()),effect_energy=float(effect[mask].square().sum())))
 summary={}
 for name in arms:
  summary[name]={}
  for cohort in masks:
   rr=[v for v in records if v['candidate']==name and v['cohort']==cohort];num=sum(v['sites'] for v in rr);energy=sum(v['reference_energy'] for v in rr)
   summary[name][cohort]=dict(sites=num,ce_added=sum(v['ce_added_sum'] for v in rr)/num,effect_relative_error=(sum(v['error_energy'] for v in rr)/energy)**.5)
 previous=json.loads((P/'MIDPOINT_CONTINUATION_GROUP_NATIVE_V1.json').read_text())['summary']['real'];ce_replay=max(abs(summary['frozen'][k]['ce_added']-previous[k]['ce_added']) for k in ['continuation','spaced_word'])
 pred=dict(pred_a_instrument=max(checks)<1e-8 and ce_replay<1e-6,pred_b_fidelity=all(summary['rank3'][k]['effect_relative_error']<=.15 for k in masks),pred_c_behavior=summary['rank1']['continuation']['ce_added']>.05 and abs(summary['rank1']['spaced_word']['ce_added'])<.02)
 result=dict(predictions=pred,summary=summary,records=records,scalar_weight_replay=max(checks),frozen_ce_replay=ce_replay,artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),seconds=time.perf_counter()-start,scope='Original-weight centered bilinear output projection onto fixed learned writer. Mode factors fit original weights using calibration covariance only. Reused32FW validation panel, no new data claim. Removal of an explicitly defined projection, not whole native module or upstream ablation; final nonlinearities native.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
