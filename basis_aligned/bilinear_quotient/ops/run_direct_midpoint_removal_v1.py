#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_export pred_b_features pred_c_joint
"""Native removals for extracted16product midpoint program.
pred_a_export reader/writer duality<1e-10 and exportedscalar vs factor FP64replay<1e-5;
pred_b_features each individual centered effect error<.3 and cosine>.95 bothdomains;
pred_c_joint simultaneous four-feature centered effect error<.2 bothdomains.
Null good scalar prediction fails nonlinear native intervention fidelity.
Price48nativeforwards plus10editedreadouts/doc. Constant calibration mean stays in background.
These remove operational feature variations through specified writers, not whole upstream sources.
"""
import os,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,features=4,products=16,edited_readouts=480)));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/'MIDPOINT_REMOVAL_V1.json';assert not out.exists()
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');e=torch.load(P/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);a=torch.load(P/'MIDPOINT_FACTOR_PROGRAMS_V1.pt',weights_only=True)['programs']['separable_moment_4'];e={k:v.cuda().double() for k,v in e.items()};writer=torch.linalg.solve(ru,e['reduced_writers']);duality=float((e['scalar_readers'].T@e['reduced_writers']-torch.eye(4,device='cuda')).norm());L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();C=e['scalar_readers'].T@ru@b17.mlp.Down.weight.double();records=[];checks=[]
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 for domain in ['fineweb','code']:
  ids=torch.load(P/f'SELECTIVE_CONFIRMATION_{domain.upper()}_V1.pt',weights_only=True)
  for doc,row in enumerate(ids):
   tokens=row.cuda()[None];c=capture(model,tokens[:,:256]);h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);s=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-m/2)/s;m=m/s;truth=((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@C.T-e['offset'];pred=((n@e['A'])*(m@e['B']))@e['readout']-e['offset'];factor=(torch.einsum('bi,gir->bgr',n,a['A'].cuda().double())*torch.einsum('bi,gir->bgr',m,a['B'].cuda().double())).sum(-1)-a['offset'].cuda().double();checks.append(float((factor-pred).norm()/factor.norm()));state=c['final'].flatten(0,1);native=logits(state);targets=tokens[:,1:257].flatten();basece=F.cross_entropy(native,targets)
   for mode in [0,1,2,3,'joint']:
    keep=torch.ones(4,device='cuda',dtype=torch.float64) if mode=='joint' else F.one_hot(torch.tensor(mode,device='cuda'),4).double();truez=logits(state-((truth*keep)@writer.T).float());predz=logits(state-((pred*keep)@writer.T).float());te=(truez-native).double();pe=(predz-native).double();te-=te.mean(-1,keepdim=True);pe-=pe.mean(-1,keepdim=True);records.append(dict(domain=domain,document=doc,mode=str(mode),true_energy=float(te.square().sum()),pred_energy=float(pe.square().sum()),dot=float((te*pe).sum()),error_energy=float((te-pe).square().sum()),true_ce_added=float(F.cross_entropy(truez,targets)-basece),pred_ce_added=float(F.cross_entropy(predz,targets)-basece)))
 summary={}
 for domain in ['fineweb','code']:
  summary[domain]={}
  for mode in ['0','1','2','3','joint']:
   rr=[r for r in records if r['domain']==domain and r['mode']==mode];s={k:sum(r[k] for r in rr) for k in ['true_energy','pred_energy','dot','error_energy']};s.update(relative_error=(s['error_energy']/s['true_energy'])**.5,cosine=s['dot']/(s['true_energy']*s['pred_energy'])**.5,true_ce_added=sum(r['true_ce_added'] for r in rr)/len(rr),pred_ce_added=sum(r['pred_ce_added'] for r in rr)/len(rr));summary[domain][mode]=s
 pred=dict(pred_a_export=duality<1e-10 and max(checks)<1e-5,pred_b_features=all(summary[d][m]['relative_error']<.3 and summary[d][m]['cosine']>.95 for d in summary for m in ['0','1','2','3']),pred_c_joint=all(summary[d]['joint']['relative_error']<.2 for d in summary))
 result=dict(predictions=pred,summary=summary,records=records,duality=duality,export_replay=max(checks),seconds=time.perf_counter()-start,scope='Removal of four calibration-centered feature amplitudes via explicit native residual writers. Exact native final normalization and softcap. Reused diagnostic panels, no semantic selectivity or source-ablation claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
