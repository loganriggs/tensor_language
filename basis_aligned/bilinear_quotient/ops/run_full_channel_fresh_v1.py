#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_fidelity pred_c_repair pred_d_ce
"""Frozen full-layer baseline on32newFineWebdocs+16newstdlibfiles.
No fits. Replay<1e-5; botharms/domain effecterror<10%; affineimprovesbothdomains;
bootstrap document95%upper meanCEadded<.02nats/token forcorrectedbothdomains.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(captures=48,fit=False)));return
 import torch,numpy as np
 import torch.nn.functional as F
 torch.set_num_threads(2);torch.set_grad_enabled(False);sys.path.insert(0,str(P));from native_feature_capture import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 plan=json.loads((P/'FULL_CHANNEL_FRESH_PLAN_V1.json').read_text());export=json.loads((P/'FULL_CHANNEL_EXPORT_V1.json').read_text());assert hashlib.sha256((P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt').read_bytes()).hexdigest()==export['program_sha256']
 assert hashlib.sha256((P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt').read_bytes()).hexdigest()==plan['tokens_sha256']
 p={k:v.cuda() for k,v in torch.load(P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt',weights_only=True).items()};tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);model=Bilin18TorchBackend.load('cuda').model.float();start=time.monotonic();rows=[];checks=[]
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 for domain,documents in tokens.items():
  for docid,row in enumerate(documents):
   cache=capture(model,row[None,:256].cuda());h=cache['h17'];z=F.rms_norm(h,(1152,));poly=model.transformer.h[17].mlp(z)-model.transformer.h[17].mlp.Down_bias;final=cache['final'];checks.append(float((h+poly+model.transformer.h[17].mlp.Down_bias-final).norm()/final.norm()));background=final-poly;truth=logits(final)[:,16:255];effect=truth-logits(background)[:,16:255];labels=row[17:256].cuda()[None];ce=F.cross_entropy(truth.flatten(0,1),labels.flatten(),reduction='sum');raw=((z@p['a'].T)*(z@p['b'].T))@p['writer'].T
   for arm in ('raw','affine'):
    replacement=raw if arm=='raw' else raw+z@p['linear'].T+p['bias'];changed=logits(background+replacement)[:,16:255];rows.append(dict(domain=domain,document_index=docid,arm=arm,error_energy=float((changed-truth).double().square().sum()),truth_energy=float(effect.double().square().sum()),ce_added=float(F.cross_entropy(changed.flatten(0,1),labels.flatten(),reduction='sum')-ce)/labels.numel(),tokens=labels.numel()))
 summaries=[];rng=np.random.default_rng(938)
 for domain in tokens:
  for arm in ('raw','affine'):
   rr=[r for r in rows if r['domain']==domain and r['arm']==arm];E=np.array([r['error_energy'] for r in rr]);T=np.array([r['truth_energy'] for r in rr]);ce=np.array([r['ce_added'] for r in rr]);idx=rng.integers(0,len(rr),(2000,len(rr)));summaries.append(dict(domain=domain,arm=arm,effect_relative_error=float((E.sum()/T.sum())**.5),effect_error_ci95=np.quantile((E[idx].sum(1)/T[idx].sum(1))**.5,[.025,.975]).tolist(),ce_added=float(ce.mean()),ce_ci95=np.quantile(ce[idx].mean(1),[.025,.975]).tolist()))
 pred=dict(pred_a_replay=max(checks)<1e-5,pred_b_fidelity=all(r['effect_relative_error']<.1 for r in summaries),pred_c_repair=all(r['effect_relative_error']<next(t['effect_relative_error'] for t in summaries if t['domain']==r['domain'] and t['arm']=='raw') for r in summaries if r['arm']=='affine'),pred_d_ce=all(r['ce_ci95'][1]<.02 for r in summaries if r['arm']=='affine'))
 (P/'FULL_CHANNEL_FRESH_V1.json').write_text(json.dumps(dict(rows=rows,summaries=summaries,predictions=pred,maximum_replay=max(checks),program_sha256=export['program_sha256'],plan=plan,seconds=time.monotonic()-start,scope='Frozen full lastMLP replacement on document-indexed excluded panel; no pretraining/project-independence guarantee, no semantic units or selective intervention evidence.'),indent=2)+'\n');print(json.dumps(dict(predictions=pred,summaries=summaries)),flush=True)
if __name__=='__main__':main()
