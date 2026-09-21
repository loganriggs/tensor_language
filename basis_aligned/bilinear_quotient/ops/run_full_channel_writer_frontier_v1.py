#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_fidelity pred_c_intervention
"""Same-cost writer anchoring/refit frontier t=0,.5,1 on opened48docs.
Replay<1e-5; midpoint effecterror<10%bothdomains; midpoint retainedp90<10%."""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(captures=48,fit=False)));return
 import torch,numpy as np
 import torch.nn.functional as F
 torch.set_num_threads(2);torch.set_grad_enabled(False);sys.path.insert(0,str(P));from native_feature_capture import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 plan=json.loads((P/'FULL_CHANNEL_WRITER_FRONTIER_PLAN_V1.json').read_text());export=json.loads((P/'FULL_CHANNEL_EXPORT_V1.json').read_text());assert hashlib.sha256((P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt').read_bytes()).hexdigest()==export['program_sha256']
 assert hashlib.sha256((P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt').read_bytes()).hexdigest()==plan['tokens_sha256']
 p={k:v.cuda() for k,v in torch.load(P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt',weights_only=True).items()};tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);model=Bilin18TorchBackend.load('cuda').model.float();start=time.monotonic();rows=[];checks=[]
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 selected=next(r['selected_channels'] for r in json.loads((P/'FULL_CHANNEL_DELETION_V1.json').read_text())['records'] if r['geometry']=='activation_covariance' and r['policy']=='conditional' and r['width']==3686);ids=torch.tensor(selected,device='cuda');mlp=model.transformer.h[17].mlp;L,R,D=mlp.Left.weight.double(),mlp.Right.weight.double(),mlp.Down.weight.double();an=L[ids].norm(dim=1)/p['a'].double().norm(dim=1);bn=R[ids].norm(dim=1)/p['b'].double().norm(dim=1);w0=D[:,ids]*(an*bn);a,b=p['a'].double(),p['b'].double();historical=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)['h'].cuda().double();mu=(historical/(historical.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()).mean(0)
 linear0=D@((mu@R.T)[:,None]*L+(mu@L.T)[:,None]*R)-w0@((mu@b.T)[:,None]*a+(mu@a.T)[:,None]*b);bias0=D@((L@mu)*(R@mu))-w0@((a@mu)*(b@mu))-linear0@mu
 alternatives={str(t):dict(writer=((1-t)*w0+t*p['writer'].double()).float(),linear=((1-t)*linear0+t*p['linear'].double()).float(),bias=((1-t)*bias0+t*p['bias'].double()).float()) for t in (0.,.5,1.)}

 for domain,documents in tokens.items():
  for docid,row in enumerate(documents):
   cache=capture(model,row[None,:256].cuda());h=cache['h17'];z=F.rms_norm(h,(1152,));poly=model.transformer.h[17].mlp(z)-model.transformer.h[17].mlp.Down_bias;final=cache['final'];checks.append(float((h+poly+model.transformer.h[17].mlp.Down_bias-final).norm()/final.norm()));background=final-poly;truth=logits(final)[:,16:255];effect=truth-logits(background)[:,16:255];labels=row[17:256].cuda()[None];ce=F.cross_entropy(truth.flatten(0,1),labels.flatten(),reduction='sum');raw=((z@p['a'].T)*(z@p['b'].T))@p['writer'].T
   for arm in alternatives:
    v=alternatives[arm];replacement=((z@p['a'].T)*(z@p['b'].T))@v['writer'].T+z@v['linear'].T+v['bias'];changed=logits(background+replacement)[:,16:255];rows.append(dict(domain=domain,document_index=docid,arm=arm,error_energy=float((changed-truth).double().square().sum()),truth_energy=float(effect.double().square().sum()),ce_added=float(F.cross_entropy(changed.flatten(0,1),labels.flatten(),reduction='sum')-ce)/labels.numel(),tokens=labels.numel()))
 summaries=[];rng=np.random.default_rng(938)
 for domain in tokens:
  for arm in alternatives:
   rr=[r for r in rows if r['domain']==domain and r['arm']==arm];E=np.array([r['error_energy'] for r in rr]);T=np.array([r['truth_energy'] for r in rr]);ce=np.array([r['ce_added'] for r in rr]);idx=rng.integers(0,len(rr),(2000,len(rr)));summaries.append(dict(domain=domain,arm=arm,effect_relative_error=float((E.sum()/T.sum())**.5),effect_error_ci95=np.quantile((E[idx].sum(1)/T[idx].sum(1))**.5,[.025,.975]).tolist(),ce_added=float(ce.mean()),ce_ci95=np.quantile(ce[idx].mean(1),[.025,.975]).tolist()))
 pred=dict(pred_a_replay=max(checks)<1e-5,pred_b_fidelity=all(r['effect_relative_error']<.1 for r in summaries if r['arm']=='0.5'),pred_c_intervention=.5*json.loads((P/'FULL_CHANNEL_INTERVENTION_MAP_V1.json').read_text())['retained_relative_error_quantiles'][2]<.1)
 (P/'FULL_CHANNEL_WRITER_FRONTIER_V1.json').write_text(json.dumps(dict(rows=rows,summaries=summaries,predictions=pred,maximum_replay=max(checks),program_sha256=export['program_sha256'],plan=plan,seconds=time.monotonic()-start,scope='Opened freshpanel reused for a fixed weight-derived interpolation, not independent validation. No export/adoption. Internal retained-product response fidelity predicted analytically; omitted product interventions remain unmapped.'),indent=2)+'\n');print(json.dumps(dict(predictions=pred,summaries=summaries)),flush=True)
if __name__=='__main__':main()
