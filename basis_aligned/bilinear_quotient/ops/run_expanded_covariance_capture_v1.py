#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_tokens pred_b_source pred_c_product
"""232 new training-prefix captures for expanded source weighting.
Exact folded mode3 source reads and composed scalar replay<1e-4.
Unique64tokenprefixes, disjoint from all32oldcalibration prefixes.
No fitting/evaluation selection in this capture.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 torch.set_num_threads(2)
 tokens=torch.load(P/'EXPANDED_COVARIANCE_TOKENS_V1.pt',weights_only=True)
 plan=json.loads((P/'EXPANDED_COVARIANCE_PANEL_V1.json').read_text())
 old=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True)['tokens']
 digest=lambda row:hashlib.sha256(row[:64].numpy().tobytes()).hexdigest()
 hashes={digest(row) for row in tokens}
 tokencheck=len(hashes)==232 and not(hashes&{digest(row) for row in old}) and hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==plan['token_sha256']
 assert tokencheck and tokens.shape==(232,65)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=232,context=64,prefix_disjointness='PASS',fit=False,calibration_only=True)));return
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 out=P/'EXPANDED_COVARIANCE_CAPTURE_V1.json';assert not out.exists()
 mode={k:v.cuda().double() for k,v in torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True).items()}
 a=mode['A'][:,2];b=mode['B'][:,2];alpha=mode['mean_n']@a;beta=mode['mean_m']@b
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17]
 state=model.state_dict();L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();Down=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'][0].double();Qs=[]
 for reader in [a,b]:
  raw=L.T@((lam*(Down.T@reader))[:,None]*R);Qs.append((raw+raw.T)/2)
 fields={k:[] for k in ['z','t','scale','source_reads','native_phi']};checks=[];num=den=0.
 for index,row in enumerate(tokens):
  c=capture(model,row[None,:64].cuda());z=c['x16'].flatten(0,1).double();h=c['h17'].flatten(0,1).double()
  source=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).flatten(0,1).double()
  scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();t=h@a
  reads=torch.stack([source@a,source@b],1)
  folded=torch.stack([((z@Q)*z).sum(1) for Q in Qs],1)
  checks.append(float((folded-reads).norm()/reads.norm()))
  phi=((t-.5*reads[:,0])/scale-alpha)*(reads[:,1]/scale-beta)
  replay=((t-.5*folded[:,0])/scale-alpha)*(folded[:,1]/scale-beta)
  num+=float((replay-phi).square().sum());den+=float(phi.square().sum())
  for key,value in [('z',z.float()),('t',t),('scale',scale),('source_reads',reads),('native_phi',phi)]:fields[key].append(value.cpu())
  if (index+1)%32==0:print('captured',index+1,flush=True)
 result=dict(predictions=dict(pred_a_tokens=tokencheck,pred_b_source=max(checks)<1e-4,pred_c_product=(num/den)**.5<1e-4),
  max_source_relative_error=max(checks),product_relative_error=(num/den)**.5,forwards=232,context=64,seconds=time.perf_counter()-start,token_sha256=plan['token_sha256'],
  scope='Additional covariance training states. Fullnativeinputdependencies remain; exactweightfold checkedagainstnativefloat32. No heldout/semantic/intervention confirmation.')
 torch.save({**{k:torch.stack(v) for k,v in fields.items()},'tokens':tokens},P/'EXPANDED_COVARIANCE_STATES_V1.pt')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
