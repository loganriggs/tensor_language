#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_capacity pred_c_writer_gap
"""Opened finite-response readout oracle, fixed4/32 dictionaries.
Orthogonality<1e-8, frame replay<1e-5; wideoracle<.10 and<=.5oldwriter eachcell.
No export: diagnostic unrestricted writer, not nonlinear logit fidelity or OOD.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 sys.path.insert(0,str(P))
 import torch
 from audit_wide_quartic_readout_span import oracle,controls
 from empirical_quartic_dictionary import features
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 control=controls()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  x=torch.randn(33,7,dtype=torch.float64);u=torch.randn(4,2,7,dtype=torch.float64);v=torch.randn_like(u);phi=features(x,u,v);delta=features(x.roll(1,0),u,v)-phi;_,info=oracle(delta,delta@torch.randn(10,3,dtype=torch.float64));assert info['error']<1e-10;print(json.dumps(dict(controls=control,shape=info,captures=32,fit='oracle_only')));return
 from native_feature_capture import capture
 from native_quartic_branch import bilinear
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'QUARTIC_RESPONSE_SPAN_V1.json';assert not out.exists()
 filenames={'tokens':'FULL_CHANNEL_FRESH_TOKENS_V1.pt','narrow26':'WIDE_NATIVE_QUARTIC_4_INHERITED_LONG_V1.pt','parent656':'QUARTIC_JOINT_READOUT_LAMBDA_1_V1.pt'}
 hashes={k:hashlib.sha256((P/f).read_bytes()).hexdigest() for k,f in filenames.items()};frozen=json.loads((P/'PAIRED_ROOT_BRANCH_INPUTS_V1.json').read_text())['sha256'];assert all(v==frozen[k] for k,v in hashes.items())
 tokens=torch.load(P/filenames['tokens'],weights_only=True);programs={k:{a:t.cuda().double() for a,t in torch.load(P/f,weights_only=True).items()} for k,f in filenames.items() if k!='tokens'}
 model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight);ru=ru.double();scale=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['teacher_scale']
 def physical(x):return bilinear(b17.lambdas[0]*bilinear(x,b16.mlp.Left.weight,b16.mlp.Right.weight,b16.mlp.Down.weight),b17.mlp.Left.weight,b17.mlp.Right.weight,b17.mlp.Down.weight)
 def solve(p,y):
  writer,info=oracle(p,y);res=p@writer-y;info['orthogonality']=float((p.T@res).norm()/(p.norm()*y.norm()).clamp_min(1e-30));return writer,info
 records=[];pooled=[];checks=[]
 for domain,documents in tokens.items():
  xx=[]
  for row in documents[:16]:xx.append(capture(model,row[None,:256].cuda())['x16'].flatten(0,1)[16:255])
  x=torch.cat(xx);donor=torch.cat(xx[1:]+xx[:1]);base=physical(x).double();ys=[];changed=[]
  for alpha in [.5,1.]:
   z=x+alpha*(donor-x);changed.append(z);delta=physical(z).double()-base;ys.append(delta@ru.T/scale)
  for name,program in programs.items():
   u,v=program['U'],program['V'];p0=features(x.double(),u,v);deltas=[features(z.double(),u,v)-p0 for z in changed];writer=program['C'].T@ru.T/scale
   physical_delta=deltas[0]@program['C'].T;check=(physical_delta@ru.T/scale-deltas[0]@writer).norm()/(deltas[0]@writer).norm();checks.append(float(check))
   for alpha,p,y in zip([.5,1.],deltas,ys):
    _,info=solve(p,y);row=dict(domain=domain,arm=name,alpha=alpha,baseline_error=float((p@writer-y).norm()/y.norm()),oracle=info);records.append(row);print(json.dumps(row),flush=True)
   joint,info=solve(torch.cat(deltas),torch.cat(ys));pooled.append(dict(domain=domain,arm=name,oracle=info,individual_errors=[float((p@joint-y).norm()/y.norm()) for p,y in zip(deltas,ys)]))
 primary=[r for r in records if r['arm']=='parent656'];pred=dict(pred_a_integrity=max(checks)<1e-5 and all(r['oracle']['orthogonality']<1e-8 for r in records+pooled),pred_b_capacity=all(r['oracle']['error']<.1 for r in primary),pred_c_writer_gap=all(r['oracle']['error']<=.5*r['baseline_error'] for r in primary))
 result=dict(predictions=pred,records=records,pooled=pooled,maximum_frame_replay=max(checks),controls=control,sha256=hashes,seconds=time.monotonic()-start,scope='Oracle projection of purequartic finite differences in unembedding Euclidean metric; opened targets; not nonlinear native-logit capacity or OOD.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,pooled=pooled,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
