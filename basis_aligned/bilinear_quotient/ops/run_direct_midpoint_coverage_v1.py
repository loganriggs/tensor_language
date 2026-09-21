#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_coverage pred_c_weighted
"""Expand original-weight output coverage to 256 directions, 80 native forwards.
pred_a_replay dense scalar teacher<1e-9 and orthogonal energy identity<1e-9.
pred_b_coverage weighted rank4/256output full variation error<.3 both held domains.
pred_c_weighted weighted rank4/256output MSE<isotropic in both held domains.
Calibration-only output basis and second moments; reused held panels; no native intervention claim.
"""
import os,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=80,output_widths=[4,16,64,256],ranks=[1,4,16],matrix_svds=512)));return
 import torch
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from weighted_bilinear_svd import roots,decompose
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/'MIDPOINT_COVERAGE_SWEEP_V1.json';assert not out.exists()
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();C=ru@b17.mlp.Down.weight.double()
 art=torch.load(P/'MIDPOINT_NATIVE_V1.pt',weights_only=True);S=torch.load(P/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].cuda();st=art['stats']['calibration']['native'];mu=st['mean'].cuda();cov=st['second'].cuda()-mu[:,None]*mu[None,:];_,W=torch.linalg.eigh(S@cov@S);W=W.flip(1)[:,:256];q=S@W;v=torch.linalg.solve(S,W);offset=mu@q;channel=q.T@C
 panels={'calibration':torch.load(BQ/'.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:32,:65],'fineweb':torch.load(P/'SELECTIVE_CONFIRMATION_FINEWEB_V1.pt',weights_only=True),'code':torch.load(P/'SELECTIVE_CONFIRMATION_CODE_V1.pt',weights_only=True)};data={}
 for name,ids in panels.items():
  rows=[];length=64 if name=='calibration' else 256
  for row in ids:
   c=capture(model,row[None,:length].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);s=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-m/2)/s;m=m/s;y=((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@C.T;rows.append((n,m,(y-mu)@S))
  data[name]=tuple(torch.cat([z[i] for z in rows]) for i in range(3));print('captured',name,flush=True)
 n,m,_=data['calibration'];left=roots(n.T@n/len(n));right=roots(m.T@m/len(m));eye=torch.eye(1152,device='cuda',dtype=torch.float64);errs={metric:{rank:{d:[] for d in data} for rank in [1,4,16]} for metric in ['isotropic','separable_moment']};savedA=[];savedB=[];replay=[]
 for g in range(256):
  k=L.T@(channel[g,:,None]*R)+R.T@(channel[g,:,None]*L)
  if g in [0,63,255]:
   direct=torch.einsum('bi,ij,bj->b',n,k,m)-offset[g];truth=data['calibration'][2]@W[:,g];replay.append(float((direct-truth).norm()/truth.norm()))
  for metric in errs:
   a,b,_=decompose(k,(eye,eye) if metric=='isotropic' else left,(eye,eye) if metric=='isotropic' else right)
   if metric=='separable_moment':savedA.append(a[:,:4].float().cpu());savedB.append(b[:,:4].float().cpu())
   for d,(nn,mm,yy) in data.items():
    products=(nn@a[:,:16])*(mm@b[:,:16]);truth=yy@W[:,g]
    for rank in errs[metric]:errs[metric][rank][d].append(float((products[:,:rank].sum(-1)-offset[g]-truth).square().sum()))
  if (g+1)%32==0:print('directions',g+1,flush=True)
 records=[];identity=[]
 for d,(nn,mm,yy) in data.items():
  total=float(yy.square().sum());projected=yy@W
  for width in [4,16,64,256]:
   kept=float(projected[:,:width].square().sum());omitted=total-kept;res=yy-projected[:,:width]@W[:,:width].T;identity.append(abs(float(res.square().sum())-omitted)/total)
   for metric in errs:
    for rank in errs[metric]:
     retained=sum(errs[metric][rank][d][:width]);records.append(dict(domain=d,outputs=width,rank_per_output=rank,metric=metric,products=width*rank,input_coefficients=2304*width*rank,writer_coefficients=1152*width,full_variation_relative_error=((omitted+retained)/total)**.5,output_projection_floor=(omitted/total)**.5,retained_error_over_full_variation=(retained/total)**.5))
 def chosen(d,metric):return next(x for x in records if x['domain']==d and x['outputs']==256 and x['rank_per_output']==4 and x['metric']==metric)
 pred=dict(pred_a_replay=max(replay)<1e-9 and max(identity)<1e-9,pred_b_coverage=all(chosen(d,'separable_moment')['full_variation_relative_error']<.3 for d in ['fineweb','code']),pred_c_weighted=all(chosen(d,'separable_moment')['full_variation_relative_error']<chosen(d,'isotropic')['full_variation_relative_error'] for d in ['fineweb','code']))
 A=torch.stack(savedA,dim=1).reshape(1152,1024);B=torch.stack(savedB,dim=1).reshape(1152,1024)
 for width in [64,256]:
  readout=torch.zeros(width*4,width);readout[torch.arange(width*4),torch.arange(width*4)//4]=1
  torch.save(dict(A=A[:,:width*4],B=B[:,:width*4],readout=readout,offset=offset[:width].cpu(),scalar_readers=q[:,:width].cpu(),reduced_writers=v[:,:width].cpu()),P/f'MIDPOINT_COVERAGE_{width}_R4_V1.pt')
 result=dict(predictions=pred,records=records,teacher_replay=max(replay),projection_identity=max(identity),seconds=time.perf_counter()-start,scope='Full centered output variation relative to fixed calibration mean, original folded weight target, explicit normalization; reused held panels, no interventions yet. Input products assume upstream midpoint/source coordinates available.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
