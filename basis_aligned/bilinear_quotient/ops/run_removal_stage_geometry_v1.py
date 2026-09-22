#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_scalar pred_c_geometry
"""Six frozen programs, five metric stages. Replay<1e-4, native<1e-5.
Atleast4/5leanfailurecells scalarerror>.1; final/scalar ratiosall[.5,2].
Opened32prefixes; cacheinterfaces; no fitting or semanticadoption.
"""
import os,sys,json,time,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch,tiktoken
 import torch.nn.functional as F
 sys.path.insert(0,str(P));sys.path.insert(0,str(ROOT/'basis_aligned/bilinear_quotient/ops'))
 from native_quartic_branch import pure_branch,bilinear
 from conditional_quartic_cp import evaluate as full
 from lean_conditional_cp import evaluate as lean
 torch.set_num_threads(2)
 def cp(p,x):return torch.stack([x@a.T for a in p['factors']]).prod(0)@p['coefficients'].T
 def amounts(err,ref):return err.double().square().sum(-1),ref.double().square().sum(-1)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  x=torch.randn(7,1152);w=torch.randn(1152);a=torch.randn(7);den=torch.rand(7)+.5;z=x-a[:,None]*w/den[:,None];raw=torch.nn.functional.rms_norm(x,(1152,));edit=torch.nn.functional.rms_norm(z,(1152,));e,r=amounts(edit-edit,edit-raw);assert torch.equal(e,torch.zeros_like(e)) and r.shape==(7,)
  for selected in [torch.tensor([True,False,True,False,True,False,True]),torch.zeros(7,dtype=torch.bool)]:assert float(e[selected].sum())==0
  print(json.dumps(dict(stages=5,programs=6,domains=2,strata=2,amounts=2,zero_replay=True)));return
 from native_feature_capture import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 out=P/'REMOVAL_STAGE_GEOMETRY_V1.json';assert not out.exists();frozen=json.loads((P/'REMOVAL_STAGE_GEOMETRY_INPUTS_V1.json').read_text())
 for name,h in frozen.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h,name
 torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();programs=[];sources={}
 for kind,stem,fn,receipt in [('parent','MIXED_CP_FEATURES',cp,'MIXED_CP_REMOVAL_V1.json'),('full','CONDITIONAL_CP',full,'CONDITIONAL_CP_REMOVAL_V1.json'),('lean','LEAN_CONDITIONAL_CP',lean,'LEAN_CONDITIONAL_CP_REMOVAL_V1.json')]:
  sources[kind]=json.loads((P/receipt).read_text())
  for seed in [1001,1002]:
   p=torch.load(P/(f'{stem}_SEED{seed}'+('' if kind=='parent' else '_RANK256')+'_V1.pt'),weights_only=True);p={k:[a.cuda().float() for a in v] if isinstance(v,list) else v.cuda().float() if isinstance(v,torch.Tensor) else v for k,v in p.items()};programs.append((kind,seed,p,fn))
 tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];w=programs[0][2]['writer'][:,1];uw=model.lm_head(w);reader=model.lm_head.weight.T@uw/(uw@uw);enc=tiktoken.get_encoding('gpt2');rows=[];cached=[];native_replays=[];zero_errors=[]
 for domain,documents in tokens.items():
  for doc,row in enumerate(documents[:16]):
   cap=capture(model,row[None,:256].cuda());x,m,h,final=[cap[k].flatten(0,1)[16:255] for k in ['x16','m16','h17','final']];l,r,d=[getattr(b17.mlp,k).weight for k in ['Left','Right','Down']];branch=pure_branch(m,b16.mlp.Down_bias,b17.lambdas[0],l,r,d);ind=bilinear(b17.lambdas[0]*bilinear(x,b16.mlp.Left.weight,b16.mlp.Right.weight,b16.mlp.Down.weight),l,r,d);native_replays.append(float((branch-ind).norm()/branch.norm()));exact=branch@reader;den=h.square().mean(-1)+torch.finfo(h.dtype).eps;mask=torch.tensor([b'\n' in enc.decode_single_token_bytes(int(t)) for t in row[16:255]],device='cuda');baseR=F.rms_norm(final,(1152,));baseL=model.lm_head(baseR);baseZ=30*torch.tanh(baseL/30);approx={(kind,seed):fn(p,x)[:,1] for kind,seed,p,fn in programs}
   zero=30*torch.tanh(model.lm_head(F.rms_norm(final-0*exact[:,None]*w/den[:,None],(1152,)))/30);zero_errors.append(float((zero-baseZ).abs().max()))
   cached.append(dict(domain=domain,document=doc,x=x.cpu(),final=final.cpu(),denominator=den.cpu(),exact=exact.cpu(),newline=mask.cpu(),tokens=row[16:255].cpu(),candidates={f'{k}_{s}':a.cpu() for (k,s),a in approx.items()}))
   for alpha in [.25,1.]:
    nativeR=F.rms_norm(final-alpha*exact[:,None]*w/den[:,None],(1152,));nativeL=model.lm_head(nativeR);nativeZ=30*torch.tanh(nativeL/30)
    for kind,seed,p,fn in programs:
     a=approx[kind,seed];candidateR=F.rms_norm(final-alpha*a[:,None]*w/den[:,None],(1152,));candidateL=model.lm_head(candidateR);candidateZ=30*torch.tanh(candidateL/30)
     stages=[((a-exact).double().square(),exact.double().square()),(((a-exact)/den).double().square(),(exact/den).double().square()),amounts(candidateR-nativeR,nativeR-baseR),amounts(candidateL-nativeL,nativeL-baseL),amounts(candidateZ-nativeZ,nativeZ-baseZ)]
     for condition,selected in [('newline',mask),('other',~mask)]:
      if int(selected.sum()):rows.append(dict(kind=kind,seed=seed,domain=domain,document=doc,alpha=alpha,condition=condition,n=int(selected.sum()),stage_error_energy=[float(e[selected].sum()) for e,ref in stages],stage_reference_energy=[float(ref[selected].sum()) for e,ref in stages]))
 keys=sorted({(r['kind'],r['seed'],r['domain'],r['alpha'],r['condition']) for r in rows});summary=[];replay=[]
 for kind,seed,domain,alpha,condition in keys:
  rr=[r for r in rows if (r['kind'],r['seed'],r['domain'],r['alpha'],r['condition'])==(kind,seed,domain,alpha,condition)];en=[sum(r['stage_error_energy'][i] for r in rr) for i in range(5)];rn=[sum(r['stage_reference_energy'][i] for r in rr) for i in range(5)];errors=[math.sqrt(e/n) for e,n in zip(en,rn)];old=next(r for r in sources[kind]['summary'] if (r['seed'],r['domain'],r['alpha'],r['condition'])==(seed,domain,alpha,condition));replay.append(abs(errors[-1]-old['error']));summary.append(dict(kind=kind,seed=seed,domain=domain,alpha=alpha,condition=condition,n=sum(r['n'] for r in rr),errors=errors,final_over_scalar=errors[-1]/errors[0],previous_error=old['error']))
 failed=[r for r in summary if r['kind']=='lean' and r['previous_error']>.1];pred=dict(pred_a_instrument=len(summary)==48 and max(replay)<1e-4 and max(native_replays)<1e-5 and max(zero_errors)==0,pred_b_scalar=sum(r['errors'][0]>.1 for r in failed)>=4,pred_c_geometry=all(.5<=r['final_over_scalar']<=2 for r in summary));cachepath=P/'REMOVAL_STAGE_STATES_V1.pt';torch.save(dict(rows=cached,writer=w.cpu(),scope='Opened32prefixes,positions16–254; nativecoordinate1removalinterface, not freshvalidation.'),cachepath);out.write_text(json.dumps(dict(predictions=pred,summary=summary,rows=rows,maximum_final_replay=max(replay),maximum_native_replay=max(native_replays),maximum_zero_edit=max(zero_errors),stage_names=['scalar','local_denominator','final_RMS','unembedding','softcap'],seconds=time.monotonic()-start,cache_sha256=hashlib.sha256(cachepath.read_bytes()).hexdigest(),scope='Openednativecoordinate1failurelocalization acrosssuccessivemetrics. Not additive attribution, no fitting/selectivity/OOD claim.'),indent=2)+'\n');print(json.dumps(pred),flush=True)
if __name__=='__main__':main()
