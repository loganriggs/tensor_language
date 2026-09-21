#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_full pred_c_mode
"""Compare frozen observer removals inside native/compressed backgrounds.
Native background replay<1e-5. Full observer and leading-mode removal effect
errors each<.10 in every nonempty domain/cohort. Opened48documents, no fitting.
Null: operator preservation fails after normalized native composition.
Price unchanged:3686 products,14,067,072 coefficients; interventions diagnostic.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(captures=48,fit=False,cohorts=['all','continuation','spaced_word'])));return
 import torch,tiktoken
 import torch.nn.functional as F
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 from token_boundary_conditions import annotate
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic()
 out=P/'FULL_CHANNEL_CONTINUATION_EFFECT_V1.json';assert not out.exists()
 e={k:v.cuda().double() for k,v in torch.load(P/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True).items()};p={k:v.cuda() for k,v in torch.load(P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt',weights_only=True).items()}
 a,b=p['a'].double(),p['b'].double();c=e['q']@e['R_U']@p['writer'].double();student=a.T@(c[:,None]*b)+b.T@(c[:,None]*a);native=e['native_matrix'];rw=torch.linalg.solve(e['R_U'],e['writer'])
 cal=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);roots=[];inverses=[]
 for key,mean in [('n','mean_n'),('m','mean_m')]:
  x=cal[key].flatten(0,1).cuda().double()-e[mean];ev,V=torch.linalg.eigh(x.T@x/len(x));mask=ev>1e-10*ev.max();roots.append((V*(ev.clamp_min(0)*mask).sqrt())@V.T);inverses.append((V*torch.where(mask,ev.clamp_min(1e-30).rsqrt(),0))@V.T)
 matrices={'native':native,'student':student};modes={}
 for name,matrix in matrices.items():
  U,s,Vh=torch.linalg.svd(roots[0]@matrix@roots[1],full_matrices=False);modes[name]=(inverses[0]@U[:,0]*s[0].sqrt(),inverses[1]@Vh[0]*s[0].sqrt())
 tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);model=Bilin18TorchBackend.load('cuda').model.float();enc=tiktoken.get_encoding('gpt2');records=[];checks=[]
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 b16,b17=model.transformer.h[16],model.transformer.h[17]
 for domain,documents in tokens.items():
  for doc,row in enumerate(documents):
   cache=capture(model,row[None,:256].cuda());h=cache['h17'];z=F.rms_norm(h,(1152,));poly=b17.mlp(z)-b17.mlp.Down_bias;final=cache['final'];checks.append(float((h+poly+b17.mlp.Down_bias-final).norm()/final.norm()))
   replacement=((z@p['a'].T)*(z@p['b'].T))@p['writer'].T+z@p['linear'].T+p['bias'];states={'native':final.flatten(0,1)[16:255],'student':(final-poly+replacement).flatten(0,1)[16:255]}
   hd=h.double().flatten(0,1);md=(b17.lambdas[0]*(cache['m16']-b16.mlp.Down_bias)).double().flatten(0,1);scale=(hd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=((hd-md/2)/scale)[16:255]-e['mean_n'];m=(md/scale)[16:255]-e['mean_m'];labels=annotate(row.tolist(),enc)[16:255];targets=row[17:256].cuda();masks={'all':torch.ones(239,dtype=torch.bool,device='cuda')}
   for cohort in ('continuation','spaced_word'):masks[cohort]=torch.tensor([v[cohort] for v in labels],device='cuda',dtype=torch.bool)
   effects={};damage={}
   for name,state in states.items():
    base=logits(state);basece=F.cross_entropy(base,targets,reduction='none');scalars={'full':((n@matrices[name])*m).sum(1),'mode':(n@modes[name][0])*(m@modes[name][1])}
    for kind,scalar in scalars.items():
     changed=logits(state-(scalar[:,None]*rw).float());effect=(changed-base).double();effect-=effect.mean(1,keepdim=True);effects[name,kind]=effect;damage[name,kind]=F.cross_entropy(changed,targets,reduction='none')-basece
   for kind in ('full','mode'):
    for cohort,mask in masks.items():
     records.append(dict(domain=domain,document_index=doc,kind=kind,cohort=cohort,sites=int(mask.sum()),reference_energy=float(effects['native',kind][mask].square().sum()),error_energy=float((effects['student',kind][mask]-effects['native',kind][mask]).square().sum()),native_ce_sum=float(damage['native',kind][mask].sum()),student_ce_sum=float(damage['student',kind][mask].sum())))
 summary=[]
 for domain in tokens:
  for kind in ('full','mode'):
   for cohort in masks:
    rr=[r for r in records if (r['domain'],r['kind'],r['cohort'])==(domain,kind,cohort)];sites=sum(r['sites'] for r in rr);energy=sum(r['reference_energy'] for r in rr)
    if sites and energy:summary.append(dict(domain=domain,kind=kind,cohort=cohort,sites=sites,effect_error=(sum(r['error_energy'] for r in rr)/energy)**.5,native_removal_ce=sum(r['native_ce_sum'] for r in rr)/sites,student_removal_ce=sum(r['student_ce_sum'] for r in rr)/sites))
 pred=dict(pred_a_replay=max(checks)<1e-5,pred_b_full=all(r['effect_error']<.1 for r in summary if r['kind']=='full'),pred_c_mode=all(r['effect_error']<.1 for r in summary if r['kind']=='mode'))
 result=dict(predictions=pred,summary=summary,records=records,maximum_replay=max(checks),seconds=time.monotonic()-start,scope='Opened48doc panel diagnostic. Fixed observer and calibration-derived modes; each removal is measured against its own native or compressed background. Centered logit effects compared. Supplied native upstream midpoint ports and original normalization; not full extraction, fresh confirmation or automatic semantic identification.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
