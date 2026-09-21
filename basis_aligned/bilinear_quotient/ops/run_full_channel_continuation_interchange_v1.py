#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_full pred_c_mode
"""Frozen normalized-port interchange inside original/compressed backgrounds.
Self-swap scalar error<1e-10. Same-cohort full and mode effect errors<.10 in all
nonempty domain/cohort/edit cells; source/context/both swaps all retained.
Opposite-cohort is secondary due donor concentration. No fit, opened48docs.
Price unchanged3686products/14,067,072coefficients; swaps diagnostic only.
"""
import os,sys,json,time,hashlib
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
 out=P/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_V1.json';assert not out.exists()
 e={k:v.cuda().double() for k,v in torch.load(P/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True).items()};p={k:v.cuda() for k,v in torch.load(P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt',weights_only=True).items()}
 a,b=p['a'].double(),p['b'].double();c=e['q']@e['R_U']@p['writer'].double();student=a.T@(c[:,None]*b)+b.T@(c[:,None]*a);native=e['native_matrix'];rw=torch.linalg.solve(e['R_U'],e['writer'])
 cal=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);roots=[];inverses=[]
 for key,mean in [('n','mean_n'),('m','mean_m')]:
  x=cal[key].flatten(0,1).cuda().double()-e[mean];ev,V=torch.linalg.eigh(x.T@x/len(x));mask=ev>1e-10*ev.max();roots.append((V*(ev.clamp_min(0)*mask).sqrt())@V.T);inverses.append((V*torch.where(mask,ev.clamp_min(1e-30).rsqrt(),0))@V.T)
 matrices={'native':native,'student':student};modes={}
 for name,matrix in matrices.items():
  U,s,Vh=torch.linalg.svd(roots[0]@matrix@roots[1],full_matrices=False);modes[name]=(inverses[0]@U[:,0]*s[0].sqrt(),inverses[1]@Vh[0]*s[0].sqrt())

 tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);donors=json.loads((P/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_DONORS_V1.json').read_text());assert hashlib.sha256((P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt').read_bytes()).hexdigest()==donors['token_sha256']
 model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];records=[];checks=[]
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 for domain,documents in tokens.items():
  caches={'native':[],'student':[],'n':[],'m':[]}
  for row in documents:
   cache=capture(model,row[None,:256].cuda());h=cache['h17'];z=F.rms_norm(h,(1152,));poly=b17.mlp(z)-b17.mlp.Down_bias;final=cache['final'];replacement=((z@p['a'].T)*(z@p['b'].T))@p['writer'].T+z@p['linear'].T+p['bias']
   caches['native'].append(final.flatten(0,1)[16:255]);caches['student'].append((final-poly+replacement).flatten(0,1)[16:255]);hd=h.double().flatten(0,1);md=(b17.lambdas[0]*(cache['m16']-b16.mlp.Down_bias)).double().flatten(0,1);scale=(hd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();caches['n'].append(((hd-md/2)/scale)[16:255]-e['mean_n']);caches['m'].append((md/scale)[16:255]-e['mean_m'])
  caches={k:torch.cat(v) for k,v in caches.items()};cohorts=torch.tensor(donors['domains'][domain]['cohort_labels'],device='cuda');targets=documents[:,17:256].flatten().cuda()
  for family,mapping in donors['domains'][domain]['maps'].items():
   mapping=torch.tensor(mapping,device='cuda')
   for doc in range(len(documents)):
    ids=torch.arange(doc*239,(doc+1)*239,device='cuda');ids=ids[mapping[ids]>=0]
    if not len(ids):continue
    dst=mapping[ids];n,m=caches['n'][ids],caches['m'][ids];nd,md=caches['n'][dst],caches['m'][dst];bases={name:logits(caches[name][ids]) for name in matrices};ces={name:F.cross_entropy(base,targets[ids],reduction='none') for name,base in bases.items()}
    for kind in ('full','mode'):
     original={}
     for name in matrices:
      original[name]=((n@matrices[name])*m).sum(1) if kind=='full' else (n@modes[name][0])*(m@modes[name][1])
     for edit,nn,mm in [('source',n,md),('context',nd,m),('both',nd,md)]:
      effects={};damage={}
      for name in matrices:
       swapped=((nn@matrices[name])*mm).sum(1) if kind=='full' else (nn@modes[name][0])*(mm@modes[name][1]);delta=swapped-original[name];changed=logits(caches[name][ids]+(delta[:,None]*rw).float());effect=(changed-bases[name]).double();effect-=effect.mean(1,keepdim=True);effects[name]=effect;damage[name]=F.cross_entropy(changed,targets[ids],reduction='none')-ces[name]
       if edit=='source':
        selfswap=((n@matrices[name])*m).sum(1) if kind=='full' else (n@modes[name][0])*(m@modes[name][1]);checks.append(float((selfswap-original[name]).abs().max()))
      for cohort,label in [('continuation',1),('spaced_word',2)]:
       mask=cohorts[ids]==label
       if not mask.any():continue
       records.append(dict(domain=domain,family=family,document_index=doc,kind=kind,edit=edit,cohort=cohort,sites=int(mask.sum()),reference_energy=float(effects['native'][mask].square().sum()),student_energy=float(effects['student'][mask].square().sum()),error_energy=float((effects['student'][mask]-effects['native'][mask]).square().sum()),native_ce_sum=float(damage['native'][mask].sum()),student_ce_sum=float(damage['student'][mask].sum())))
 keys=('domain','family','kind','edit','cohort');summary=[]
 for values in sorted(set(tuple(r[k] for k in keys) for r in records)):
  rr=[r for r in records if tuple(r[k] for k in keys)==values];sites=sum(r['sites'] for r in rr);energy=sum(r['reference_energy'] for r in rr)
  summary.append(dict(zip(keys,values),sites=sites,effect_error=(sum(r['error_energy'] for r in rr)/energy)**.5 if energy else None,native_effect_rms=(energy/(sites*50304))**.5,native_ce_added=sum(r['native_ce_sum'] for r in rr)/sites,student_ce_added=sum(r['student_ce_sum'] for r in rr)/sites))
 primary=[r for r in summary if r['family']=='same_cohort'];pred=dict(pred_a_replay=max(checks)<1e-10,pred_b_full=all(r['effect_error'] is not None and r['effect_error']<.1 for r in primary if r['kind']=='full'),pred_c_mode=all(r['effect_error'] is not None and r['effect_error']<.1 for r in primary if r['kind']=='mode'))
 result=dict(predictions=pred,summary=summary,records=records,self_swap_max=max(checks),donor_stats={k:v['stats'] for k,v in donors['domains'].items()},seconds=time.monotonic()-start,scope='Opened panel, centered normalized midpoint ports swapped independently, native upstream context supplied. Same boundary cohort is not answer preservation; opposite cohort suffers sparse/concentrated donor support. Each program receives its own computed scalar delta in its own background. No fitting or semantic adoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('records','summary')}),flush=True)
if __name__=='__main__':main()
