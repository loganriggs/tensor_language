"""All-output cross-domain audit consuming the immutable removal-state cache."""
import argparse,hashlib,json,time
from collections import defaultdict
import torch
from audit_conditional_residual_accounting import P,SCALE,load
from audit_root_matched_reader import CK
from conditional_quartic_cp import evaluate as full
from lean_conditional_cp import evaluate as lean

def cp(p,x):
 return torch.stack([x@f.T for f in p['factors']]).prod(0)@p['coefficients'].T

def teacher():
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
 def w(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].double()
 a,b,d=w(16,'Left'),w(16,'Right'),w(16,'Down');l,r,e=w(17,'Left'),w(17,'Right'),w(17,'Down');lam=state['transformer.h.17.lambdas'][0].double()
 W=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double();U=state['lm_head.weight'].double();UW=U@W;readers=U.T@UW/UW.square().sum(0);out=e.T@readers/SCALE
 def evaluate(x):
  m=((x@a.T)*(x@b.T))@(lam*d).T
  return ((m@l.T)*(m@r.T))@out
 return evaluate

def metrics(pred,y):
 e=(pred-y).square().sum(0);ref=y.square().sum(0);valid=ref>0
 return dict(error_energy=e.tolist(),reference_energy=ref.tolist(),pooled=float((e.sum()/ref.sum()).sqrt()) if ref.sum()>0 else None,per_output=[float((e[i]/ref[i]).sqrt()) if valid[i] else None for i in range(16)])

def check(native):
 x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'][:64].double()
 y=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'][:64].double()/SCALE
 actual=native(x);err=float((actual-y).norm()/y.norm());assert err<1e-5
 return dict(old_64_state_native_replay=err,outputs=actual.shape[1])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 cachepath=P/'REMOVAL_STAGE_STATES_V1.pt'
 if not args.check and not cachepath.exists():raise SystemExit('Pending dependency: REMOVAL_STAGE_STATES_V1.pt; do not recapture or alter queued runner.')
 frozen=json.loads((P/'CROSS_DOMAIN_OUTPUTS_INPUTS_V1.json').read_text())
 for name,sha in frozen.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha,name
 native=teacher();checks=check(native)
 if args.check:
  (P/'CROSS_DOMAIN_OUTPUTS_PREFLIGHT_V1.json').write_text(json.dumps(dict(**checks,seconds=time.monotonic()-start),indent=2)+'\n');print(checks);return
 receipt=json.loads((P/'REMOVAL_STAGE_GEOMETRY_V1.json').read_text());assert hashlib.sha256(cachepath.read_bytes()).hexdigest()==receipt['cache_sha256']
 cache=torch.load(cachepath,weights_only=True);programs=[]
 for kind,stem,fn in [('parent','MIXED_CP_FEATURES',cp),('full','CONDITIONAL_CP',full),('lean','LEAN_CONDITIONAL_CP',lean)]:
  for seed in [1001,1002]:
   p,sha=load(f'{stem}_SEED{seed}'+('' if kind=='parent' else '_RANK256')+'_V1.pt');programs.append((kind,seed,p,sha,fn))
 rows=[];replays=[]
 for domain in sorted({r['domain'] for r in cache['rows']}):
  docs=sorted([r for r in cache['rows'] if r['domain']==domain],key=lambda r:r['document']);assert len(docs)==16
  x=torch.cat([r['x'].double() for r in docs]);y=torch.cat([native(xx) for xx in x.split(256)])
  cached=torch.cat([r['exact'].double() for r in docs])/SCALE;replay=float((y[:,1]-cached).norm()/cached.norm());assert replay<1e-4;replays.append(replay)
  groups=defaultdict(list);offset=0
  for row in docs:
   assert len(row['x'])==239
   for position,token in enumerate(row['tokens'].tolist()):groups[(position,token)].append(offset+position)
   offset+=239
  pairs=[(ids[j],ids[j+1]) for ids in groups.values() for j in range(0,len(ids)-1,2)]
  for kind,seed,p,sha,fn in programs:
   pred=torch.cat([fn(p,xx)/SCALE for xx in x.split(1024)]);m=metrics(pred,y);document=[metrics(a,b) for a,b in zip(pred.split(239),y.split(239))]
   response=None
   if pairs:
    rec,don=torch.tensor(pairs).T;response=metrics(pred[don]-pred[rec],y[don]-y[rec])
   rows.append(dict(domain=domain,kind=kind,seed=seed,sha256=sha,states=len(x),documents=16,value=m,document_values=document,matched_pairs=len(pairs),pairs=pairs,response=response))
   print(domain,kind,seed,m['pooled'],len(pairs),flush=True)
 domains=sorted({r['domain'] for r in rows});fine=next(d for d in domains if 'fine' in d.lower());other=next(d for d in domains if d!=fine)
 parents=[r for r in rows if r['kind']=='parent']
 pred_transfer=all(next(r for r in parents if r['seed']==seed and r['domain']==other)['value']['pooled']<=1.25*next(r for r in parents if r['seed']==seed and r['domain']==fine)['value']['pooled'] for seed in [1001,1002])
 pred_small=all(v is not None and v<=.3 for r in parents for v in r['value']['per_output'][4:])
 result=dict(predictions=dict(parent_domain_retention=pred_transfer,parent_small_outputs=pred_small),rows=rows,checks=checks,coordinate1_cache_replays=replays,cache_sha256=hashlib.sha256(cachepath.read_bytes()).hexdigest(),seconds=time.monotonic()-start,scope='Already opened FineWeb/stdlib documents, all 16 selected quartic outputs; no fitting, semantic claim, or untouched OOD confirmation. Same-token same-position disjoint document pairs; no significance claim from individual correlated states.')
 (P/'CROSS_DOMAIN_OUTPUTS_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
