#!/usr/bin/env python3
# BQGATE:48prefixes;432localresponses;120seconds.
"""pred_a constrained writer and response error identities<=1e-10.
pred_b rank<=256 achieves<=1%aggregate ownresponseerror everybranch/strength.
pred_c a pred_b rank also has maxpercontext<=5%ownresponseerror.
Price48nativeprefixes,432localresponses,120seconds; no text fitting or suffix.
"""
from pathlib import Path
from hashlib import sha256
import json,torch,sys,os,time,signal
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from fastload import load_model_fast
from live_crossfirst_prefix_v1 import prepare as live_prefix,assembled
from directional_mlp_response_context_v1 import prepare,evaluate
from protected_sparse_map_v1 import fit as sparse_fit
@torch.no_grad()
def main():
 files=json.loads((P/'UPSTREAM_SPARSE_LIVE_V1_BINDING.json').read_text())['files'];assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('48prefixes;432localresponses;120seconds');return
 assert not (P/'UPSTREAM_SPARSE_LIVE_V1_RESULT.json').exists();signal.alarm(120);torch.set_num_threads(2);tic=time.perf_counter()
 pr=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True);j=pr['mixed_map'].double();w=pr['direction'].double()
 fits={256:sparse_fit(j,w).cuda()};j=j.cuda();w=w.cuda();pr={k:v.cuda() for k,v in pr.items()}
 writer_errors=[float(((fit-j)@w).norm()/(j@w).norm()) for fit in fits.values()]
 model=load_model_fast().cuda().eval();weights=assembled.load_weights(model.state_dict(),'cuda');rows=json.loads((P/'THREE_TERM_FRESH_TEMPLATE_V1_ROWS.json').read_text())['rows'];cells=[];identity=0.
 for index,row in enumerate(rows):
  ids=torch.tensor([row['ids']],device='cuda');live=live_prefix(model,ids,weights);z=live['z9'].double();m=live['m9'].double()-model.transformer.h[9].mlp.Down_bias.double();context=prepare(z,m,pr)
  for rank,fit in fits.items():
   approximate=dict(context,Jz=z@fit.T,Jw=fit@w)
   for name,amplitude in [('child',live['child']),('remainder',live['remainder']),('parent',live['child']+live['remainder'])]:
    for strength in [-1,1,2]:
     a=strength*amplitude;ref=evaluate(a,context);pred=evaluate(a,approximate);rho=context['perpendicular_rms']+context['writer_rms']*(a-context['parallel']).square()
     expected=-a/rho*(z@(fit-j).T)+a.square()/(2*rho)*((fit-j)@w)
     identity=max(identity,float(((pred-ref)-expected).norm()/ref.norm().clamp_min(1e-30)))
     cells.append(dict(row=index,rank=rank,branch=name,strength=strength,error_squared=float((pred-ref).square().sum()),reference_squared=float(ref.square().sum()),relative_error=float((pred-ref).norm()/ref.norm().clamp_min(1e-30))))
 groups=[]
 for rank in fits:
  for name in ['child','remainder','parent']:
   for strength in [-1,1,2]:
    sub=[c for c in cells if c['rank']==rank and c['branch']==name and c['strength']==strength];groups.append(dict(rank=rank,branch=name,strength=strength,relative_error=(sum(c['error_squared'] for c in sub)/sum(c['reference_squared'] for c in sub))**.5,max_context_error=max(c['relative_error'] for c in sub)))
 passing=[r for r in fits if all(g['relative_error']<=.01 for g in groups if g['rank']==r)]
 result={'pred_a':max(writer_errors)<=1e-10 and identity<=1e-10,'pred_b':bool(passing),'pred_c':any(all(g['max_context_error']<=.05 for g in groups if g['rank']==r) for r in passing),'writer_errors':writer_errors,'error_identity_max':identity,'groups':groups,'cells':cells,'seconds':time.perf_counter()-tic,'scope':'Protected sparse weights-only J9 map at rank256 matched nominal bytes on48live contexts and9signed branch responses; reference exact response algebra with native baseline. Local input-domain validation, not suffix/behavioral prediction or independent fresh OOD.'}
 (P/'UPSTREAM_SPARSE_LIVE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
if __name__=='__main__':main()
