"""Coefficient-regularized empirical fourth-moment readout screen at fixed graph topology."""
from pathlib import Path
import json,time,torch
from empirical_pair_metric import prepare,solve
from global_mixed_source_graph import export,score,source_reads
from shared_private_metric import SharedPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);p=torch.load(P/'COMPACT_GROUP_FROZEN_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);L=root@p['left_reader'];R=root@p['right_reader'];V=root@p['square_reader'];m=L.shape[1];T=d['teacher'];scales=d['scales'];x=(d['z'][:1536]-d['mu'])@d['inverse_root'];K=d['inverse_root']@d['old_covariance']@d['inverse_root'];z=d['z'][:1536];h=d['h'][:1536];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();trueq=torch.einsum('ni,oij,nj->no',z,torch.stack([q for pair in d['pairs'] for q in pair['Qs']]),z)
stats={};records=[];programs={}
for family in ['source','downstream']:
 stats[family]=[]
 for j,pair in enumerate(d['pairs']):
  A=(h@pair['a']-.5*trueq[:,2*j])/s-pair['alpha'];B=trueq[:,2*j+1]/s-pair['beta'];phi=A*B;std=(phi-phi.mean()).square().mean().sqrt()
  ca=-.5*B/s*scales[2*j]/std;cb=A/s*scales[2*j+1]/std
  LB=torch.cat([L,V],1) if j==2 else L;RB=torch.cat([R,V],1) if j==2 else R
  stats[family].append(prepare(T[2*j],T[2*j+1],L,R,LB,RB,x,K,ca,cb,family))
 for lam in [0,.01,.1,1,10]:
  W=torch.zeros((6,m),dtype=T.dtype);private=torch.zeros(V.shape[1],dtype=T.dtype);res=[];emp=[]
  for j,a in enumerate(stats[family]):
   w,ridge,replay=solve(a,lam);res.append(replay);W[2*j]=w[:m];W[2*j+1]=w[m:2*m]
   if j==2:private=w[2*m:]
   emp.append(float((a['design']@w-a['target']).square().sum()/a['divisor']))
  extra=torch.zeros((6,len(private)),dtype=T.dtype);extra[5]=private;big=export(torch.cat([L,V],1),torch.cat([R,V],1),torch.cat([W,extra],1),d);compiled={k:v.clone() for k,v in big.items()}
  for n in ['left_reader','right_reader']:compiled[n]=big[n][:,:m].clone()
  compiled['product_weights']=big['product_weights'][:m].clone();compiled['square_reader']=big['left_reader'][:,m:].clone();compiled['square_weights']=big['product_weights'][m:,5].clone();compiled['square_output']=torch.tensor(5)
  floats=sum(v.numel() for v in compiled.values() if v.is_floating_point());assert floats==896198
  a=source_reads(z[:32],compiled);a[:,5]+=(z[:32]@compiled['square_reader']).square()@compiled['square_weights'];b=source_reads(z[:32],big);replay=float((a-b).norm()/b.norm());assert max(replay,*res)<1e-8
  metric=SharedPrivateMetric(T,torch.eye(6,dtype=T.dtype));hat=metric.dense(L,R,V,W,private);coef=float((hat-T).norm()/T.norm());train=dict(d);train['indices']=torch.arange(1536);training=score(big,train);opened=score(big,d)
  if lam==0:
   _,controlW,controlv=metric.loss(L,R,V);control=metric.dense(L,R,V,controlW,controlv);controlreplay=float((hat-control).norm()/control.norm());assert controlreplay<1e-7
  else:controlreplay=None
  key=f'{family}_{lam}';programs[key]=compiled;records.append(dict(key=key,family=family,lam=lam,original_coefficient_error=coef,empirical_objective_terms=emp,normal_equation_replay=max(res),execution_replay=replay,coefficient_control_replay=controlreplay,training=training,opened=opened,source_products=399,stored_floats=floats));print(key,coef,training['per_mode_errors'],opened['per_mode_errors'],flush=True)
selected=next(r for r in records if r['key']=='downstream_0.1');control=next(r for r in records if r['key']=='downstream_0');baseline=[.03058409729022641,.027558449717507608,.11941807478056159]
out=dict(records=records,primary='downstream_0.1',predictions=dict(pred_a_instrument=max(max(r['normal_equation_replay'],r['execution_replay']) for r in records)<1e-8,pred_b_component_values=all(a<=.15 and a<=1.10*b for a,b in zip(selected['opened']['per_mode_errors'],baseline)),pred_c_coefficient_fidelity=selected['original_coefficient_error']<=1.1*control['original_coefficient_error']),seconds=time.perf_counter()-start,scope='Fixed input directions and399product graph; actual empirical quadratic features on1536training sites,448opened diagnostics, no newdata or model forwards. Source loss and coupled linearized component loss each added to original isotropic-output coefficient loss. Original geometry baseline differs from parent half-output weighting and is explicitly recomputed. No fresh/OOD adoption.')
torch.save(programs,P/'EMPIRICAL_PAIR_READOUT_PROGRAMS_V1.pt');(P/'EMPIRICAL_PAIR_READOUT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'])
