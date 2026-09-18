"""Fresh native-weight value mediation with independent MLP8 reference and16nulls."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
from mlp8_current_value_response_v1 import execute
STEM='CITY_VALUE_MEDIATION_FRESH_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 audit=json.loads((P/(STEM+'_ROW_AUDIT.json')).read_text());assert audit['passes'] and audit['row_sha256']==hashlib.sha256((P/(STEM+'_ROWS.json')).read_bytes()).hexdigest()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('800sequence equivalents;360block calls;20arms;CPU fresh value mediation');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(600)
 out=P/(STEM+'_RESULT.json');assert not out.exists()
 from fastload import load_model_fast
 model=load_model_fast().eval();program=torch.load(P/'CITY_FINEWEB_MLP8_VALUE_FOLD_V1_PROGRAM.pt',weights_only=True)
 ids=torch.tensor([r['ids'] for r in groups]);assert ids.shape==(40,32)
 city=groups[0]['city_position'];masks=torch.zeros(40,32,dtype=torch.bool)
 for i,row in enumerate(groups):masks[i,row['destination_positions']]=True;assert row['city_position']==city
 state={'arm':0};cache={};contexts={};changes={};errors=[];other_errors=[];norm_errors=[]
 attn8=model.transformer.h[8].attn;original8=attn8.squared_attention
 def capture8(module,q,k,v,q2,k2):
  if state['arm']==0:
   donor=torch.arange(40)^1
   route=((q[:,:,2]*k[:,city,None,2]).sum(-1)/128)*((q2[:,:,2]*k2[:,city,None,2]).sum(-1)/128)
   dr=((q[:,:,2]*k[donor,city,None,2]).sum(-1)/128)*((q2[:,:,2]*k2[donor,city,None,2]).sum(-1)/128)
   cache['edit']=F.linear(dr[...,None]*v[donor,city,None,2]-route[...,None]*v[:,city,None,2],module.c_proj.weight[:,256:384])*masks[...,None]
  return original8(q,k,v,q2,k2)
 attn8.squared_attention=MethodType(capture8,attn8)
 def pre8(module,args):
  if state['arm']<2:contexts[(state['arm'],'mixed8')]=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone()
 def post8(module,args,out):return (out[0]+cache['edit'],out[1]) if state['arm'] else out
 def postz(module,args,out):
  if state['arm']<2:contexts[(state['arm'],'z')]=(contexts[(state['arm'],'mixed8')]+out[0]).clone()
 def mlp8(module,args,out):
  if state['arm']<2:contexts[(state['arm'],'mlp8')]=out.clone()
 def pre9(module,args):
  if state['arm']<2:contexts[(state['arm'],'mixed9')]=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone()
 handles=[model.transformer.h[8].register_forward_pre_hook(pre8),attn8.register_forward_hook(post8),attn8.register_forward_hook(postz),model.transformer.h[8].mlp.register_forward_hook(mlp8),model.transformer.h[9].register_forward_pre_hook(pre9)]
 attn9=model.transformer.h[9].attn;original9=attn9.squared_attention
 def changed(module,q,k,v,q2,k2):
  arm=state['arm']
  if arm>=2:
   new=v.clone();new[:,:,8]-=changes[arm]
   actual=v[:,:,8]-new[:,:,8];errors.append(float((actual.double()-changes[arm].double()).norm()/changes[arm].double().norm()))
   other_errors.append(float((new[:,:,:8]-v[:,:,:8]).abs().max()));v=new
  return original9(q,k,v,q2,k2)
 attn9.squared_attention=MethodType(changed,attn9)
 values=torch.zeros(20,40,10,dtype=torch.float64);calls=0;fixtures=[];local_errors=[];outside=None
 try:
  for arm in range(20):
   state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
   if arm==1:
    inputs={'z':contexts[(0,'z')],'delta':cache['edit'],'mixed9_native':contexts[(0,'mixed9')],'mixed9_edited':contexts[(1,'mixed9')]}
    pieces=execute(program,**inputs);candidate=pieces[1:5].sum(0)
    rho=(contexts[(1,'mixed9')].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
    native=(1-attn9.lamb.double())*model.transformer.h[9].lambdas[0].double()/rho*F.linear(contexts[(1,'mlp8')].double()-contexts[(0,'mlp8')].double(),attn9.c_v.weight[1024:1152].double())
    local_errors=[float((c-n).norm()/n.norm()) for c,n in zip(candidate,native)]
    changes[2]=native.float();changes[3]=candidate.float()
    for k in range(16):
     null=[]
     for i,row in enumerate(groups):
      gen=torch.Generator().manual_seed(18102000+1000*k+row['context_id']);random=torch.randn(32,128,generator=gen).double();norm=candidate[i].norm(dim=-1,keepdim=True)
      d=(random*norm/random.norm(dim=-1,keepdim=True)*(1 if row['cue']=='British' else -1)).float()
      norm_errors.append(float(((d.double().norm(dim=-1,keepdim=True)-norm).abs()/norm.clamp_min(1e-8)).max()));null.append(d)
     changes[4+k]=torch.stack(null)
    outside=max(float(d[~masks].abs().max()) for d in changes.values())
    preflight={'passes':max(local_errors)<=1e-4 and outside==0,'max_local_error':max(local_errors),'local_errors':local_errors,'outside':outside}
    (P/(STEM+'_PREFLIGHT.json')).write_text(json.dumps(preflight,indent=2)+'\n');assert preflight['passes'],preflight
    fixtures=[{'inputs':{k:v[i:i+1] for k,v in inputs.items()},'native_value_delta':native[i:i+1],'candidate_value_delta':candidate[i:i+1]} for i in range(40)]
   print(f'arm{arm+1}/20 elapsed{time.perf_counter()-start:.2f}s',flush=True)
 finally:
  for h in handles:h.remove()
  attn8.squared_attention=original8;attn9.squared_attention=original9
 v=expand(values,mapping,6);total=v-v[:1];correction=v-v[1:2];rms=correction.square().mean(1).sqrt();total_rms=total.square().mean(1).sqrt()
 reference=correction[2,:,0];candidate=correction[3,:,0];prediction=float((candidate-reference).norm()/reference.norm())
 base=v[0,:,0][::2]-v[0,:,0][1::2];cap=base>=.1;records={}
 for arm,name in [(1,'swap'),(2,'native_corrected'),(3,'folded_corrected')]:
  pair=v[arm,:,0][::2]-v[arm,:,0][1::2];atten=(base[cap]-pair[cap])/base[cap]
  records[name]={'positive_fraction':float((atten>0).double().mean()),'mean_attenuation':float(atten.mean()),'total_target_rms':float(total_rms[arm,0]),'total_control_over_target':(total_rms[arm,1:]/total_rms[arm,0]).tolist()}
 null=rms[4:,0];ordered=null.sort().values;median=float((ordered[7]+ordered[8])/2)
 documents=[];reversed_docs=[]
 for context in range(20):
  idx=[i for i,r in enumerate(rows) if r['context_id']==context];b=v[0,idx,0][::2]-v[0,idx,0][1::2];capable=b>=.1
  a={name:((b[capable]-(v[arm,idx,0][::2]-v[arm,idx,0][1::2])[capable])/b[capable]).tolist() for arm,name in [(1,'swap'),(2,'native_corrected'),(3,'folded_corrected')]}
  if any(x<0 for x in a['swap']):reversed_docs.append(context)
  documents.append({'context_id':context,'capable_pairs':int(capable.sum()),'attenuations':a})
 strata={}
 for label,idx in [('reversed',[i for i,r in enumerate(rows) if r['context_id'] in reversed_docs]),('other',[i for i,r in enumerate(rows) if r['context_id'] not in reversed_docs]),('untouched',[i for i,r in enumerate(rows) if r['is_untouched_natural_arm']]),('substituted',[i for i,r in enumerate(rows) if not r['is_untouched_natural_arm']])]:
  if not idx:strata[label]={'rows':0};continue
  rr=correction[:,idx].square().mean(1).sqrt();strata[label]={'rows':len(idx),'prediction_error':float((candidate[idx]-reference[idx]).norm()/reference[idx].norm()),'correction_control_over_target':(rr[3,1:]/rr[3,0]).tolist(),'nulls_beaten':int((rr[4:,0]<rr[3,0]).sum())}
 r={'pred_a':audit['passes'] and len(rows)==240 and max(local_errors)<=1e-4 and outside==0 and max(norm_errors)<=1e-5 and max(other_errors)==0 and max(errors)<=1e-5 and bool(torch.isfinite(v).all()) and calls==360 and not torch.cuda.is_initialized(),
 'pred_b':prediction<=.05 and float(rms[3,0])>=1e-5 and int(cap.sum())>=90,
 'pred_c':bool((rms[3,1:]<=.5*rms[3,0]).all()) and bool((total_rms[3,1:]<=.5*total_rms[3,0]).all()),
 'pred_d':records['folded_corrected']['positive_fraction']>=.90 and records['folded_corrected']['mean_attenuation']>=.02,
 'pred_e':float(rms[3,0])>=2*median and bool((null<rms[3,0]).all()),
 'prediction_error':prediction,'arms':records,'capable_pairs':int(cap.sum()),'correction_target_rms':float(rms[3,0]),'correction_control_over_target':(rms[3,1:]/rms[3,0]).tolist(),'target_over_null_median':float(rms[3,0])/median,'nulls_beaten':int((null<rms[3,0]).sum()),'null_readout_rms':rms[4:].tolist(),'documents':documents,'strata':strata,'max_local_error':max(local_errors),'max_null_norm_error':max(norm_errors),'max_subtraction_relative_error':max(errors),'other_head_error':max(other_errors),'outside':outside,'body_forwards':800,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Fresh FineWeb conditional MLP8→head9.8 value mediation; exact native upstream swap, native normalization contexts and full suffix. Not packed-prefix confirmation, independent composition or token-only extraction.','source_shas':binding}
 torch.save({'values':v,'fixtures':fixtures},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['source_shas','documents','strata','null_readout_rms']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
