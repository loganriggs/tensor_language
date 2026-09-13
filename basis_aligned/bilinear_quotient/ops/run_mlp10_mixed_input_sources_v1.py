#!/usr/bin/env python3
# BQGATE:2240bodyforwards;160extraMLP10;160prefixes<=248tokens;180seconds.
"""pred_a previous11arm replay<=1e-4 andinputproductpartition<=1e-10relative.
pred_b residual/residual alone predicts fullMLPcross effect<=.10 eachregionalgroup.
pred_c separateRR/mixed/AA effects compose<=.10fullcross eachregionalgroup.
Null: attention10 individualinputchanges matter insideMLPproduct despiteits smallownmixedstate.
Price2240fullforwards160extraMLP10;160reusedprefixes;180seconds;no fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
from mlp_two_edit_mixed_v1 import decompose
from mlp_mixed_input_sources_v1 import split as split_sources
STEM='MLP10_MIXED_INPUT_SOURCES_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);rows=regional+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('2240forwards160extraMLP10;14arms;RR/mixed/AA inputproducts');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda();child={i:v.cuda() for i,v in torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'].items()};old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True);parent={i:v.cuda() for i,v in old['parent_fields'].items()};ctx={};measures=torch.zeros(160,14,2,dtype=torch.float64);statechecks=[];count=0
 mlpweights=[model.state_dict()['transformer.h.10.mlp.'+name+'.weight'].double() for name in ('Left','Right','Down')];formula_checks=[];input_partition_checks=[]
 def write9(module,args,output):
  arm=ctx['arm'];i=ctx['i']
  if arm==0:return output
  amplitude=child[i] if arm==1 else parent[i]-child[i] if arm==3 else parent[i]
  return output[0]-(amplitude[...,None]*w).to(output[0].dtype),output[1]
 def after9(module,args,output):
  arm=ctx['arm']
  if arm<4:ctx['h9'][arm]=output[0].double()
  if arm==4:
   s=ctx['h9'];return (s[1]+s[3]-s[0]).to(output[0].dtype),output[1]
  return output
 def pre10(module,args):
  ctx['raw10']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
  if ctx['arm']<4:ctx['raw10s'][ctx['arm']]=ctx['raw10'].double()
 def att10(module,args,output):
  if ctx['arm']<5:ctx['z10'][ctx['arm']]=(ctx['raw10']+output[0]).double()
 def after10(module,args,output):
  arm=ctx['arm'];x=output[0]
  if arm<4:ctx['h10'][arm]=x.double();return output
  if arm==4:
   z=ctx['z10'];s=ctx['h10'];bar=s[1]+s[3]-s[0];zbar=(z[1]+z[3]-z[0]).to(x.dtype)
   gbar=(zbar+module.mlp(F.rms_norm(zbar,(1152,)))).double();gm=gbar-bar;ga=x.double()-gbar;ctx['bar']=bar;ctx['gm']=gm;ctx['ga']=ga
   parts=decompose(z[0],z[1]-z[0],z[3]-z[0],*mlpweights);ctx['parts']=parts;formula_checks.append(dict(error_sq=float((parts['total']-gm).square().sum()),reference_sq=float(gm.square().sum())))
   raw=ctx['raw10s'];sources=split_sources(z[0],z[1]-z[0],z[3]-z[0],raw[1]-raw[0],raw[3]-raw[0],*mlpweights);ctx['sources']=sources;input_partition_checks.append(float((sources['total']-parts['cross']).norm()/parts['cross'].norm().clamp_min(1e-30)))
   total=x.double()-bar;statechecks.append(float((gm+ga-total).norm()/total.norm().clamp_min(1e-30)));return output
  target=ctx['bar']+(ctx['gm'] if arm==6 else ctx['ga'] if arm==7 else ctx['parts']['cross'] if arm==8 else ctx['parts']['normalization'] if arm==9 else ctx['parts']['total'] if arm==10 else ctx['sources']['residual_residual'] if arm==11 else ctx['sources']['residual_attention'] if arm==12 else ctx['sources']['attention_attention'] if arm==13 else 0)
  return target.to(x.dtype),output[1]
 handles=[model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[9].register_forward_hook(after9),model.transformer.h[10].register_forward_pre_hook(pre10),model.transformer.h[10].attn.register_forward_hook(att10),model.transformer.h[10].register_forward_hook(after10)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,h9={},h10={},z10={},raw10s={})
   for arm in range(14):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==2240 and len(statechecks)==160
 prior=torch.load(P/'MLP10_MIXED_PRODUCT_NORM_V1_ARTIFACT.pt',weights_only=True)['measures'];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[rel(measures[:96,:11],prior[:96]),rel(measures[96:,:11],prior[96:])]
 panel_errors=[(sum(c['error_sq'] for c in formula_checks[lo:hi])/sum(c['reference_sq'] for c in formula_checks[lo:hi]))**.5 for lo,hi in [(0,96),(96,160)]]
 cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=measures[lo:hi,:,0];full=z[:,4]-z[:,5];mlp=z[:,6]-z[:,5];att=z[:,7]-z[:,5];err=full-mlp-att
  cross=z[:,8]-z[:,5];norm=z[:,9]-z[:,5];formula=z[:,10]-z[:,5]
  rr=z[:,11]-z[:,5];mixed=z[:,12]-z[:,5];aa=z[:,13]-z[:,5]
  cells.append(dict(residual_only_error=rel(rr,cross),input_effect_composition_error=rel(rr+mixed+aa,cross),residual_aligned=float((rr*cross).sum()/cross.square().sum()),mixed_aligned=float((mixed*cross).sum()/cross.square().sum()),attention_squared_aligned=float((aa*cross).sum()/cross.square().sum()),formula_effect_error=rel(formula,mlp),cross_only_error=rel(cross,mlp),norm_only_error=rel(norm,mlp),cross_norm_composition_error=rel(cross+norm,formula),cross_aligned=float((cross*mlp).sum()/mlp.square().sum()),norm_aligned=float((norm*mlp).sum()/mlp.square().sum()),maxabs_formula_error=float((formula-mlp).abs().max()),cell=label,MLP_only_error=rel(mlp,full),attention_only_error=rel(att,full),composition_error=float(err.norm()/full.norm().clamp_min(1e-30)),MLP_aligned=float((mlp*full).sum()/full.square().sum().clamp_min(1e-30)),attention_aligned=float((att*full).sum()/full.square().sum().clamp_min(1e-30)),full_norm=float(full.norm()),maxabs_composition=float(err.abs().max())))
 result={'pred_a':max(replay+statechecks)<=1e-4 and max(input_partition_checks)<=1e-10,'pred_b':all(c['residual_only_error']<=.1 for c in cells[:4]),'pred_c':all(c['input_effect_composition_error']<=.1 for c in cells[:4]),'max_input_partition_error':max(input_partition_checks),'formula_state_errors':panel_errors,'anchor_replay':replay,'max_state_partition_error':max(statechecks),'cells':cells,'body_forwards':count,'extra_MLP10_evaluations':160,'seconds':time.perf_counter()-tic,'scope':'MLP10 cross product split byrawresidualreentry versusattention/add-rounding inputchanges. RR,mixed,AA sharejointRMSdenominator andadditiveblock10background. Actualsuffix11..17; reused160rows, no datafit ornewOOD.'}
 torch.save(dict(measures=measures,formula_checks=formula_checks),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
