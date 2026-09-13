#!/usr/bin/env python3
# BQGATE:2880bodyforwards;160extraMLP10;160prefixes<=248tokens;180seconds.
"""pred_a original17armreplay<=1e-4; sixRR/joint tensor replay<=1e-10.
pred_b six-product joint target-effect error<=.01 each regional group vs old folded joint.
pred_c regional target/control maxabsolute change<=1e-5 and zero material sign reversals.
Null: exact algebra suffers native numerical amplification or incorrect integration.
Price2880bodyforwards160extraMLP10;160historicalprefixes;180seconds; nofit.
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
from mlp9_to_mlp10_residual_fold_v1 import execute as fold_residual
from response_product_basis_v2 import prepare_basis, coefficients, prepare_products, combine
STEM='RESPONSE_PRODUCT_BASIS_V2_NATIVE'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());regional=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];validate(regional);rows=regional+json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==160
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('2880forwards160extraMLP10;18arms;joint+MLP9fold');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();w=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)['direction'].cuda();child={i:v.cuda() for i,v in torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'].items()};old=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True);parent={i:v.cuda() for i,v in old['parent_fields'].items()};ctx={};measures=torch.zeros(160,18,2,dtype=torch.float64);statechecks=[];count=0
 mlpweights=[model.state_dict()['transformer.h.10.mlp.'+name+'.weight'].double() for name in ('Left','Right','Down')];formula_checks=[];input_partition_checks=[]
 response_program={k:v.cuda() for k,v in torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True).items()};fold_checks=[];six_checks=[]
 def before9(module,args):
  if ctx['arm']==0:ctx['raw9']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def mlp9out(module,args,output):
  if ctx['arm']==0:ctx['m9']=output.double()-module.Down_bias.double()
 def write9(module,args,output):
  arm=ctx['arm'];i=ctx['i']
  if arm==0:
   ctx['z9']=(ctx['raw9']+output[0]).double();return output
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
   i=ctx['i'];fc,fr=fold_residual(ctx['z9'],ctx['m9'],child[i][...,None],(parent[i]-child[i])[...,None],response_program,float(module.lambdas[0]));cr=raw[1]-raw[0];rr=raw[3]-raw[0];ca=z[1]-z[0]-cr;ra=z[3]-z[0]-rr;rho=(z[0]+(z[1]-z[0])+(z[3]-z[0])).square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
   def cross(x,y):
    L,R,D=mlpweights;return ((x@L.T)*(y@R.T)+(y@L.T)*(x@R.T))@D.T/rho
   ctx['folded_joint']=cross(fc,fr)+cross(fc,ra)+cross(ca,fr);ctx['folded_full']=ctx['folded_joint']+sources['attention_attention'];fold_checks.append(dict(error_sq=float((fc-cr).square().sum()+(fr-rr).square().sum()),reference_sq=float(cr.square().sum()+rr.square().sum())))
   context,basis=prepare_basis(ctx['z9'],ctx['m9'],response_program,float(module.lambdas[0]));uc=coefficients(child[i][...,None],context);ur=coefficients((parent[i]-child[i])[...,None],context)
   newfc=torch.einsum('...k,...kd->...d',uc,basis);newfr=torch.einsum('...k,...kd->...d',ur,basis)
   pairids,bank=prepare_products(basis,*mlpweights);sixrr=combine(uc,ur,pairids,bank)/rho
   ctx['six_joint']=sixrr+cross(newfc,ra)+cross(ca,newfr)
   six_checks.append(dict(rr_error=float((sixrr-cross(fc,fr)).norm()/cross(fc,fr).norm().clamp_min(1e-30)),joint_error=float((ctx['six_joint']-ctx['folded_joint']).norm()/ctx['folded_joint'].norm().clamp_min(1e-30))))
   total=x.double()-bar;statechecks.append(float((gm+ga-total).norm()/total.norm().clamp_min(1e-30)));return output
  target=ctx['bar']+(ctx['gm'] if arm==6 else ctx['ga'] if arm==7 else ctx['parts']['cross'] if arm==8 else ctx['parts']['normalization'] if arm==9 else ctx['parts']['total'] if arm==10 else ctx['sources']['residual_residual'] if arm==11 else ctx['sources']['residual_attention'] if arm==12 else ctx['sources']['attention_attention'] if arm==13 else ctx['sources']['residual_residual']+ctx['sources']['residual_attention'] if arm==14 else ctx['folded_joint'] if arm==15 else ctx['folded_full'] if arm==16 else ctx['six_joint'] if arm==17 else 0)
  return target.to(x.dtype),output[1]
 handles=[model.transformer.h[9].register_forward_pre_hook(before9),model.transformer.h[9].mlp.register_forward_hook(mlp9out),model.transformer.h[9].attn.register_forward_hook(write9),model.transformer.h[9].register_forward_hook(after9),model.transformer.h[10].register_forward_pre_hook(pre10),model.transformer.h[10].attn.register_forward_hook(att10),model.transformer.h[10].register_forward_hook(after10)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,h9={},h10={},z10={},raw10s={})
   for arm in range(18):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1
    if i<96:measures[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();measures[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    else:measures[i,arm,0]=-logits.log_softmax(-1)[198].cpu();measures[i,arm,1]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==2880 and len(statechecks)==160
 prior=torch.load(P/'MLP10_MIXED_INPUT_SOURCES_V1_ARTIFACT.pt',weights_only=True)['measures'];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));replay=[rel(measures[:96,:14],prior[:96]),rel(measures[96:,:14],prior[96:])]
 panel_errors=[(sum(c['error_sq'] for c in formula_checks[lo:hi])/sum(c['reference_sq'] for c in formula_checks[lo:hi]))**.5 for lo,hi in [(0,96),(96,160)]]
 fold_errors=[(sum(c['error_sq'] for c in fold_checks[lo:hi])/sum(c['reference_sq'] for c in fold_checks[lo:hi]))**.5 for lo,hi in [(0,96),(96,160)]]
 cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  z=measures[lo:hi,:,0];full=z[:,4]-z[:,5];mlp=z[:,6]-z[:,5];att=z[:,7]-z[:,5];err=full-mlp-att
  cross=z[:,8]-z[:,5];norm=z[:,9]-z[:,5];formula=z[:,10]-z[:,5]
  rr=z[:,11]-z[:,5];mixed=z[:,12]-z[:,5];aa=z[:,13]-z[:,5]
  joint=z[:,14]-z[:,5];folded=z[:,15]-z[:,5];foldfull=z[:,16]-z[:,5]
  cells.append(dict(joint_effect_error=rel(joint,cross),folded_joint_preservation_error=rel(folded,joint),folded_full_effect_error=rel(foldfull,cross),residual_only_error=rel(rr,cross),input_effect_composition_error=rel(rr+mixed+aa,cross),residual_aligned=float((rr*cross).sum()/cross.square().sum()),mixed_aligned=float((mixed*cross).sum()/cross.square().sum()),attention_squared_aligned=float((aa*cross).sum()/cross.square().sum()),formula_effect_error=rel(formula,mlp),cross_only_error=rel(cross,mlp),norm_only_error=rel(norm,mlp),cross_norm_composition_error=rel(cross+norm,formula),cross_aligned=float((cross*mlp).sum()/mlp.square().sum()),norm_aligned=float((norm*mlp).sum()/mlp.square().sum()),maxabs_formula_error=float((formula-mlp).abs().max()),cell=label,MLP_only_error=rel(mlp,full),attention_only_error=rel(att,full),composition_error=float(err.norm()/full.norm().clamp_min(1e-30)),MLP_aligned=float((mlp*full).sum()/full.square().sum().clamp_min(1e-30)),attention_aligned=float((att*full).sum()/full.square().sum().clamp_min(1e-30)),full_norm=float(full.norm()),maxabs_composition=float(err.abs().max())))
 result={'legacy_a':max(replay+statechecks)<=1e-4 and max(input_partition_checks)<=1e-10,'legacy_b':all(c['joint_effect_error']<=.01 for c in cells[:4]),'legacy_c':max(fold_errors)<=.01 and all(c['folded_joint_preservation_error']<=.01 for c in cells[:4]),'folded_input_errors':fold_errors,'max_input_partition_error':max(input_partition_checks),'formula_state_errors':panel_errors,'anchor_replay':replay,'max_state_partition_error':max(statechecks),'cells':cells,'body_forwards':count,'extra_MLP10_evaluations':160,'seconds':time.perf_counter()-tic,'scope':'MLP9 fixedwriterresponse generates residualinputs toMLP10 RR+mixedproduct. Nativeattentionpartners,jointnormandbackground retained. JointRR+mixed physicallyscored;160reusedrows, no fit ornewOOD.'}
 oldfull=torch.load(P/'MLP9_TO_MLP10_RESIDUAL_FOLD_V1_ARTIFACT.pt',weights_only=True)['measures'];six_cells=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  ref=measures[lo:hi,15]-measures[lo:hi,5];candidate=measures[lo:hi,17]-measures[lo:hi,5]
  six_cells.append(dict(cell=label,target_effect_error=rel(candidate[:,0],ref[:,0]),control_effect_error=rel(candidate[:,1],ref[:,1]),max_absolute_error=float((candidate-ref).abs().max()),material_sign_reversals=int(((candidate[:,0]*ref[:,0]<0)&(ref[:,0].abs()>=1e-5)).sum())))
 result['legacy_predictions']={k:result.pop('legacy_'+k) for k in ['a','b','c']}
 result.update({'pred_a':rel(measures[:,:17],oldfull)<=1e-4 and max(c['rr_error'] for c in six_checks)<=1e-10 and max(c['joint_error'] for c in six_checks)<=1e-10,
  'pred_b':all(c['target_effect_error']<=.01 for c in six_cells[:4]),'pred_c':all(c['max_absolute_error']<=1e-5 and c['material_sign_reversals']==0 for c in six_cells[:4]),
  'original17_replay':rel(measures[:,:17],oldfull),'six_product_checks':six_checks,'six_product_cells':six_cells,
  'scope':'Six-product RR and three-vector residual response installed jointly with native attention partners. Original normalizers, background and suffix retained. Historical160prefixes, no new OOD, no data fitting.'})
 torch.save(dict(measures=measures,formula_checks=formula_checks,fold_checks=fold_checks),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
