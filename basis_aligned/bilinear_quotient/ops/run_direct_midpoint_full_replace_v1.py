#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_effect pred_c_replace
"""Full retained-path replacement, including omitted output directions.
pred_a_instrument self replacement exact and selected-program algebra replay<1e-8.
pred_b_effect 256output rank4 full removal/swap centered effect error<.3 both domains.
pred_c_replace 256output rank4 replacement CE increase<.05 both domains.
48 capture forwards; five candidates; reused panels; no native-data fitting.
"""
import os,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
OUTPUT_NAME='MIDPOINT_FULL_REPLACE_V1.json'
PANEL_PREFIX='SELECTIVE_CONFIRMATION'
DONOR_PREFIX='SELECTIVE_CONFIRMATION_DONORS'
EXTRA_CHANNEL_FILE=None
EXTRA_PRODUCT_FILE=None
SOURCE_CONTEXT_FAMILIES=False
OUTPUT_SPAN_CONTROLS=False
EXTRA_ORACLE_FILE=None
POSITION_BINS=False
ERROR_REFERENCE=None
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,candidates=['constant','projected64','projected256','program64','program256'])));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from logit_effect_partition import partition,position_partition,error_composition
 from midpoint_program import product_source_delta
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/OUTPUT_NAME;assert not out.exists()
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');invru=torch.linalg.inv(ru);L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();C=ru@b17.mlp.Down.weight.double();mu=torch.load(P/'MIDPOINT_NATIVE_V1.pt',weights_only=True)['stats']['calibration']['native']['mean'].cuda();programs={w:{k:v.cuda().double() for k,v in torch.load(P/f'MIDPOINT_COVERAGE_{w}_R4_V1.pt',weights_only=True).items()} for w in [64,256]};donors=torch.load(P/f'{DONOR_PREFIX}_V1.pt',weights_only=True);records=[];checks=[]
 extra={} if EXTRA_CHANNEL_FILE is None else {name:{k:v.cuda().double() for k,v in e.items()} for name,e in torch.load(P/EXTRA_CHANNEL_FILE,weights_only=True).items()}
 for e in extra.values():
  ids=e['indices'].long();checks.extend([float((e['L']-L[ids]).abs().max()),float((e['R']-R[ids]).abs().max())])
 product_extra={} if EXTRA_PRODUCT_FILE is None else {name:{k:v.cuda().double() for k,v in e.items()} for name,e in torch.load(P/EXTRA_PRODUCT_FILE,weights_only=True).items()}
 oracle_extra={} if EXTRA_ORACLE_FILE is None else {name:{k:v.cuda().double() for k,v in e.items()} for name,e in torch.load(P/EXTRA_ORACLE_FILE,weights_only=True).items()}
 if ERROR_REFERENCE is not None:assert 'product_'+next(iter(product_extra))==ERROR_REFERENCE
 source_metric=torch.load(P/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].cuda().double() if SOURCE_CONTEXT_FAMILIES else None
 mean_n=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)['n'].double().mean((0,1)).cuda() if SOURCE_CONTEXT_FAMILIES else None
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 for domain in ['fineweb','code']:
  tokens=torch.load(P/f'{PANEL_PREFIX}_{domain.upper()}_V1.pt',weights_only=True);states=[];teacher=[];normalized_inputs=[];preds={k:[] for k in ['constant','projected64','projected256','program64','program256']+['channel_'+name for name in extra]+['product_'+name for name in product_extra]}
  for row in tokens:
   c=capture(model,row[None,:256].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);s=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-m/2)/s;m=m/s;y=((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@C.T-mu;teacher.append(y@invru.T);states.append(c['final'].flatten(0,1));preds['constant'].append(torch.zeros_like(y))
   if SOURCE_CONTEXT_FAMILIES:normalized_inputs.append((n,m))
   for w,e in programs.items():
    scalar=((n@e['A'])*(m@e['B']))@e['readout']-e['offset'];reduced=scalar@e['reduced_writers'].T;preds[f'program{w}'].append(reduced@invru.T);preds[f'projected{w}'].append(((y@e['scalar_readers'])@e['reduced_writers'].T)@invru.T)
    if len(states)==1:checks.append(float((reduced@e['scalar_readers']-scalar).norm()/scalar.norm()))
   for name,e in extra.items():
    phi=(n@e['L'].T)*(m@e['R'].T)+(n@e['R'].T)*(m@e['L'].T)-e['channel_mean'];reduced=phi@e['reduced_writers'].T+e['full_mean']-mu;preds['channel_'+name].append(reduced@invru.T)
   for name,e in product_extra.items():
    left=((n@e['Pn'])@e['Tn']) if 'Pn' in e else n@e['A'];right=((m@e['Pm'])@e['Tm']) if 'Pm' in e else m@e['B'];products=left*right
    if 'base_left_mean' in e:products=(left-e['base_left_mean'])*(right-e['base_right_mean'])
    phi=products-e['product_mean']
    if 'group_writers' in e:
     groups=e['group_writers'].shape[1];reduced=phi.reshape(len(phi),groups,-1).sum(-1)@e['group_writers'].T+(phi@e['correction_left'])@e['correction_writers'].T
    elif 'output_basis' in e:reduced=(phi@e['output_core'].T)@e['output_basis'].T
    else:reduced=phi@e['reduced_writers'].T
    if 'centered_correction_left' in e:
     centered=products-left*e['right_mean']-right*e['left_mean']+e['left_mean']*e['right_mean'];reduced=reduced+(centered@e['centered_correction_left'])@e['centered_correction_writers'].T
    if 'linear_n' in e:reduced=reduced+n@e['linear_n']+m@e['linear_m']
    if 'linear_right_writer' in e:reduced=reduced+(n@e['linear_left_reader'])@e['linear_writer'].T+(m@e['linear_right_reader'])@e['linear_right_writer'].T
    elif 'linear_left_reader' in e:reduced=reduced+(n@e['linear_left_reader']+m@e['linear_right_reader'])@e['linear_writer'].T
    reduced=reduced+e['full_mean']-mu;preds['product_'+name].append(reduced@invru.T)
  states=torch.cat(states);true=torch.cat(teacher);preds={k:torch.cat(v) for k,v in preds.items()};targets=tokens[:,1:257].reshape(-1).cuda();flat=tokens[:,:256].flatten();mapping=donors[domain]['same_token'];valid=mapping>=0;docs=torch.arange(len(flat))//256;assert torch.all(docs[valid]!=docs[mapping[valid]]) and torch.all(flat[valid]==flat[mapping[valid]])
  for doc in range(len(tokens)):
   idx=torch.arange(doc*256+16,(doc+1)*256,device='cuda');state=states[idx];base=logits(state);basece=F.cross_entropy(base,targets[idx],reduction='none');checks.append(float((logits(state+(true[idx]-true[idx]).float())-base).abs().max()))
   for key,approx in preds.items():
    z=logits(state+(approx[idx]-true[idx]).float());records.append(dict(domain=domain,candidate=key,family='replacement',sites=len(idx),ce_added=float((F.cross_entropy(z,targets[idx],reduction='none')-basece).sum())))
   for family in ['removal','same_token']:
    ids=idx if family=='removal' else torch.where(valid&(docs==doc))[0].cuda();dst=mapping[ids.cpu()].cuda();state=states[ids];base=logits(state);delta=-true[ids] if family=='removal' else true[dst]-true[ids];reference=(logits(state+delta.float())-base).double()
    for key,approx in preds.items():
     delta=-approx[ids] if family=='removal' else approx[dst]-approx[ids];effect=(logits(state+delta.float())-base).double()
     if key==ERROR_REFERENCE:baseline_effect=effect
     composition=error_composition(reference,baseline_effect,effect) if ERROR_REFERENCE is not None and key.startswith('product_') else {}
     records.append(dict(domain=domain,candidate=key,family=family,sites=len(ids),error_composition=composition,position_bins=position_partition(reference,effect,ids%256) if POSITION_BINS and key.startswith('product_') else {},**partition(reference,effect)))
  if SOURCE_CONTEXT_FAMILIES:
   nn=torch.cat([z[0] for z in normalized_inputs]);mm=torch.cat([z[1] for z in normalized_inputs])
   for doc in range(len(tokens)):
    ids=torch.where(valid&(docs==doc))[0].cuda();dst=mapping[ids.cpu()].cuda();dm=mm[dst]-mm[ids];state=states[ids];base=logits(state)
    for family in ['source_only','context_only']:
     ni=nn[ids] if family=='source_only' else nn[ids]-mean_n
     delta=(((ni@L.T)*(dm@R.T)+(ni@R.T)*(dm@L.T))@C.T)@invru.T
     reference=(logits(state+delta.float())-base).double()
     for name,e in product_extra.items():
      approx=product_source_delta(e,ni,dm,context_only=family=='context_only')@invru.T
      effect=(logits(state+approx.float())-base).double()
      if 'product_'+name==ERROR_REFERENCE:baseline_effect=effect
      composition=error_composition(reference,baseline_effect,effect) if ERROR_REFERENCE is not None else {}
      records.append(dict(domain=domain,candidate='product_'+name,family=family,sites=len(ids),error_composition=composition,position_bins=position_partition(reference,effect,ids%256) if POSITION_BINS else {},linear_reference_energy=float(((delta@ru.T)@source_metric).square().sum()),linear_error_energy=float((((approx-delta)@ru.T)@source_metric).square().sum()),**partition(reference,effect)))
      if OUTPUT_SPAN_CONTROLS:
       target_red=delta@ru.T;approx_red=approx@ru.T;projected=(target_red@e['span_readers'])@e['span_writers'].T
       reference_energy=float((target_red@source_metric).square().sum());omitted=float(((target_red-projected)@source_metric).square().sum());inspace=float(((approx_red-projected)@source_metric).square().sum());total=float(((approx_red-target_red)@source_metric).square().sum())
       projected_effect=(logits(state+(projected@invru.T).float())-base).double()
       records.append(dict(domain=domain,candidate='span_'+name,family=family,sites=len(ids),linear_reference_energy=reference_energy,linear_error_energy=omitted,in_span_error_energy=inspace,program_error_energy=total,pythagorean_relative_error=abs(total-omitted-inspace)/max(total,1e-30),**partition(reference,projected_effect)))
     for name,e in oracle_extra.items():
      target_red=delta@ru.T;projected=(target_red@e['span_readers'])@e['span_writers'].T
      projected_effect=(logits(state+(projected@invru.T).float())-base).double()
      records.append(dict(domain=domain,candidate='oracle_'+name,family=family,sites=len(ids),linear_reference_energy=float((target_red@source_metric).square().sum()),linear_error_energy=float(((target_red-projected)@source_metric).square().sum()),**partition(reference,projected_effect)))
 summary={}
 for d in ['fineweb','code']:
  summary[d]={}
  for key in list(preds)+(['span_'+name for name in product_extra] if OUTPUT_SPAN_CONTROLS else [])+['oracle_'+name for name in oracle_extra]:
   result={}
   families=['source_only','context_only'] if key.startswith(('span_','oracle_')) else ['replacement','removal','same_token']+(['source_only','context_only'] if SOURCE_CONTEXT_FAMILIES and key.startswith('product_') else [])
   for family in families:
    rr=[x for x in records if x['domain']==d and x['candidate']==key and x['family']==family];sites=sum(x['sites'] for x in rr)
    if family=='replacement':result[family]=dict(ce_added=sum(x['ce_added'] for x in rr)/sites)
    else:
     en=sum(x['native_centered_effect_energy'] for x in rr);err=sum(x['centered_effect_error_energy'] for x in rr);result[family]=dict(centered_effect_relative_error=(err/en)**.5)
   summary[d][key]=result
 pred=dict(pred_a_instrument=max(checks)<1e-8,pred_b_effect=all(summary[d]['program256'][f]['centered_effect_relative_error']<.3 for d in summary for f in ['removal','same_token']),pred_c_replace=all(summary[d]['program256']['replacement']['ce_added']<.05 for d in summary));result=dict(predictions=pred,summary=summary,records=records,instrument=max(checks),seconds=time.perf_counter()-start,scope='Full source-dependent midpoint contribution centered on calibration mean. Replacement retains that mean but approximates all varying output coordinates. Residual background and true source states fixed; no upstream ablation or whole-model speed claim. Reused panels, no fitting.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
