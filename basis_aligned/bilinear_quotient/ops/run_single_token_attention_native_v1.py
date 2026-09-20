#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fidelity pred_c_selectivity pred_d_capability
"""Prospective frozen source-column transfer; SINGLE_TOKEN_ATTENTION_NATIVE_V1_PREREGISTRATION.md."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'single_token_attention_native_v1_result.json';PLAN=dict(prefix=18,native_suffix=48,baseline_attention=6)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import numpy as np
 import torch,tiktoken
 import torch.nn.functional as F
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_write
 sys.path.insert(0,str(P))
 from refined_native_sources import build_refined,PARENTS
 from native_source_observables import source_observables32
 from native_single_token_attention import prepare,evaluate
 from shared_selective_source_lp import choose
 binding_path=P/'FIXED_QUERY_FRESH_V4_BINDING.json';binding=json.loads(binding_path.read_text())
 for file,digest in binding.items():assert hashlib.sha256((P/file).read_bytes()).hexdigest()==digest
 banks=torch.load(P/'RESIDUAL_ROLE_BANK_FROZEN_V1.pt',map_location='cpu',weights_only=True)['role_banks']
 oldbinding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());prior=json.loads((A/'refined_selective_sources_v1_result.json').read_text());installed=json.loads((A/'fixed_query_native_install_v1_result.json').read_text())
 contexts=[dict(dataset='v2_replay',**c) for c in oldbinding['contexts'] if c['panel']=='congruent']
 for panel in ['opposite','congruent']:
  filename=f'SOURCE_OOD_V4_{panel.upper()}_ROWS.json';rows=json.loads((P/filename).read_text())
  for template in dict.fromkeys(r['template'] for r in rows):
   n=sum(r['template']==template for r in rows);contexts.append(dict(dataset='v4',panel=panel,template=template,rows_file=filename,amplitudes={role:[[0.,0.,1.,1.,1.,0.]]*n for role in ['subject','attractor']}))
 assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model;enc=tiktoken.get_encoding('gpt2');is_id=enc.encode(' is')[0]
 controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text());modal=torch.tensor([[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]+controls['token_ids'],device='cuda');banks={k:v.to('cuda') for k,v in banks.items()}
 reference=np.array([0.,0.,1.,1.,1.,0.])[PARENTS];counts={k:0 for k in PLAN};records=[];replays=[];zero_checks=[];checks=[];selection=[]
 for context in contexts:
  dataset,panel,template=context['dataset'],context['panel'],context['template'];c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];checks.extend(c['refinement_checks']);n=len(c['raw']);att=model.transformer.h[11].attn;original=att.forward
  background=prepare(att,F.rms_norm(c['raw'],(c['raw'].shape[-1],)),c['first']);counts['baseline_attention']+=1
  def native(role,amplitudes,compiled=False):
   counts['native_suffix']+=1;pos,ds=c['source_components'][role]
   if compiled:att.forward=lambda z,v1=None:(evaluate(att,z,pos,background),v1)
   try:return source_observables32(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],torch.as_tensor(amplitudes,device='cuda',dtype=torch.float64))
   finally:att.forward=original
  base=native('subject',np.zeros((n,23)));zero_checks.append(float((base-native('subject',np.zeros((n,23)),True)).abs().max()));families=[r['family'] for r in c['entries']];orientation=torch.where(c['pairs'][:,0,0]==is_id,1.,-1.).double()
  for role in ['subject','attractor']:
   pos,ds=c['source_components'][role];g=torch.einsum('od,bkd->bok',banks[role],ds);g[:,0]*=orientation[:,None]
   if dataset=='v2_replay':weights=np.array(next(v for v in prior['amplitudes'] if v['panel']==panel and v['template']==template and v['role']==role)['amplitudes']['oracle'])
   else:
    weights=[]
    for row in g.cpu().numpy():
     aa,check=choose(row,reference);weights.append(aa);selection.append(check)
    weights=np.array(weights)
   ref=base-native(role,np.tile(reference,(n,1)));full=base-native(role,weights);compiled=base-native(role,weights,True);pred=-torch.einsum('bok,bk->bo',g,torch.tensor(weights,device='cuda',dtype=torch.float64))
   for family in dict.fromkeys(families):
    ids=[i for i,f in enumerate(families) if f==family];v=compiled[ids];truth=full[ids];b=ref[ids,0];den=truth[:,0].norm().clamp_min(1e-30);err=(v-truth).norm(dim=0)/den
    if dataset=='v2_replay':
     # Exact row+column replaces full attention; old query-frozen target no longer applies.
     old=next(r for r in prior['records'] if r['panel']==panel and r['role']==role and r['family']==family and r['arm']=='oracle');replays.append(float((truth-torch.tensor(old['effect'],device='cuda',dtype=torch.float64)).abs().max()))
    def score(a):return dict(retention=float(a[:,0]@b/b.square().sum().clamp_min(1e-30)),control_ratio=float(a[:,1:].norm(dim=0).max()/a[:,0].norm().clamp_min(1e-30)))
    records.append(dict(dataset=dataset,panel=panel,family=family,role=role,full=score(truth),compiled=score(v),fidelity_errors=err.tolist(),bank_prediction_errors=((pred[ids]-truth).norm(dim=0)/den).tolist(),capability=float((base[ids,0]>0).double().mean()),baseline_margins=base[ids,0].tolist(),effect=v.tolist(),full_effect=truth.tolist(),reference_effect=ref[ids].tolist(),amplitudes=weights[ids].tolist()))
 instrument=counts==PLAN and max(replays)<=1e-8 and max(zero_checks)<=1e-4 and max(c['collapse_error'] for c in checks)<=1e-10 and max(v['relative'] for c in checks for v in c['corrections'])<=.01
 fresh=[r for r in records if r['dataset']=='v4'];assert len(fresh)==16
 instrument=instrument and max(max(r['fidelity_errors']) for r in records)<=.001
 result=dict(binding_sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest(),plan=PLAN,counts=counts,max_old_replay=max(replays),max_zero_edit=max(zero_checks),refinement_checks=checks,selection_checks=selection,records=records,predictions=dict(pred_a_instrument=instrument,pred_b_fidelity=instrument and all(r['fidelity_errors'][0]<=.1 and max(r['fidelity_errors'][1:])<=.05 for r in fresh),pred_c_selectivity=instrument and all(r['compiled']['retention']>=.8 and r['compiled']['control_ratio']<=.1 for r in fresh),pred_d_capability=all(r['capability']>=.9 for r in fresh)),seconds=time.perf_counter()-tic)
 payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','counts','max_old_replay','max_zero_edit','seconds']}))
if __name__=='__main__':main()
