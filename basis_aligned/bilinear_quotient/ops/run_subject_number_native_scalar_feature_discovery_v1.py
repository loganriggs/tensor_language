#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_activation_instrument pred_b_native_singular_coordinates_predict_amplitude pred_c_compact_rank_selected
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, math, os
from pathlib import Path
import numpy as np
import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate

RUNNER=Path(__file__).resolve(); ROOT=RUNNER.parent.parent; POLY=ROOT.parent/'polynomial_causal'
PREREG=POLY/'SUBJECT_NUMBER_NATIVE_SCALAR_FEATURE_DISCOVERY_V1_PREREGISTRATION.md'
LAW=POLY/'SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json'; AXIS=POLY/'SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json'
BINDING=POLY/'SUBJECT_NUMBER_NATIVE_SCALAR_FEATURE_DISCOVERY_V1_BINDING.json'; OUT=ROOT/'circuits/fast_screens/subject_number_native_scalar_feature_discovery_v1_result.json'
RANKS=(1,2,4,8); PRICE={'physical_model_forwards':1,'role_sequences':96,'scalar_least_squares_fits':12,'backwards':0,'parameter_updates':0}
PREDICTION_REGISTRY={'pred_a_exact_activation_instrument':None,'pred_b_native_singular_coordinates_predict_amplitude':None,'pred_c_compact_rank_selected':None}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_bound():
 b=json.loads(BINDING.read_text()); paths={'preregistration':PREREG,'law':LAW,'native_axis':AXIS,'authority':Path(authority.__file__)}
 if b['files']!={k:sha(v) for k,v in paths.items()} or b['ranks']!=list(RANKS) or b['price']!=PRICE: raise ValueError('binding changed')
 law=json.loads(LAW.read_text()); axis=json.loads(AXIS.read_text())
 if law['terminal']!='bilinear_scalar_law_frozen_weights_only' or axis['terminal']!='native_weight_axis_frozen': raise ValueError('parent changed')
 return b,law
def plan():
 b,_=load_bound(); return {'schema':'subject_number_native_scalar_feature_discovery_v1_plan','model_loaded':False,'gpu_accessed':False,'queue_touched':False,'rows':32,'background_subsets':list(factor_gate.BACKGROUND_SUBSETS),'ranks':list(RANKS),'selection':'smallest rank within .01 relative L2 of best leave-one-construction-out result','price':PRICE,'binding_sha256':sha(BINDING)}
def stats(y,p):
 y=np.asarray(y);p=np.asarray(p);yn=np.linalg.norm(y);pn=np.linalg.norm(p)
 return {'count':len(y),'cosine':float(y@p/max(yn*pn,1e-30)),'relative_l2_error':float(np.linalg.norm(y-p)/max(yn,1e-30)),'sign_agreement':float(np.mean((y>0)==(p>0)))}
def fit_predict(x,y,train,test):
 beta=np.linalg.lstsq(x[train],y[train],rcond=None)[0]; return beta,x[test]@beta
def main():
 p=plan()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'): print(json.dumps(p,sort_keys=True));return
 if OUT.exists(): raise FileExistsError(OUT)
 _,law=load_bound(); torch,F,facade=tangent.parent.factors._dependencies(); model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32,verify_weights_sha256=True)
 rows=authority.build_rows(); n=len(rows); device=next(model.parameters()).device; tokens,finals=tangent.parent.downstream.depth.parent.v1._role_batch(rows,torch,device)
 with torch.no_grad():
  _,captured,projection,closure,inputs=tangent.parent._decomposed_forward(model,tokens,finals,torch,F,facade)
  roles={'recipient':tangent._role_slice(captured,0,n),'opposite':tangent._role_slice(captured,n,2*n)}; inp={'recipient':tangent._role_slice(inputs,0,n),'opposite':tangent._role_slice(inputs,n,2*n)}
  fn=tangent._head_function(model,roles['recipient'],roles['opposite'],model.transformer.h[tangent.parent.LAYER].attn,projection,torch,F)
  W=model.transformer.h[11].attn.c_proj.weight.detach().float()[:,3*128:4*128]; U=torch.linalg.svd(W,full_matrices=False).U[:,:max(RANKS)]
  features=[]; targets=[]; templates=[]; directions=[]; cardinalities=[]
  for subset in factor_gate.BACKGROUND_SUBSETS:
   base=fn(factor_gate._raw_for(inp['recipient'],inp['opposite'],subset,F)).detach(); coord=(base@U).cpu().numpy()
   for i,row in enumerate(rows):
    key=f"{row['direction_id']}.cardinality_{len(subset)}"; features.append(coord[i]);targets.append(law['predicted_coefficients'][key]);templates.append(row['template_id']);directions.append(row['direction_id']);cardinalities.append(len(subset))
 x0=np.asarray(features); y=np.asarray(targets); templates=np.asarray(templates); directions=np.asarray(directions); cardinalities=np.asarray(cardinalities)
 reports={}; fits=0
 for k in RANKS:
  x=np.c_[np.ones(len(y)),x0[:,:k]]; pred=np.empty(len(y)); folds={}
  for hold in sorted(set(templates)):
   test=templates==hold;train=~test;beta,q=fit_predict(x,y,train,test);pred[test]=q;folds[hold]={'beta':beta.tolist(),'metrics':stats(y[test],q)};fits+=1
  beta=np.linalg.lstsq(x,y,rcond=None)[0];fits+=1
  reports[str(k)]={'cross_construction':stats(y,pred),'folds':folds,'all_row_beta':beta.tolist(),'by_direction':{d:stats(y[directions==d],pred[directions==d]) for d in sorted(set(directions))},'by_cardinality':{str(c):stats(y[cardinalities==c],pred[cardinalities==c]) for c in sorted(set(cardinalities))}}
 best=min(v['cross_construction']['relative_l2_error'] for v in reports.values());selected=next(k for k in RANKS if reports[str(k)]['cross_construction']['relative_l2_error']<=best+.01)
 instrument=len(y)==512 and fits==PRICE['scalar_least_squares_fits'] and max(closure['input_state_closure_max_absolute_error'],closure['input_normalized_closure_max_absolute_error'])<=5e-5
 predictions={'pred_a_exact_activation_instrument':bool(instrument),'pred_b_native_singular_coordinates_predict_amplitude':bool(instrument and best<=.50),'pred_c_compact_rank_selected':bool(instrument and selected<=4)}
 terminal='invalid' if not instrument else 'native_scalar_feature_selected' if all(predictions.values()) else 'native_scalar_feature_discovery_null'
 result={'schema':'subject_number_native_scalar_feature_discovery_v1_result','terminal':terminal,'predictions':predictions,'selected_rank':selected,'best_relative_l2_error':best,'reports':reports,'instrument':{'examples':len(y),'fits':fits,'role_state_closure_max_absolute_error':closure['input_state_closure_max_absolute_error'],'role_normalized_closure_max_absolute_error':closure['input_normalized_closure_max_absolute_error']},'outcome_access':{'behavioral_effects':False,'answer_logits':False,'exact_donor_displacements':False,'fresh_authority':False},'price':PRICE,'checkpoint_weights_sha256':checkpoint.weights_sha256,'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
 managed.atomic_create_json(OUT,result); print(json.dumps({'terminal':result['terminal'],'selected_rank':selected,'best_relative_l2_error':best,'reports':{k:v['cross_construction'] for k,v in reports.items()},'instrument':result['instrument']},indent=2)); assert instrument
if __name__=='__main__': main()
