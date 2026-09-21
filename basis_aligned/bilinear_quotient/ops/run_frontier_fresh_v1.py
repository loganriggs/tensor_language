#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_absolute pred_c_relative pred_d_both_baselines
"""Fresh frozen local graph comparison:32FineWeb documents and16code files.
1120products/1131980floats versus pair baselines969products/1129758floats.
Both baseline coefficient geometries, same48captures/context256.
pred_a instrument replay; pred_b natural/hybrid<=.15,change<=.20;
pred_c <=1.10covariancebaseline; pred_d <=1.10bothbaselines, everycell.
Opened reconstruction passed; no fresh outcome or semantic adoption yet.
Null: opened-state repairs fail native effect transfer or donor composition.
"""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import run_partial_graph_fresh_v1 as run
 sys.path.insert(0,str(P));from pairwise_component_interface import component_scalars,move_preserving_dtype
 run.move=lambda value,torch:move_preserving_dtype(value,'cuda')
 run.COMPONENT_EXECUTOR=component_scalars;run.EXTRA_GRAPH_FILE=None
 run.GRAPH_FILE='FRONTIER_FRESH_GRAPH_V1.pt';run.BASELINE_FILE='FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt'
 run.ADDITIONAL_GRAPH_FILES={}
 run.ADDITIONAL_BASELINE_FILES={'isotropic_baseline':'FRONTIER_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt'}
 run.PLAN_FILE='FRONTIER_FRESH_PLAN_V1.json';run.TOKEN_FILE='FRONTIER_FRESH_TOKENS_V1.pt';run.DONOR_FILE='FRONTIER_FRESH_DONORS_V1.pt';run.OUTPUT_FILE='FRONTIER_FRESH_NATIVE_V1.json'
 plan=json.loads((P/run.PLAN_FILE).read_text());assert hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest()==plan['native_executor_sha256']
 run.main()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):return
 path=P/run.OUTPUT_FILE;result=json.loads(path.read_text());comparisons=[]
 for domain in result['summary']:
  for selection in ['mode1','mode2','mode3','combined']:
   for family,limit in [('natural',.15),('hybrid',.15),('change',.20)]:
    for cohort in ['all','continuation','spaced_word']:
     cells=result['summary'][domain][selection]
     for candidate in ['graph']:
      error=cells[candidate][family][cohort]['effect_relative_error'];cov=cells['separate'][family][cohort]['effect_relative_error'];iso=cells['isotropic_baseline'][family][cohort]['effect_relative_error']
      comparisons.append(dict(domain=domain,selection=selection,family=family,cohort=cohort,candidate=candidate,error=error,absolute_pass=error<=limit,covariance_baseline_error=cov,isotropic_baseline_error=iso,covariance_relative_pass=error<=1.1*cov,isotropic_relative_pass=error<=1.1*iso))
 result['all_candidate_comparisons']=comparisons;previous=result['predictions']
 result['predictions']=dict(pred_a_instrument=previous['pred_a_instrument'],pred_b_absolute=previous['pred_b_absolute'],pred_c_relative=previous['pred_c_relative'],pred_d_both_baselines=all(c['covariance_relative_pass'] and c['isotropic_relative_pass'] for c in comparisons if c['candidate']=='graph'))
 path.write_text(json.dumps(result,indent=2)+'\n');print(result['predictions'],flush=True)
if __name__=='__main__':main()
