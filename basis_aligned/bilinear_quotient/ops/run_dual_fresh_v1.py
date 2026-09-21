#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_absolute pred_c_relative pred_d_both_baselines
"""Fresh sixth-panel comparison of frozen wide source graphs and cost-matched pairs.
32FWdocs+16codefiles,48capturescontext256. Threewidegraphs592products1342028floats;
matchedpairbaselines1152products1340940floats. Mixedalpha.5 primary, purearmsreported.
All3constituents+sum, natural/hybrid/change, all/continuation/spacedword,2domains.
pred_a source/finalreplay<1e-4/1e-5; pred_b primarynatural/hybrid<=.15,change<=.20;
pred_c primary<=1.10covariancebaseline; pred_d primary<=1.10bothmatchedbaselines.
Earlier coefficientguardFAIL retained. Null: metricfit doesnottransfer to native
interventions. Native z/h supplied, no newdatafitting or semantic adoption.
"""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import run_partial_graph_fresh_v1 as run
 sys.path.insert(0,str(P));from compact_source_graph import component_scalars
 run.COMPONENT_EXECUTOR=component_scalars;run.EXTRA_GRAPH_FILE=None
 run.GRAPH_FILE='DUAL_FRESH_MIXED_V1.pt';run.BASELINE_FILE='DUAL_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt'
 run.ADDITIONAL_GRAPH_FILES={'isotropic_graph':'DUAL_FRESH_ISOTROPIC_V1.pt','covariance_graph':'DUAL_FRESH_COVARIANCE_V1.pt'}
 run.ADDITIONAL_BASELINE_FILES={'isotropic_baseline':'DUAL_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt'}
 run.PLAN_FILE='DUAL_FRESH_PLAN_V1.json';run.TOKEN_FILE='DUAL_FRESH_TOKENS_V1.pt';run.DONOR_FILE='DUAL_FRESH_DONORS_V1.pt';run.OUTPUT_FILE='DUAL_FRESH_NATIVE_V1.json'
 plan=json.loads((P/run.PLAN_FILE).read_text());assert hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest()==plan['native_executor_sha256']
 run.main()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):return
 path=P/run.OUTPUT_FILE;result=json.loads(path.read_text());comparisons=[]
 for domain in result['summary']:
  for selection in ['mode1','mode2','mode3','combined']:
   for family,limit in [('natural',.15),('hybrid',.15),('change',.20)]:
    for cohort in ['all','continuation','spaced_word']:
     cells=result['summary'][domain][selection]
     for candidate in ['graph','isotropic_graph','covariance_graph']:
      error=cells[candidate][family][cohort]['effect_relative_error'];cov=cells['separate'][family][cohort]['effect_relative_error'];iso=cells['isotropic_baseline'][family][cohort]['effect_relative_error']
      comparisons.append(dict(domain=domain,selection=selection,family=family,cohort=cohort,candidate=candidate,error=error,absolute_pass=error<=limit,covariance_baseline_error=cov,isotropic_baseline_error=iso,covariance_relative_pass=error<=1.1*cov,isotropic_relative_pass=error<=1.1*iso))
 result['all_candidate_comparisons']=comparisons;result['predictions']['pred_d_both_baselines']=all(c['covariance_relative_pass'] and c['isotropic_relative_pass'] for c in comparisons if c['candidate']=='graph')
 path.write_text(json.dumps(result,indent=2)+'\n');print(result['predictions'],flush=True)
if __name__=='__main__':main()
