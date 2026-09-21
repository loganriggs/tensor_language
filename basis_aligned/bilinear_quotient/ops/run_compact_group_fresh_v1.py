#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_absolute pred_c_relative
"""Fresh native tests for compact399 group and its original three constituents.
32identified FineWeb docs+16newstdlibfiles,48captures256tokens.
Original72cell gates retained: natural/hybrid<=.15,change<=.20,<=1.10separate.
Additional combined-group18cells must pass absolute and BOTH separate768 and
partial512 relativebaselines. No targetrotation, no fitting on fresh data.
Literal storage896198 vs897804bothbaselines. Null: aggregatefit masksOOD or
constituent failure; native z,h remain explicit. Semantic adoption unclaimed.
"""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import run_partial_graph_fresh_v1 as run
 sys.path.insert(0,str(P));from compact_source_graph import component_scalars
 run.COMPONENT_EXECUTOR=component_scalars;run.EXTRA_GRAPH_FILE='PARTIAL_GRAPH_FROZEN_V1.pt'
 run.GRAPH_FILE='COMPACT_GROUP_FROZEN_V1.pt';run.BASELINE_FILE='MULTIMODE_PAIR_BASELINES_V1.pt';run.PLAN_FILE='COMPACT_GROUP_FRESH_PLAN_V1.json';run.TOKEN_FILE='COMPACT_GROUP_FRESH_TOKENS_V1.pt';run.DONOR_FILE='COMPACT_GROUP_FRESH_DONORS_V1.pt';run.OUTPUT_FILE='COMPACT_GROUP_FRESH_NATIVE_V1.json'
 plan=json.loads((P/run.PLAN_FILE).read_text())
 assert hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest()==plan['native_executor_sha256']
 run.main()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):return
 path=P/run.OUTPUT_FILE;result=json.loads(path.read_text());pred=result['predictions'];combined=[]
 for cell in result['cells']:
  if cell['selection']!='combined':continue
  old=result['summary'][cell['domain']]['combined']['partial'][cell['family']][cell['cohort']]['effect_relative_error']
  combined.append(dict(**cell,partial_error=old,partial_ratio=cell['graph_error']/old,partial_relative_pass=cell['graph_error']<=1.10*old))
 result['predictions']=dict(pred_a_instrument=pred['pred_a_instrument'],pred_b_absolute=pred['pred_b_absolute'],pred_c_relative=pred['pred_c_relative'],pred_d_combined_group=all(c['absolute_pass'] and c['relative_pass'] and c['partial_relative_pass'] for c in combined))
 result['combined_group_cells']=combined;path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=result['predictions'],combined_group_cells=combined)),flush=True)
if __name__=='__main__':main()
