#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_absolute pred_c_relative
"""New native panel for frozen private320 graph, samegates andmatchedbaseline.
48captures;576vs832products,971660floatseach. Primary320 predeclared;
not selected from new outcomes. Allindividual+combined selections retained.
"""
import os,json

def main():
 import run_partial_graph_fresh_v1 as run
 run.GRAPH_FILE='PARTIAL_GRAPH_FROZEN_V2.pt';run.BASELINE_FILE='PARTIAL_GRAPH_BASELINES_V2.pt';run.PLAN_FILE='PARTIAL_GRAPH_FRESH_PLAN_V2.json';run.TOKEN_FILE='PARTIAL_GRAPH_FRESH_TOKENS_V2.pt';run.DONOR_FILE='PARTIAL_GRAPH_FRESH_DONORS_V2.pt';run.OUTPUT_FILE='PARTIAL_GRAPH_FRESH_NATIVE_V2.json';run.main()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):return
 path=run.P/run.OUTPUT_FILE;result=json.loads(path.read_text());pred=result['predictions'];result['predictions']=dict(pred_a_instrument=pred['pred_a_instrument'],pred_b_absolute=pred['pred_b_absolute'],pred_c_relative=pred['pred_c_relative']);path.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
