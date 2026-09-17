#!/usr/bin/env python3
# BQGATE:800bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a-f unchanged structural screen; pred_g face error<=.8 each frozen null.
800 forwards/300 seconds; all five new construction families must pass.
Base-trained constants and OLS fixed before outcomes, no refit.
"""
import json,os
from pathlib import Path
import torch
import run_typed_face_structure_v1 as screen
STEM='TYPED_FACE_PROSPECTIVE_V1'
@torch.no_grad()
def main():
 screen.STEM=STEM
 screen.main()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):return
 P=screen.P;doc=json.loads((P/(STEM+'_ROWS.json')).read_text());rows=doc['rows']
 baseline=json.loads((P/'STRUCTURE_PREDICTION_NULLS_V1_RESULT.json').read_text())
 v=torch.load(P/(STEM+'_ARTIFACT.pt'),map_location='cpu',weights_only=True)['values'];target=v[1,:,0]-v[0,:,0];face=v[2,:,0]-v[0,:,0]
 X=torch.tensor([[1.,1. if r['cue']=='British' else -1.,len(r['ids'])/32.,r['city_position']/32.,float(r['pair'])]+[float(r['endpoint']==i) for i in range(1,6)] for r in rows],dtype=torch.float64)
 fitted=X@torch.tensor(baseline['coefficients'],dtype=torch.float64)
 constant=torch.tensor([baseline['cue_constants'][r['cue']] for r in rows],dtype=torch.float64)
 metrics={}
 for family in doc['variants']:
  idx=[i for i,r in enumerate(rows) if r['variant']==family];den=target[idx].norm().clamp_min(1e-8)
  metrics[family]={name:float((pred[idx]-target[idx]).norm()/den) for name,pred in [('face',face),('cue_constant',constant),('text_ols',fitted)]}
 result_path=P/(STEM+'_RESULT.json');result=json.loads(result_path.read_text());result['pred_g']=all(x['face']<=.8*x['cue_constant'] and x['face']<=.8*x['text_ols'] for x in metrics.values())
 result['frozen_baseline_errors']=metrics;result['panel_status']='All ten constructions fresh at freeze; same cities and endpoints as baseline training'
 result['scope']='Prospective frozen constant/text-feature baselines and all matched-direction screen gates. Conditional native-state formula has more information than token-only baselines. Rank-deficient base fit acknowledged. No new-city, new-endpoint, standalone token-only, composition or simplicity certification.'
 result_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pred_g':result['pred_g'],'frozen_baseline_errors':metrics},indent=2))
if __name__=='__main__':main()
