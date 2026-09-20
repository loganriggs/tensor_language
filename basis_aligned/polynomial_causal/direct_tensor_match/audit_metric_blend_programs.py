"""Literal graph and reused polynomial diagnostics for every frozen metric blend."""
import json
from pathlib import Path
import torch
from audit_quadratic_skip_graph import build
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(3);torch.set_grad_enabled(False);panels=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];U=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True)['output_directions'].double();rows=[]
 for alpha,s in torch.load(P/'SKIP_METRIC_BLEND_PROGRAM_V1.pt',weights_only=True)['programs'].items():
  s={k:t.double() for k,t in s.items()};d,out=build(s);x=panels[0]['rows'][:16].double();pred=quartic(s,x);replay=float((d.evaluate(out,x)-pred).norm()/pred.norm());cost=d.cost(out);assert replay<1e-12 and cost['products']==10 and cost['stored_coefficients']==21948;record=dict(alpha=alpha,cost=cost,replay_error=replay,panels=[])
  for panel in panels:
   x=panel['rows'].double();y=panel['targets'].double();delta=quartic(s,x)-y;record['panels'].append(dict(context=panel['context'],relative_error=float(delta.norm()/y.norm()),mode_mse=(delta@U).square().mean(0).tolist()))
  rows.append(record)
 result=dict(records=rows,scope='Frozen matched-cost programs, literal graph replay, reused FineWeb diagnostics only; no fitting or alpha selection.')
 (P/'SKIP_METRIC_BLEND_GRAPH_V1.json').write_text(json.dumps(result,indent=2)+'\n');print([(r['alpha'],[v['relative_error'] for v in r['panels']]) for r in rows])
if __name__=='__main__':main()
