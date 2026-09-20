"""Compile skip programs to literal DAGs with shared quadratic nodes."""
import json
from pathlib import Path
import torch
from fuse_root_program import build as base_build
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def build(s):
 d,out=base_build(s);inputs=[d.input(i) for i in range(s['A'].shape[1])]
 # Interning reuses exactly the primitive reader and product nodes already built.
 left=[d.linear(zip(inputs,r.tolist())) for r in s['A']];right=[d.linear(zip(inputs,r.tolist())) for r in s['B']];p=[d.product(a,b) for a,b in zip(left,right)]
 features=[d.linear(zip(p,row.tolist())) for row in s['skip_reader']] if 'skip_reader' in s else p
 outputs=[]
 for original,row in zip(out,s['skip_writer']):
  key=d.nodes[original];terms=list(key[1]) if key[0]=='linear' else [(original,1)]
  outputs.append(d.linear(terms+list(zip(features,row.tolist()))))
 return d,outputs

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 x=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'][0]['rows'][:16].double();rows=[]
 for rank,s in torch.load(P/'QUADRATIC_SKIP_PROGRAM_V1.pt',weights_only=True)['programs'].items():
  s={k:v.double() for k,v in s.items()};d,out=build(s);cost=d.cost(out);expected=quartic(s,x);error=float((d.evaluate(out,x)-expected).norm()/expected.norm())
  assert error<1e-12 and cost['products']==10 and cost['stored_coefficients']==sum(t.numel() for t in s.values())
  rows.append(dict(rank=rank,cost=cost,replay_error=error))
 result=dict(records=rows,scope='Literal reachable DAG with primitive quadratic nodes shared between quartic roots and direct skip readout; fixed shared vocabulary frame excluded as in baseline.')
 (P/'QUADRATIC_SKIP_GRAPH_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
