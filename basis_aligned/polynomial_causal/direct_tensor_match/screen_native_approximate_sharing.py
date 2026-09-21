"""Proposal-only native screen; no refit or adoption from local similarity."""
import json,time
from fractions import Fraction
from pathlib import Path
import torch
from arithmetic_dag import DAG
from dag_approximate_sharing import proposals

def main():
 torch.set_num_threads(2);start=time.monotonic();p=Path(__file__).resolve().parent;source=json.loads((p/'NATIVE_DAG_EXPORT_V1.graph.json').read_text());d=DAG();refs=[]
 for node in source['nodes']:
  if node['op']=='input':r=d.input(node['coordinate'])
  elif node['op']=='constant':r=d.constant()
  elif node['op']=='product':r=d.product(*(refs[n] for n in node['inputs']))
  else:r=d.linear([(refs[n],Fraction(a,b)) for n,a,b in node['terms']])
  refs.append(r)
 out=[refs[n] for n in source['outputs']];panels=torch.load(p/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];x=panels[0]['rows'][:512].double();weights=torch.full((len(x),),1/len(x),dtype=x.dtype);baseline=d.cost(out);reference=[d.evaluate(out,panel['rows'][:512].double()) for panel in panels];rows=[]
 for proposal in proposals(d,out,x,weights,max_candidates=4,max_relative_error=.1):
  replacement=d.linear([(proposal['source'],proposal['scale'])]);new=d.replace(out,proposal['target'],replacement);errors=[]
  for panel,truth in zip(panels,reference):
   predicted=d.evaluate(new,panel['rows'][:512].double());errors.append(float((predicted-truth).norm()/truth.norm()))
  rows.append(dict(**proposal,cost=d.cost(new),frozen_program_errors_by_panel=errors))
 result=dict(baseline=baseline,proposals=rows,seconds=time.monotonic()-start,scope='Local functional similarity <=10% on at most512old calibration rows proposes at most4internal-node merges. Global function errors measured against archived approximate program on old cached panels, not native teacher and not fresh/OOD. No refitting or adoption; global cost counts remaining consumers.')
 (p/'NATIVE_APPROXIMATE_SHARING_SCREEN_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
