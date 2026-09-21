"""Apply bounded exact equality sharing to an existing exported native quartic DAG."""
import json,time
from fractions import Fraction
from pathlib import Path
import torch
from arithmetic_dag import DAG
from dag_functional_sharing import best_equivalence_edit

def main():
 torch.set_num_threads(2);start=time.monotonic();p=Path(__file__).resolve().parent;source=json.loads((p/'NATIVE_DAG_EXPORT_V1.graph.json').read_text());d=DAG();refs=[]
 for node in source['nodes']:
  if node['op']=='input':r=d.input(node['coordinate'])
  elif node['op']=='constant':r=d.constant()
  elif node['op']=='product':r=d.product(*(refs[n] for n in node['inputs']))
  else:r=d.linear([(refs[n],Fraction(a,b)) for n,a,b in node['terms']])
  refs.append(r)
 out=[refs[n] for n in source['outputs']];before=d.cost(out);x=torch.load(p/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'][:16].double();reference=d.evaluate(out,x);history=[]
 for _ in range(10):
  new,report=best_equivalence_edit(d,out,max_terms=256,max_pairs=4096);history.append(report)
  if new is None:break
  out=new
 actual=d.evaluate(out,x);error=float((actual-reference).norm()/reference.norm());after=d.cost(out)
 result=dict(before=before,after=after,edits=sum(h['accepted'] for h in history),probe_replay_error=error,history=history,seconds=time.monotonic()-start,scope='Exact bounded formal-polynomial equivalence on preexisting approximate native quartic program, not on full trained model. Dense leaves may be opaque; null does not prove irreducibility. Numeric replay checks evaluator consistency, equality itself uses rational certificates. Same fixed output frame and earlier approximation errors remain.')
 assert error<1e-12;(p/'NATIVE_FUNCTIONAL_SHARING_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='history'}))
if __name__=='__main__':main()
