"""Four global quadratic-dictionary deletions, peer substitution and readout refit.
Preregistered native screen: >=15%productsaving and<1%archived-program error
on both existing panels for at least one candidate. Not full-model adoption.
"""
import json,time
from fractions import Fraction
from pathlib import Path
import torch
from arithmetic_dag import DAG
from dag_dictionary_substitution import proposals,refit_outputs

def main():
 torch.set_num_threads(2);start=time.monotonic();p=Path(__file__).resolve().parent;source=json.loads((p/'NATIVE_DAG_EXPORT_V1.graph.json').read_text());d=DAG(degree_limit=4);refs=[]
 for node in source['nodes']:
  if node['op']=='input':r=d.input(node['coordinate'])
  elif node['op']=='constant':r=d.constant()
  elif node['op']=='product':r=d.product(*(refs[n] for n in node['inputs']))
  else:r=d.linear([(refs[n],Fraction(a,b)) for n,a,b in node['terms']])
  refs.append(r)
 out=[refs[n] for n in source['outputs']]
 bank=[n for n in d.reachable(out) if d.nodes[n][0]=='linear' and len(d.nodes[n][1])==4 and all(c==1 and d.nodes[r][0]=='product' and d.degrees[r]==2 for r,c in d.nodes[n][1])];assert len(bank)==4
 panels=torch.load(p/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];x=panels[0]['rows'][:512].double();weights=torch.full((len(x),),1/len(x),dtype=x.dtype);baseline=d.cost(out);reference=[d.evaluate(out,panel['rows'][:512].double()) for panel in panels];rows=[]
 for new,proposal in proposals(d,out,bank,x,weights):
  before_error=float((d.evaluate(new,x)-reference[0]).norm()/reference[0].norm());new,refit=refit_outputs(d,new,x,reference[0],weights);errors=[]
  for panel,truth in zip(panels,reference):
   prediction=d.evaluate(new,panel['rows'][:512].double());errors.append(float((prediction-truth).norm()/truth.norm()))
  row=dict(**proposal,unrefitted_calibration_error=before_error,refit=refit,program_errors_by_panel=errors);rows.append(row);print(json.dumps(row),flush=True)
 pred=dict(pred_a_boundary=len(bank)==4,pred_b_saving=all(r['refit']['cost']['products']<=.85*baseline['products'] for r in rows),pred_c_fidelity=any(max(r['program_errors_by_panel'])<.01 and r['refit']['cost']['products']<=.85*baseline['products'] for r in rows))
 result=dict(predictions=pred,baseline=baseline,bank_nodes=bank,records=rows,seconds=time.monotonic()-start,scope='Full global replacement of each four-product quadratic bank feature by peers. Exact least-squares readout refit on <=512oldcalibrationrows. Errors target archived approximate quartic program on two reusedpanels; no native teacher/OOD/semanticadoption. No direction refit in this first screen.')
 (p/'NATIVE_DICTIONARY_SUBSTITUTION_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])))
if __name__=='__main__':main()
