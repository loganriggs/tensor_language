"""Exact readout contraction following a previously screened dictionary edit."""
import json,time
from fractions import Fraction
from pathlib import Path
import torch
from arithmetic_dag import DAG
from dag_dictionary_substitution import proposals,refit_outputs,contract_readout

def main():
 torch.set_num_threads(2);start=time.monotonic();p=Path(__file__).resolve().parent;source=json.loads((p/'NATIVE_DAG_EXPORT_V1.graph.json').read_text());d=DAG(degree_limit=4);refs=[]
 for node in source['nodes']:
  if node['op']=='input':r=d.input(node['coordinate'])
  elif node['op']=='constant':r=d.constant()
  elif node['op']=='product':r=d.product(*(refs[n] for n in node['inputs']))
  else:r=d.linear([(refs[n],Fraction(a,b)) for n,a,b in node['terms']])
  refs.append(r)
 out=[refs[n] for n in source['outputs']];bank=[n for n in d.reachable(out) if d.nodes[n][0]=='linear' and len(d.nodes[n][1])==4 and all(c==1 and d.nodes[r][0]=='product' and d.degrees[r]==2 for r,c in d.nodes[n][1])]
 panels=torch.load(p/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];x=panels[0]['rows'][:512].double();w=torch.full((len(x),),1/len(x),dtype=x.dtype);y=d.evaluate(out,x);rows=[]
 original=contract_readout(d,out,bank);rows.append(dict(arm='original_four_feature_boundary',before=d.cost(out),after=d.cost(original),relative_replay=float((d.evaluate(original,x)-y).norm()/y.norm())))
 prior=json.loads((p/'NATIVE_DICTIONARY_SUBSTITUTION_V1.json').read_text());selected=min(prior['records'],key=lambda r:r['program_errors_by_panel'][0]);new,proposal=next((new,record) for new,record in proposals(d,out,bank,x,w) if record['target']==selected['target']);new,_=refit_outputs(d,new,x,y,w);value=d.evaluate(new,x);compiled=contract_readout(d,new,proposal['peers']);replay=float((d.evaluate(compiled,x)-value).norm()/value.norm());fidelity=float((value-y).norm()/y.norm());assert abs(fidelity-selected['program_errors_by_panel'][0])<1e-8
 rows.append(dict(arm='selected_three_feature_boundary',selected_by='previous calibration error only',before=d.cost(new),after=d.cost(compiled),relative_replay=replay,archived_program_error_unchanged=fidelity))
 pred=dict(pred_a_exact=all(r['relative_replay']<1e-12 for r in rows),pred_b_smaller=rows[1]['after']['products']<rows[1]['before']['products'] and rows[1]['after']['stored_coefficients']<rows[1]['before']['stored_coefficients'])
 result=dict(predictions=pred,records=rows,seconds=time.monotonic()-start,scope='Bounded exact polynomial compilation only above quadratic-feature boundary. Does not expand full native input tensor. Three-feature replacement retains large previous approximation error; cheaper compilation is not adoption. Four-feature compilation also priced, not automatically accepted.')
 (p/'DICTIONARY_READOUT_CONTRACTION_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
