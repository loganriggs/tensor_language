"""Compile a frozen centered approximation to explicitly shared scalar nodes."""
import json
from pathlib import Path
import torch
from arithmetic_dag import DAG
from audit_centered_compact import evaluate
P=Path(__file__).resolve().parent

def build(s):
 d=DAG(degree_limit=4);inputs=[d.input(i) for i in range(len(s['mu']))];one=d.constant();delta=[d.linear([(n,1),(one,-float(mu))]) for n,mu in zip(inputs,s['mu'])]
 linear=[d.linear(zip(delta,row.tolist())) for row in s['linear_reader']]
 left=[d.linear(zip(delta,row.tolist())) for row in s['quadratic_left']];right=[d.linear(zip(delta,row.tolist())) for row in s['quadratic_right']];products=[d.product(a,b) for a,b in zip(left,right)];outputs=[]
 if 'Z' in s:products=[d.linear(zip(products,row.tolist())) for row in s['Z']]
 quadratic_writer=s['quadratic_writer'] if 'quadratic_writer' in s else s['W']
 for rowl,rowq,c in zip(s['linear_writer'],quadratic_writer,s['constant']):outputs.append(d.linear(list(zip(linear,rowl.tolist()))+list(zip(products,rowq.tolist()))+[(one,float(c))]))
 return d,outputs

def main():
 torch.set_num_threads(1);source=torch.load(P/'CENTERED_CONTINUOUS_REFACTOR_V1.pt',weights_only=True);x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'][:32].double();rows=[]
 for width,s in source['programs'].items():
  d,out=build(s);actual=d.evaluate(out,x);expected=evaluate(s,x);replay=float((actual-expected).norm()/expected.norm());price=d.cost(out);assert replay<1e-10 and price['stored_coefficients']==sum(v.numel() for v in s.values()) and price['products']==width;assert max(d.degrees[k] for k in out)==2;rows.append(dict(width=width,cost=price,replay_relative_error=replay,maximum_degree=2))
 result=dict(records=rows,scope='Actual FP32 frozen refactor programs compiled to scalarDAG; evaluation inFP64. Includes centering,constant,linearbranch,quadraticproducts,fullreducedreadout. Commonvocabularyframe excluded.');(P/'CENTERED_DAG_EXPORT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
