"""Replay the preregistered >=99% retained-energy output-sharing choice."""
import json
from pathlib import Path
import torch
from arithmetic_dag import DAG
P=Path(__file__).resolve().parent

def build(U,V,W,Z=None):
 d=DAG();inputs=[d.input(i) for i in range(U.shape[-1])];q=[]
 for u,v in zip(U,V):
  products=[d.product(d.linear(zip(inputs,a.tolist())),d.linear(zip(inputs,b.tolist()))) for a,b in zip(u,v)]
  q.append(d.linear([(n,1) for n in products]))
 pairs=torch.triu_indices(len(q),len(q)).T
 roots=[d.product(q[int(i)],q[int(j)]) for i,j in pairs]
 features=roots if Z is None else [d.linear(zip(roots,row.tolist())) for row in Z]
 outputs=[d.linear(zip(features,row.tolist())) for row in W]
 return d,outputs

def main():
 torch.set_num_threads(1);key=('second_floor01','muon',1)
 s=torch.load(P/'DAG_OUTPUT_SHARING_V1.pt',weights_only=True)['students'][key]
 original=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['students'][key]
 U,V,W,Z=[s[n].double() for n in ['U','V','W','Z']]
 x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'][:32].double()
 d,out=build(U,V,W,Z);base,bout=build(U,V,original['C'].double())
 q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);i,j=torch.triu_indices(4,4);expected=(q[:,i]*q[:,j])@Z.T@W.T
 error=float((d.evaluate(out,x)-expected).norm()/expected.norm())
 result=dict(source_key=list(key),selection='Smallest tested rank retaining >=99% student energy in original metric',rank=W.shape[1],cost=d.cost(out),baseline_cost=base.cost(bout),factor_program_replay_relative_error=error)
 assert error<1e-10 and result['cost']['stored_coefficients']==41512 and result['cost']['products']==26
 (P/'SHARED_OUTPUT_DAG_REPLAY_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
