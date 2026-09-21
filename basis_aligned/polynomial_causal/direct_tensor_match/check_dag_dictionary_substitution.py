import json,time
from pathlib import Path
import torch
from arithmetic_dag import DAG
from dag_dictionary_substitution import proposals,refit_outputs
from dag_square_optimizer import rule

def build(name):
 d=DAG(degree_limit=4);a,b,c,e=[d.input(i) for i in range(4)];sq=lambda x:d.product(x,x);lin=lambda *t:d.linear(t)
 if name=='cross_sum':
  q,r=d.product(a,b),d.product(c,e);s=lin((d.product(lin((a,1),(c,1)),lin((b,1),(e,1))),.5),(d.product(lin((a,1),(c,-1)),lin((b,1),(e,-1))),.5))
 elif name in ('squares_sum','squares_difference'):
  q,r=sq(a),sq(b)
  s=lin((sq(lin((a,1),(b,1))),.5),(sq(lin((a,1),(b,-1))),.5)) if name=='squares_sum' else d.product(lin((a,1),(b,1)),lin((a,1),(b,-1)))
 elif name=='affine_shift':q,r=d.product(a,b),a;s=d.product(a,lin((b,1),(d.constant(),1)))
 else:
  aa,bb=d.product(a,b),d.product(c,e);q,r=sq(aa),sq(bb);s=lin((sq(lin((aa,1),(bb,1))),.5),(sq(lin((aa,1),(bb,-1))),.5))
 out=[lin((q,.7)),lin((r,.9)),lin((s,1.1))];return d,out,[q,r,s]

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.monotonic();xx,w=rule(5);x=xx[:,:4];torch.manual_seed(2701);xv=torch.randn(4096,4);rows=[]
 for name in ['cross_sum','squares_sum','squares_difference','affine_shift','quartic_sum']:
  d,out,nodes=build(name);y=d.evaluate(out,x);yv=d.evaluate(out,xv);baseline=d.cost(out);candidates=[]
  for new,record in proposals(d,out,nodes,x,w):
   new,refit=refit_outputs(d,new,x,y,w);error=float((d.evaluate(new,xv)-yv).norm()/yv.norm());candidates.append(dict(**record,refit=refit,fresh_error=error))
  good=[r for r in candidates if r['fresh_error']<1e-10 and r['refit']['cost']['products']<baseline['products']];rows.append(dict(family=name,baseline=baseline,candidates=candidates,passed=bool(good)))
 # Three independent quadratic outputs cannot be represented by two scalar peers.
 d=DAG(degree_limit=2);a,b,c,e=[d.input(i) for i in range(4)];q=d.product(a,b);r=d.product(c,e);s=d.linear([(q,1),(r,1),(d.product(a,e),.01)]);out=[d.linear([(q,.7)]),d.linear([(r,.9)]),d.linear([(s,100),(q,-100),(r,-100)])];y=d.evaluate(out,x);new,proposal=next((new,row) for new,row in proposals(d,out,[q,r,s],x,w) if row['target']==s);new,refit=refit_outputs(d,new,x,y,w);negative_error=float((d.evaluate(new,x)-y).norm()/y.norm())
 pred=dict(pred_a_recovery=all(r['passed'] for r in rows),pred_b_negative=proposal['local_error']<.01 and negative_error>.1,pred_c_degree=max(d.degrees[n] for n in new)<=2)
 result=dict(predictions=pred,records=rows,negative=dict(proposal=proposal,refit=refit,output_error=negative_error),seconds=time.monotonic()-start,scope='Global peer-combination replacement plus exact linear-output refit; five planted dictionaries and downstream selective contrast. Fresh artificial probes, no native or semantic claim.')
 Path(__file__).with_name('DAG_DICTIONARY_SUBSTITUTION_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,negative_error=negative_error,seconds=result['seconds'])))
if __name__=='__main__':main()
