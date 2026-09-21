import json,time
from pathlib import Path
import torch
from arithmetic_dag import DAG
from dag_square_optimizer import rule
from dag_approximate_sharing import one_round

def build(family):
 d=DAG(degree_limit=4);a,b,c,e=[d.input(i) for i in range(4)];one=d.constant();sq=lambda x:d.product(x,x)
 altered=d.linear([(a,1),(c,.025)])
 def feature(t):
  if family=='product':return d.product(t,b)
  if family=='sum_squares':return d.linear([(sq(t),1),(sq(b),1)])
  if family=='sum_products':return d.linear([(d.product(t,b),1),(d.product(c,e),1)])
  if family=='affine_product':return d.product(d.linear([(t,1),(one,.2)]),d.linear([(b,1),(one,-.4)]))
  if family=='quartic_square':return sq(d.linear([(d.product(t,b),1),(d.product(c,e),1)]))
 q,r=feature(a),feature(altered)
 branches=(one,one) if family=='quartic_square' else (sq(c),sq(e))
 outputs=[d.linear([(d.product(q,branches[0]),.7)]),d.linear([(d.product(r,branches[1]),.9)])]
 truth=[d.linear([(d.product(q,branches[0]),.7)]),d.linear([(d.product(q,branches[1]),.9)])]
 return d,outputs,truth

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.monotonic();xx,w=rule(5);x=xx[:,:4];torch.manual_seed(1122);xv=torch.randn(4096,4);rows=[]
 for family in ['product','sum_squares','sum_products','affine_product','quartic_square']:
  d,out,target=build(family);y=d.evaluate(target,x);yv=d.evaluate(target,xv);model,row=one_round(d,out,x,y,w,steps=200);row['family']=family
  if model is not None:
   with torch.no_grad():pred=model(xv);graph,outputs=model.export();row.update(fresh_error=float((pred-yv).norm()/yv.norm()),export_error=float((graph.evaluate(outputs,xv)-pred).norm()/pred.norm()),accepted_price=model.price())
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
 # Highly similar features feed a contrast that exposes the small difference.
 d=DAG(degree_limit=2);a,b,c,e=[d.input(i) for i in range(4)];q=d.product(a,b);r=d.linear([(q,1),(d.product(c,e),.01)]);out=[d.linear([(q,.7)]),d.linear([(r,100),(q,-100)])];y=d.evaluate(out,x);model,negative=one_round(d,out,x,y,w,steps=200);negative['case']='amplified_difference';assert model is None
 pred=dict(pred_a_replay=all(r.get('export_error',0)<1e-10 for r in rows),pred_b_recover=sum(r['accepted'] and r['fresh_error']<.01 and r['accepted_price']['products']<r['baseline_price']['products'] for r in rows)>=4,pred_c_selectivity=not negative['accepted'])
 result=dict(predictions=pred,records=rows,negative=negative,seconds=time.monotonic()-start,scope='Approximate sharing plus joint fixed-graph coefficient refitting, bounded four-proposal search. Five planted near-duplicate intermediate families; Gaussian quadrature objective, fresh artificial probes. Correlation-only proposals tested against downstream amplified-difference control. Not native discovery or uniqueness.')
 Path(__file__).with_name('DAG_APPROXIMATE_SHARING_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])))
if __name__=='__main__':main()
