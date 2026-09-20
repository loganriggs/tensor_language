import json,time
from pathlib import Path
import torch
from arithmetic_dag import DAG
from dag_edit_refit import one_round
from dag_square_optimizer import rule
P=Path(__file__).resolve().parent

def main():
 start=time.perf_counter();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);xx,w=rule(5);x=xx[:,:4];torch.manual_seed(910);xv=torch.randn(4096,4);records=[]
 for case in ['near_duplicate','independent']:
  d=DAG(degree_limit=4);a,b,c,e=[d.input(i) for i in range(4)];ab=d.product(a,b)
  other=d.product(d.linear([(a,2),(c,.05)]),d.linear([(b,.5)])) if case=='near_duplicate' else d.product(c,e)
  out=[d.linear([(ab,.7),(other,.3)]),d.linear([(ab,-.4),(other,.8)])]
  def target(z):
   v=z[:,0]*z[:,1];return torch.stack([v,.4*v if case=='near_duplicate' else z[:,2]*z[:,3]],1)
  model,row=one_round(d,out,x,target(x),w);row['case']=case
  if model:
   with torch.no_grad():prediction=model(xv);row['fresh_relative_error']=float((prediction-target(xv)).norm()/target(xv).norm());graph,outputs=model.export();row['replay_error']=float((graph.evaluate(outputs,xv)-prediction).norm()/prediction.norm());row['degree']=max(graph.degrees[k] for k in outputs);row['accepted_price']=model.price()
  records.append(row);print(row,flush=True)
 a,b=records;pred=dict(pred_a=a['accepted'] and a['accepted_price']['products']==1 and a['fresh_relative_error']<.01,pred_b=not b['accepted'],pred_c=all(r.get('replay_error',0)<1e-10 and r.get('degree',0)<=4 for r in records));result=dict(records=records,predictions=pred,seconds=time.perf_counter()-start,scope='First approximate delete/refit/accept loop on opposing planted cases. No feature discovery or native generality claim.');(P/'DAG_EDIT_REFIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',pred)
if __name__=='__main__':main()
