import json,time
from pathlib import Path
import torch
from empirical_quartic_dictionary import features,fit,evaluate,controls
from shared_quadratic_bank import normalize_bank

def main():
 torch.set_num_threads(2);start=time.monotonic();check=controls();rows=[]
 for index,family in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
  torch.manual_seed(2870+index);rand=lambda *s:torch.randn(*s,dtype=torch.float64);u,v=rand(3,2,4),rand(3,2,4);c=rand(5,6)
  if family=='shared_input':u[:]=u[0]
  if family=='shared_output':c=c[:,:1]@rand(1,6)
  if family=='squares':v=u.clone()
  if family=='cancellation':u[2]=u[0];v[2]=v[0];c[:,5]=-c[:,0]
  u,v=normalize_bank(u,v);x,xv=rand(256,4),rand(1024,4);y=features(x,u,v)@c.T;yv=features(xv,u,v)@c.T
  for optimizer in ['adam','muon']:
   for seed in [0,1]:
    row,program=fit(x,y,3,2,steps=300,rate=.03,optimizer=optimizer,seed=2890+seed);row.update(family=family,optimizer=optimizer,seed=seed,validation_error=float((evaluate(program,xv)-yv).norm()/yv.norm()));rows.append(row)
  print(json.dumps(dict(family=family,best_validation={op:min(r['validation_error'] for r in rows if r['family']==family and r['optimizer']==op) for op in ['adam','muon']})),flush=True)
 wins={op:sum(r['validation_error']<.01 for r in rows if r['optimizer']==op) for op in ['adam','muon']};chosen=min(wins,key=lambda op:(-wins[op],sum(r['validation_error'] for r in rows if r['optimizer']==op)))
 result=dict(controls=check,records=rows,recoveries_below_one_percent=wins,selected_optimizer=chosen,seconds=time.monotonic()-start,scope='Five planted quadratic dictionaries, two random starts per Adam/Muon at one shared learning rate and300steps. Functional fit on artificial Gaussian probes with fresh probes; no coefficient or semantic-identification claim. Optimizer choice scoped to this pilot.')
 Path(__file__).with_name('EMPIRICAL_QUARTIC_DICTIONARY_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(recoveries=wins,selected_optimizer=chosen,seconds=result['seconds'])))
if __name__=='__main__':main()
