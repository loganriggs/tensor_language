import itertools,json,time
from pathlib import Path
import torch
from quadratic_student_fit import fit
from audit_centered_compact import evaluate
P=Path(__file__).resolve().parent

def main():
 start=time.perf_counter();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);core=torch.load(P/'CENTERED_QUADRATIC_CAPACITY_V1.pt',weights_only=True);capacity=json.load(open(P/'CENTERED_QUADRATIC_CAPACITY_V1.json'));C,A,B=[core[k] for k in ['C','A','B']];norm=capacity['weighted_quadratic_energy'];bounds=capacity['output_rank_retention_upper_bounds'];source=torch.load(P/'NATIVE_CENTERED_COMPACT_V1.pt',weights_only=True);s={k:v.double() for k,v in source['students'][core['source_key']].items()};panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];trace_teacher=core['output_frame']@(C@(A*B).sum(1));records=[];best={}
 for width,optimizer,lr,seed in itertools.product([3,4,6],['adam','muon'],[.005,.03],[0,1]):
  row,(a,b,c)=fit(C,A,B,width,torch.eye(16),optimizer,lr,seed,1000);retained=row['gain']/norm;assert retained<=bounds[width-1]+1e-8;row.update(width=width,optimizer=optimizer,lr=lr,seed=seed,retained_quadratic_energy=retained,quadratic_relative_error=max(0,1-retained)**.5);records.append(row)
  if width not in best or row['gain']>best[width][0]['gain']:best[width]=(row,a,b,c)
  print(row,flush=True)
 exports={};winners=[]
 for width,(row,a,b,c) in best.items():
  model={k:v.clone() for k,v in s.items()};co=core['output_frame']@c;model.update(quadratic_left=a@core['input_mapback'],quadratic_right=b@core['input_mapback'],quadratic_writer=co,constant=s['constant']+trace_teacher-co@(a*b).sum(1));diagnostics=[]
  for panel,y in zip(panels,targets):
   x=panel['rows'].double();pred=evaluate(model,x);pc=pred-pred.mean(0);yc=y-y.mean(0);diagnostics.append(dict(full_quartic_error=float((pred-y).norm()/y.norm()),centered_variation_error=float((pc-yc).norm()/yc.norm())))
  delta=panels[0]['rows'][:32].double()-s['mu'];small=delta@core['input_mapback'].T;native=((delta@model['quadratic_left'].T)*(delta@model['quadratic_right'].T))@co.T;reduced=((small@a.T)*(small@b.T))@c.T@core['output_frame'].T;replay=float((native-reduced).norm()/native.norm());assert replay<1e-10;winner=dict(row,diagnostics=diagnostics,scalar_count=sum(t.numel() for t in model.values()),map_replay=replay);winners.append(winner);exports[width]={k:v.float() for k,v in model.items()};print('WINNER',winner,flush=True)
 four=next(r for r in winners if r['width']==4);pred=dict(pred_a=all(r['map_replay']<1e-10 for r in winners),pred_b=four['retained_quadratic_energy']>=.95,pred_c=four['diagnostics'][1]['full_quartic_error']-.231228871375852<.02);out=dict(records=records,winners=winners,predictions=pred,seconds=time.perf_counter()-start,scope='Continuous refactor of frozenquadraticstudent in exactweighted16input/8outputcore. Fits neverseeempiricaltargets. Diagnosticpanels reused; no globalCPoptimality or semanticidentityclaim.');(P/'CENTERED_CONTINUOUS_REFACTOR_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(dict(programs=exports,teacher_scale=source['teacher_scale']),P/'CENTERED_CONTINUOUS_REFACTOR_V1.pt');print('SUMMARY',pred)
if __name__=='__main__':main()
