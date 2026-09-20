import itertools,json,time
from pathlib import Path
import torch
from quadratic_student_fit import gram
from audit_centered_compact import evaluate
P=Path(__file__).resolve().parent

def main():
 start=time.perf_counter();torch.set_num_threads(4);torch.set_default_dtype(torch.float64);key=('centered_floor01',8,8,'muon',1);source=torch.load(P/'NATIVE_CENTERED_COMPACT_V1.pt',weights_only=True);s={k:v.double() for k,v in source['students'][key].items()};panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];M=panels[0]['covariance'].double();M=(M+M.T)/2;ev,V=torch.linalg.eigh(M);L=(V*ev.clamp_min(.01*float(M.trace()/1152)).sqrt())@V.T;A,B,C=[s[k] for k in ['quadratic_left','quadratic_right','quadratic_writer']];a,b=A@L,B@L;G=gram(a,b);cross=C@G;norm=((C@G)*C).sum();tr=(a*b).sum(1);mean=s['constant']+C@tr;linear_norm=(s['linear_writer']@s['linear_reader']@L).square().sum();function_norm=mean.square().sum()+linear_norm+2*norm;allrows=[];rows=[];exports={};normal=[];means=[]
 for width in range(9):
  best=float('inf');saved=None
  for indices in itertools.combinations(range(8),width):
   idx=torch.tensor(indices,dtype=torch.long);g=G[idx][:,idx];x=cross[:,idx]
   if width:
    writer=torch.linalg.lstsq(g,x.T,rcond=1e-12,driver='gelsd').solution.T;equation=float((writer@g-x).norm()/x.norm());normal.append(equation);rank=int(torch.linalg.matrix_rank(g,rtol=1e-12))
   else:writer=C[:,:0];equation=0.;rank=0
   error=float(norm+((writer@g)*writer).sum()-2*(writer*x).sum());allrows.append(dict(width=width,indices=list(indices),coefficient_squared_error=error,normal_residual=equation,gram_rank=rank))
   if error<best:best=error;saved=(idx,writer)
  idx,writer=saved;model={k:v.clone() for k,v in s.items()};model.update(quadratic_left=A[idx],quadratic_right=B[idx],quadratic_writer=writer,constant=mean-writer@tr[idx]);means.append(float((model['constant']+writer@tr[idx]-mean).norm()/mean.norm()));price=sum(v.numel() for v in model.values());assert price==20736+3456*width;diagnostics=[]
  for panel,y in zip(panels,targets):
   prediction=evaluate(model,panel['rows'].double());base=evaluate(s,panel['rows'].double());pc=prediction-prediction.mean(0);yc=y-y.mean(0);diagnostics.append(dict(full_quartic_error=float((prediction-y).norm()/y.norm()),centered_variation_error=float((pc-yc).norm()/yc.norm()),edit_error_relative_to_frozen=float((prediction-base).norm()/base.norm())))
  row=dict(width=width,indices=idx.tolist(),scalar_count=price,products=width,quadratic_retained_energy=1-best/float(norm),gaussian_function_relative_error=max(0,2*best/float(function_norm))**.5,diagnostics=diagnostics);rows.append(row);exports[width]={k:v.float() for k,v in model.items()};print(row,flush=True)
 errors=[1-r['quadratic_retained_energy'] for r in rows];fullreplay=max(r['edit_error_relative_to_frozen'] for r in rows[-1]['diagnostics']);pred=dict(pred_a=fullreplay<1e-8 and max(normal)<1e-8 and max(means)<1e-10 and all(y<=x+1e-10 for x,y in zip(errors,errors[1:])),pred_b=rows[4]['quadratic_retained_energy']>=.95,pred_c=rows[4]['diagnostics'][1]['full_quartic_error']-rows[8]['diagnostics'][1]['full_quartic_error']<.03);result=dict(source_key=list(key),records=rows,subset_scores=allrows,predictions=pred,max_normal_residual=max(normal),max_gaussian_mean_residual=max(means),full_replay=fullreplay,seconds=time.perf_counter()-start,scope='Exhaustive product-deletion frontier within frozen8product dictionary; analytic writer/constant refit against frozen student. Reused empiricaldiagnostics do notselect.');(P/'CENTERED_PRODUCT_PRUNE_V1.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(programs=exports,teacher_scale=source['teacher_scale']),P/'CENTERED_PRODUCT_PRUNE_V1.pt');print('SUMMARY',pred)
if __name__=='__main__':main()
