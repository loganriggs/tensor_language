"""Share linear output directions using separate input-role covariance."""
from pathlib import Path
import json,torch
from midpoint_program import product_source_delta
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);out=p/'MIDPOINT_ROLE_SHARED_LINEAR_V1.json';assert not out.exists()
old=torch.load(p/'MIDPOINT_ADAPTIVE_SHARED_LINEAR_GRAPHS_V1.pt',weights_only=True);e=old['exact_linear'];programs={'exact_linear':e,'paired256':old['linear256']}
rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();nb=n.mean(0);mb=m.mean(0);S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();yn=(n-nb)@e['linear_n']@S;ym=(m-mb)@e['linear_m']@S;checks=[];records=[]
for metric,z in [('separate',torch.cat([yn,ym])),('source',ym)]:
 _,_,Vh=torch.linalg.svd(z,full_matrices=False)
 for rank in ([128,256] if metric=='separate' else [256]):
  Q=Vh[:rank].T;Hn=e['linear_n']@S@Q;Hm=e['linear_m']@S@Q;W=torch.linalg.solve(S,Q)
  new={k:v for k,v in e.items() if k not in ['linear_n','linear_m']};new.update(linear_left_reader=Hn,linear_right_reader=Hm,linear_writer=W,full_mean=e['full_mean']+nb@e['linear_n']+mb@e['linear_m']-(nb@Hn+mb@Hm)@W.T)
  dm=m[:32].roll(3,0)-m[:32];nc=n[:32]-nb;before=product_source_delta(e,nc,dm,True);after=product_source_delta(new,nc,dm,True);error=float((after-before).norm()/before.norm());assert error<1e-12;checks.append(error)
  programs[f'{metric}{rank}']=new;records.append(dict(metric=metric,rank=rank,weight_coefficients=3*1152*(512+rank)))
artifact=p/'MIDPOINT_ROLE_SHARED_LINEAR_GRAPHS_V1.pt';assert not artifact.exists();torch.save(programs,artifact)
out.write_text(json.dumps(dict(checks=checks,records=records,scope='Fixed adaptive interaction; new linear output basis from separate calibration roles or source alone. Native context interaction protected exactly; no held fitting.'),indent=2)+'\n');print(out.read_text())
