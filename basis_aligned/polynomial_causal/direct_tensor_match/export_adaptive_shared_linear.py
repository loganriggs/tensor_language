"""Hold centered interaction fixed; share first-order linear reads and writes."""
from pathlib import Path
import json,torch
from midpoint_program import product_source_delta
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
out=p/'MIDPOINT_ADAPTIVE_SHARED_LINEAR_V1.json';assert not out.exists()
e=torch.load(p/'MIDPOINT_ADAPTIVE_MIXED_GRAPHS_V1.pt',weights_only=True)['adaptive_mixed512'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();nb=n.mean(0);mb=m.mean(0)
S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();target=(n-nb)@e['linear_n']+(m-mb)@e['linear_m'];_,s,Vh=torch.linalg.svd(target@S,full_matrices=False)
programs={'exact_linear':e};records=[]
for rank in [64,128,256]:
 Q=Vh[:rank].T;Hn=e['linear_n']@S@Q;Hm=e['linear_m']@S@Q;W=torch.linalg.solve(S,Q)
 new={k:v for k,v in e.items() if k not in ['linear_n','linear_m']};new.update(linear_left_reader=Hn,linear_right_reader=Hm,linear_writer=W,full_mean=e['full_mean']+nb@e['linear_n']+mb@e['linear_m']-(nb@Hn+mb@Hm)@W.T)
 dm=m[:32].roll(3,0)-m[:32];nc=n[:32]-nb;old=product_source_delta(e,nc,dm,True);new_delta=product_source_delta(new,nc,dm,True);replay=float((new_delta-old).norm()/old.norm());assert replay<1e-12
 meanreplay=float((new['full_mean']+(nb@Hn+mb@Hm)@W.T-e['full_mean']-nb@e['linear_n']-mb@e['linear_m']).norm());assert meanreplay<1e-8
 programs[f'linear{rank}']=new;records.append(dict(rank=rank,weight_coefficients=3*1152*512+3*1152*rank,linear_variation_error=float(s[rank:].norm()/s.norm()),context_replay=replay,mean_replay=meanreplay))
artifact=p/'MIDPOINT_ADAPTIVE_SHARED_LINEAR_GRAPHS_V1.pt';assert not artifact.exists();torch.save(programs,artifact)
out.write_text(json.dumps(dict(records=records,scope='Same adaptive512 centered interaction. Sharedlinear feature projections fit only paired calibration variation; constant corrected to preserve value at calibration input means. Context-only delta is algebraically invariant; source-only and full prediction may degrade. Full literal weights charged.'),indent=2)+'\n');print(out.read_text())
