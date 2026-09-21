"""Export leading original-weight scalar-observer modes, no native-panel fitting."""
from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
out=p/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt';assert not out.exists()
e=torch.load(p/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True);rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
n=rows['n'].flatten(0,1).double()-e['mean_n'];m=rows['m'].flatten(0,1).double()-e['mean_m'];roots=[];inverses=[]
for x in [n,m]:
 M=x.T@x/len(x);ev,V=torch.linalg.eigh(M);mask=ev>1e-10*ev.max();cut=ev.clamp_min(0)*mask;roots.append((V*cut.sqrt())@V.T);inverses.append((V*torch.where(mask,ev.clamp_min(1e-30).rsqrt(),0))@V.T)
U,s,Vh=torch.linalg.svd(roots[0]@e['native_matrix']@roots[1],full_matrices=False)
A=(inverses[0]@U[:,:8])*s[:8].sqrt();B=(inverses[1]@Vh[:8].T)*s[:8].sqrt();exports={k:e[k] for k in ['q','native_matrix','writer','mean_n','mean_m','R_U','frozen_a','frozen_b']};exports.update(A=A,B=B)
checks=[];records=[]
for rank in [1,2,3,8]:
 error=float((roots[0]@(A[:,:rank]@B[:,:rank].T-e['native_matrix'])@roots[1]).norm()/(roots[0]@e['native_matrix']@roots[1]).norm());optimal=float(s[rank:].norm()/s.norm());assert abs(error-optimal)<1e-8
 records.append(dict(rank=rank,weighted_coefficient_error=error,local_weight_coefficients=1152*(2*rank+1),first_to_second_singular_ratio=float(s[0]/s[1])))
torch.save(exports,out);(p/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.json').write_text(json.dumps(dict(records=records,scope='Fixed learned output observer; globally optimal rank-r scalar bilinear approximation in calibration-product metric, explicit original-input centering and original native matrix. Rank1/2/3/8 exported before native effect evaluation, not semantic unit count.'),indent=2)+'\n');print(records)
