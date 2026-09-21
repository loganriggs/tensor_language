"""Check whether a modal swap or covariance mode drift explains candidate changes."""
from pathlib import Path
import json,time,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();e=torch.load(p/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True);cal=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);roots=[]
for key,mean in [('n','mean_n'),('m','mean_m')]:
 x=cal[key].flatten(0,1).double()-e[mean];ev,V=torch.linalg.eigh(x.T@x/len(x));roots.append((V*(ev.clamp_min(0)*(ev>1e-10*ev.max())).sqrt())@V.T)
N,M=roots;native=N@e['native_matrix']@M;u,s,vh=torch.linalg.svd(native,full_matrices=False);mode=s[0]*torch.outer(u[:,0],vh[0]);observer=e['q']@e['R_U'];records=[]
for name,file in [('pruned','FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt'),('response_refit','FULL_CHANNEL_RESPONSE_PROGRAM_V1.pt'),('learned','LEARNED_FULL_QUADRATIC_PROGRAM_V1.pt')]:
 prog={k:v.double() for k,v in torch.load(p/file,weights_only=True).items()};a,b=prog['a'],prog['b'];channel=observer@prog['writer'];matrix=a.T@(channel[:,None]*b)+b.T@(channel[:,None]*a);weighted=N@matrix@M;us,ss,vs=torch.linalg.svd(weighted,full_matrices=False);candidate=ss[0]*torch.outer(us[:,0],vs[0]);records.append(dict(program=name,covariance_observer_error=float((weighted-native).norm()/native.norm()),leading_mode_cosine=float((candidate*mode).sum()/(candidate.norm()*mode.norm())),leading_mode_relative_error=float((candidate-mode).norm()/mode.norm()),leading_to_second_singular_ratio=float(ss[0]/ss[1]),leading_energy_fraction=float(ss[0].square()/ss.square().sum())))
result=dict(records=records,native_leading_to_second_singular_ratio=float(s[0]/s[1]),seconds=time.monotonic()-start,scope='Post-result mode-identity diagnostic in the same original observer and covariance geometry. A large leading singular gap limits a mode-order-switch explanation; close mode cosine is not native causal equivalence or unique semantics.')
(p/'LEARNED_CONTINUATION_MODE_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
