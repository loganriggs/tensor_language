from pathlib import Path
import json,torch
from source_sobolev import SourceSobolev
from source_graph_metrics import export,score
from shared_quadratic_products import materialize_mixed
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);fit=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());p=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[fit['winner']];root=torch.linalg.inv(d['inverse_root']);s=p['shared_mixed'];L=root@s['left_reader'];R=root@s['right_reader'];L/=L.norm(dim=0);R/=R.norm(dim=0);T=d['teacher'][:4];H=d['inverse_root']@d['inverse_root'];records=[]
for lam in [0,.1,1,10]:
 m=SourceSobolev(T,H,lam);loss,W=m.loss(L,R);hat=materialize_mixed(L,R,W);replay=float(abs(loss-m.explicit(hat,W)));assert replay<1e-8
 pnew=export(L,R,W,d,p);rec=dict(lam=lam,objective=float(loss),dense_replay=replay,coefficient_error=float((hat-T).norm()/T.norm()),source_gradient_error=float(((hat-T)*(H@(hat-T))).sum().sqrt()/(T*(H@T)).sum().sqrt()),**score(pnew,d));records.append(rec);print(rec,flush=True)
base=score(p,d);primary=records[2];baseline=json.loads((P/'MULTIMODE_PROJECTION_SHARING_V1.json').read_text())['baseline_errors']
out=dict(records=records,parent=base,predictions=dict(pred_a_instrument=max(r['dense_replay'] for r in records)<1e-8,pred_b_sensitivity=all(a<=.9*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),pred_c_values=all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],baseline))),scope='Fixed256mixed product directions, exact readout solves under changed source metrics. Private third unchanged. Opened scalar/Jacobian screen, no new native forwards.')
(P/'SOURCE_SOBOLEV_READOUT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'])
