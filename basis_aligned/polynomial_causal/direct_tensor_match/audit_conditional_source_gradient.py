from pathlib import Path
import json,time,torch
from conditional_source_constraints import ConditionalSourceConstraints
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);start=time.monotonic()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);base=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text())['baseline'];receipt=json.loads((P/'CONVEX_SOURCE_V2.json').read_text())['records'][0];graph=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)['calibration_shaped_inherited'];bundle=expand(graph);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);metric=ConditionalSourceConstraints(d,H,base);S=metric.S;inverse=d['inverse_root'];second=[];offset=0
with torch.no_grad():
 for j,n in enumerate(receipt['allocation']):
  e,U=torch.linalg.eigh(S@(metric.true[2*j+1]-H[2*j+1])@S);ix=e.abs().argsort(descending=True)[:n];amplitudes=torch.tensor(receipt['solver']['x'][offset:offset+n],dtype=H.dtype);K=S@H[2*j+1]@S+(U[:,ix]*(e[ix]*amplitudes))@U[:,ix].T;second.append(K);offset+=n
 K=torch.stack(second)
K.requires_grad_(True);weights=torch.tensor(receipt['dual_weights'],dtype=H.dtype);ratios=metric.ratios(inverse@K@inverse);replay=float((ratios-torch.tensor(receipt['solver']['values'],dtype=H.dtype)).abs().max());assert replay<1e-8
loss=weights@ratios;gradient=torch.autograd.grad(loss,K)[0];rows=[]
for j in range(3):
 G=(gradient[j]+gradient[j].T)/2;e,U=torch.linalg.eigh(G);index=e.abs().argmax();v=U[:,index];direction=torch.zeros_like(K);direction[j]=-e[index].sign()*torch.outer(v,v);analytic=float((gradient*direction).sum());epsilon=1e-6
 with torch.no_grad():
  plus=weights@metric.ratios(inverse@(K+epsilon*direction)@inverse);minus=weights@metric.ratios(inverse@(K-epsilon*direction)@inverse);finite=float((plus-minus)/(2*epsilon));relative=abs(analytic-finite)/max(1,abs(analytic));assert relative<1e-5
 rows.append(dict(pair=j,gradient_frobenius=float(G.norm()),signed_square_derivative=analytic,finite_difference=finite,relative_discrepancy=relative))
out=dict(records=rows,ratio_replay=replay,dual_weights=receipt['dual_weights'],seconds=time.monotonic()-start,scope='Gradient of the active convex-combination objective in covariance coordinates at the completed primary fit. A negative weighted derivative does not guarantee decrease of the maximum constraint. No graph edit, saved candidate, cost or fresh validation claim.')
(P/'CONDITIONAL_SOURCE_GRADIENT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
