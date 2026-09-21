"""Prepare one fixed-cost gradient-directed node exchange and a random control."""
import json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from conditional_source_constraints import ConditionalSourceConstraints
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);receipt=json.loads((P/'CONVEX_SOURCE_V2.json').read_text())['records'][0];q=receipt['quadratics'];G=np.array(q['G']);b=np.array(q['b']);c=np.array(q['c']);x=np.array(receipt['solver']['x']);ablations=[]
for k in range(3,14):
 trial=x.copy();trial[k]=0;value=float((np.einsum('i,kij,j->k',trial,G,trial)+2*b@trial+c).max());ablations.append(dict(global_index=k,maximum=value))
remove=min(ablations,key=lambda r:r['maximum'])['global_index'];local=remove-3
base=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text())['baseline'];data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);graph=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)['calibration_shaped_inherited'];bundle=expand(graph);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);metric=ConditionalSourceConstraints(data,H,base);S=metric.S;inv=data['inverse_root'];second=[];offset=0
with torch.no_grad():
 for j,n in enumerate(receipt['allocation']):
  eig,U=torch.linalg.eigh(S@(metric.true[2*j+1]-H[2*j+1])@S);ix=eig.abs().argsort(descending=True)[:n];amp=torch.tensor(x[offset:offset+n],dtype=H.dtype);second.append(S@H[2*j+1]@S+(U[:,ix]*(eig[ix]*amp))@U[:,ix].T);offset+=n
K=torch.stack(second).requires_grad_(True);w=torch.tensor(receipt['dual_weights'],dtype=H.dtype);gradient=torch.autograd.grad(w@metric.ratios(inv@K@inv),K)[0][2];eig,U=torch.linalg.eigh((gradient+gradient.T)/2);v=U[:,eig.abs().argmax()].detach();rng=torch.Generator().manual_seed(53901);random=torch.randn(v.shape,generator=rng,dtype=v.dtype);random/=random.norm()
for kind,vector in [('gradient',v),('random',random)]:
 plan=json.loads((P/'CONVEX_SOURCE_PLAN_V1.json').read_text());plan.update(created_utc=datetime.now(timezone.utc).isoformat(),geometries=['calibration_shaped'],allocations=[[2,1,11]],replacement=dict(pair=2,column=local,metric_direction=vector.tolist(),kind=kind,removed_global_index=remove,ablation_profiles=ablations),scope='Single node exchange on opened states. Hold14 correction products and original fidelity/cost bars. Gradient-directed proposal versus one seeded random replacement; one random control is descriptive, not a distributional significance test.')
 (P/f'SOURCE_EXCHANGE_{kind.upper()}_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Prepared two replacements at local index',local,'baseline maximum',receipt['solver']['maximum'])
