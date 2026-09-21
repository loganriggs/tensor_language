"""Optimistic rank bound for adding dense cross-product corrections to completion."""
from pathlib import Path
import json,torch
from pairwise_reader_graph import GROUPS
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);states=torch.load(P/'PAIRWISE_SUBSPACE_NATIVE_STATES_V1.pt',weights_only=True);meta=json.loads((P/'PAIRWISE_SUBSPACE_NATIVE_V1.json').read_text());key=meta['primary'];params=states[key];S=torch.linalg.inv(d['inverse_root']);energies=[];outside=0.;rows=[]
for j,(a,b) in enumerate(GROUPS):
 T=S@torch.stack(d['pairs'][j]['Qs'])@S;common=torch.cat([params[a],params[b]],1);U,_=torch.linalg.qr(common,mode='reduced');V=torch.linalg.qr(params[3+j]-U@(U.T@params[3+j]),mode='reduced').Q;Z=torch.cat([U,V],1);core=Z.T@T@Z;r=U.shape[1];B=core[:,:r,r:];C=core[:,r:,r:]
 E=torch.linalg.lstsq(torch.cat(list(C),1).T,torch.cat(list(B),1).T,driver='gelsd').solution.T
 residual=torch.cat(list(B-E@C),1);sv=torch.linalg.svdvals(residual);energy=2*sv.square()/(3*T.square().sum());energies.extend(energy.tolist());outside+=float((T-Z@core@Z.T).square().sum()/(3*T.square().sum()));rows.append(dict(pair=j+1,cross_unfolding_rank=int((sv>1e-10*sv[0]).sum()),normalized_cross_squared_error=float(energy.sum())))
energies=sorted(energies,reverse=True);total=sum(energies);guard=meta['plan']['covariance_guard'];best=[max(outside+total-sum(energies[:k]),0)**.5 for k in range(len(energies)+1)];minimum=next((i for i,e in enumerate(best) if e<=guard),None)
base_cost=1047648;budget=.8*1330560;per_atom=128+352+1+2;max_atoms=int((budget-base_cost)//per_atom)
out=dict(primary=key,pairs=rows,outside_span_error=outside**.5,initial_completion_error=best[0],coefficient_guard=guard,optimistic_required_cross_atoms=minimum,dense_atom_multiplications=per_atom,available_extra_multiplications=budget-base_cost,maximum_dense_cross_atoms=max_atoms,error_lower_bound_at_dense_budget=best[max_atoms],scope='Only additive cross corrections to this fixed common/private completion, with bothdiagonalblocks retained. Each bilinear cross atom contributes rank<=1 to shared-input unfolding. Per-pair SVD allocation is optimistic because output/right factors may require moreproducts. Cost assumes dense128left and352right maps from existing activations plusproductand2readouts. Not a bound on sparse maps, refitting old atoms, other groupings or arbitraryDAGs.')
(P/'COMPLETION_CROSS_RANK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
