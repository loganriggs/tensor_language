"""Topology consequence: whole pencil-block sharing excludes explicit cross paths.
A mixed-slot proposal keeps the same literal budget. No fitting/adoption claim.
"""
from pathlib import Path
import json,torch
from local_shared_reader_graph import expand,factor_bundle,price
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
# Five exact fixed-input-space witnesses: xy cannot equal f(x)+g(y).
controls=[]
for seed in range(5):
 g=torch.Generator().manual_seed(30000+seed);U=torch.linalg.qr(torch.randn(8,8,dtype=torch.float64,generator=g)).Q;a=U[:,0];b=U[:,1];Q=(a[:,None]*b[None]+b[:,None]*a[None])/2
 # Orthogonal projection onto separate within-group quadratic blocks is zero.
 projected=torch.outer(a,a)*float(a@Q@a)+torch.outer(b,b)*float(b@Q@b);error=float((Q-projected).norm()/Q.norm());x=torch.randn(20,8,dtype=Q.dtype,generator=g);actual=(x@a)*(x@b);expected=torch.einsum('ni,ij,nj->n',x,Q,x);replay=float((actual-expected).norm()/expected.norm());assert abs(error-1)<1e-12 and replay<1e-12;controls.append(dict(seed=seed,separate_fixed_subspace_error=error,cross_product_execution_replay=replay))
programs=torch.load(P/'JOINT_OVERLAP_PROGRAMS_V1.pt',weights_only=True);meta=json.loads((P/'JOINT_OVERLAP_V1.json').read_text());rows=[]
for geometry,key in meta['winners'].items():
 program=programs[key];bundle=expand(program);basis=torch.linalg.qr(program['input_basis'],mode='reduced').Q;selections={};counts=[]
 for j in range(3):
  p=bundle[str(j)];indices=p['product_indices'];mask=torch.zeros(384,dtype=torch.bool);mask[program['pairs'][str(j)]['shared_indices']]=True;old=int(((mask[indices[0]]!=mask[indices[1]])&(indices[2]==2)).sum());assert old==0
  # One selected slot from the highest-energy complex blocks; then scalar blocks.
  reader=p['shared_reader'];fraction=(basis.T@reader).square().sum(0)/reader.square().sum(0);blocks=[]
  for row in torch.where(indices[2]==2)[0].tolist():
   a,b=indices[:2,row].tolist();u,v=reader[:,a],reader[:,b];energy=float(.5*(u.square().sum()*v.square().sum()+(u@v).square())*p['product_weights'][row].square().sum());chosen=a if fraction[a]>=fraction[b] else b;blocks.append((energy,chosen))
  chosen=[c for _,c in sorted(blocks,reverse=True)[:160]]
  scalar=indices[0,indices[2]==0].tolist();scalar.sort(key=lambda i:float(fraction[i]),reverse=True);chosen+=scalar[:160-len(chosen)];assert len(chosen)==len(set(chosen))==160
  selected=torch.tensor(sorted(chosen),dtype=torch.int64);selections[str(j)]=selected;mask_new=torch.zeros(384,dtype=torch.bool);mask_new[selected]=True;new=int(((mask_new[indices[0]]!=mask_new[indices[1]])&(indices[2]==2)).sum());counts.append(dict(mode=j+1,previous_cross_products=old,proposed_cross_products=new,total_complex_blocks=len(blocks)))
 proposed=factor_bundle(bundle,basis,selections);assert price(proposed)['stored_floats']==996876 and price(proposed)['source_total_multiplications']==986496
 rows.append(dict(geometry=geometry,source_program=key,pair_cross_counts=counts,price=price(proposed),shared_indices={k:v.tolist() for k,v in selections.items()}))
(P/'OVERLAP_CROSS_SLOT_TOPOLOGY_V1.json').write_text(json.dumps(dict(controls=controls,proposals=rows,scope='Exact algebraic comparison for fixed orthogonal input groups; not an impossibility statement when directions can change or overlap. Current whole-block selection has no direct shared-private quadratic products. Proposed mixed-slot selection permits these at unchanged storage/arithmetic but changes the approximating function and requires refitting. Weight-based heuristic, not a fitted native result.'),indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='shared_indices'} for r in rows],indent=2))
