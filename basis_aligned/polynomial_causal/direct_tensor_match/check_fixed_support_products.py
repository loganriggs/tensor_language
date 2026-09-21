from pathlib import Path
import json,torch
from fixed_support_products import FixedSupportProducts
from local_shared_reader_graph import product_factors
from pairwise_reader_graph import GROUPS
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
from pairwise_product_toy_fixture import fixture
for case in range(5):
 T,bases,private,maps,templates=fixture(case)
 # Add an outside-span target term so forgetting the orthogonal residual fails.
 original=T.clone();g=torch.Generator().manual_seed(36000+case)
 for j,(a,b) in enumerate(GROUPS):
  U=torch.linalg.qr(torch.cat([bases[a],bases[b],private[j]],1),mode='reduced').Q;v=torch.randn(len(T[0]),dtype=T.dtype,generator=g);v=v-U@(U.T@v);v=v/v.norm();T[2*j]+=v[:,None]*v[None,:]
 metric=FixedSupportProducts(T,bases,private,templates);params=[]
 for j in range(3):params.extend([maps[j].clone().requires_grad_(),(metric.spans[j].T@private[j]).detach().requires_grad_()])
 implicit,W,readers=metric.loss(params);dense=metric.loss(params,dense=True)[0];gi=torch.autograd.grad(implicit,params);ge=torch.autograd.grad(dense,params);gradient=max(float((a-b).norm()/(1+a.norm())) for a,b in zip(gi,ge));assert gradient<1e-8
 # Independently reconstruct full ambient tensors from physical product factors.
 errors=[]
 for j,reader in enumerate(readers):
  L,R=product_factors(metric.spans[j]@reader,templates[j]['product_indices']);raw=torch.einsum('ir,ro,jr->oij',L,W[j],R);H=(raw+raw.transpose(-1,-2))/2;errors.append((H-T[2*j:2*j+2]).square().sum()/T[2*j:2*j+2].square().sum())
 physical=float(torch.stack(errors).mean());reported=float(implicit.detach());assert abs(float((implicit-dense).detach()))<1e-8 and abs(physical-reported)<1e-7 and float(metric.outside)>0
 rows.append(dict(case=case,outside_squared_error=float(metric.outside),ambient_loss=physical,regularized_loss=reported,dense_gradient_replay=gradient))
(P/'FIXED_SUPPORT_PRODUCTS_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Five planted pairwise product graphs with added unrepresentable coefficient residual; reduced-space loss retains that error. No native fit or optimizer recovery.'),indent=2)+'\n');print('PASS five full-space/reduced-space losses and gradients with nonzero outside residual')
