from pathlib import Path
import json,torch
from shared_linear_source_graph import source_reads,component_scalars,expand,factor,arithmetic
from compact_source_graph import source_reads as reference,component_scalars as reference_components
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
rows=[]
for seed in range(5):
 g=torch.Generator().manual_seed(24000+seed);dtype=torch.float64
 def rand(*shape):return torch.randn(*shape,dtype=dtype,generator=g)
 d,m,v,r=16,10,2,seed+2;basis=torch.linalg.qr(rand(d,r)).Q
 p=dict(left_reader=basis@rand(r,m),right_reader=basis@rand(r,m),square_reader=basis@rand(r,v),product_weights=rand(m,6),square_weights=rand(v),square_output=torch.tensor(5),source_linear=rand(d,6),source_bias=rand(6),h_readers=rand(d,3),alpha=rand(3),beta=rand(3),residual_writer=rand(d))
 z=rand(29,d);h=rand(29,d);joint=torch.cat([p[k] for k in ('left_reader','right_reader','square_reader')],1);U,s,_=torch.linalg.svd(joint,full_matrices=False);q=factor(p,U[:,:r]);E=expand(q)
 err=float((source_reads(z,q)-reference(z,p)).norm()/reference(z,p).norm());valueerr=float((component_scalars(z,h,q)-reference_components(z,h,p)).norm()/reference_components(z,h,p).norm());readererr=max(float((E[k]-p[k]).norm()/p[k].norm()) for k in ('left_reader','right_reader','square_reader'))
 assert max(err,valueerr,readererr)<1e-12
 price=arithmetic(q);assert price['source_total_multiplications']==r*(d+2*m+v)+6*m+v+m+v
 rows.append(dict(seed=seed,planted_shared_linear_width=r,source_replay=err,component_replay=valueerr,reader_replay=readererr,price=price))
(P/'SHARED_LINEAR_GRAPH_PREFLIGHT_V1.json').write_text(json.dumps(dict(controls=rows),indent=2)+'\n');print('PASS five planted shared linear dictionaries, execution and literal arithmetic')
