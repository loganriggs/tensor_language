from pathlib import Path
import json,torch
from dual_geometry_source_metric import DualGeometrySourceMetric
from global_mixed_source_graph import export,source_reads
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);S=torch.linalg.inv(d['inverse_root']);T=torch.stack([q for p in d['pairs'] for q in p['Qs']])/d['scales'][:,None,None];rows=[]
for width in (367,560):
 torch.manual_seed(10220);params=[]
 for n in [width,width,32]:
  a=torch.randn(1152,n,dtype=T.dtype);a/=a.norm(dim=0);a.requires_grad_();params.append(a)
 L,R,V=params;m=DualGeometrySourceMetric(T,S,.5);loss,W,v=m.loss(L,R,V);dense=m.explicit(L,R,V,W,v);replay=abs(float((loss-dense).detach()));assert replay<1e-10
 gradients=torch.autograd.grad(loss,params);assert all(torch.isfinite(g).all() for g in gradients)
 with torch.no_grad():
  extra=torch.zeros(6,32,dtype=T.dtype);extra[5]=v
  big=export(S@torch.cat([L,V],1),S@torch.cat([R,V],1),torch.cat([W,extra],1),d);p={k:a.clone() for k,a in big.items()}
  for k in ('left_reader','right_reader'):p[k]=big[k][:,:width].clone()
  p['product_weights']=big['product_weights'][:width].clone();p['square_reader']=big['left_reader'][:,width:].clone();p['square_weights']=big['product_weights'][width:,5].clone();p['square_output']=torch.tensor(5)
  z=d['z'][:32];actual=source_reads(z,p);actual[:,5]+=(z@p['square_reader']).square()@p['square_weights'];ref=source_reads(z,big);execution=float((actual-ref).norm()/ref.norm());price=sum(a.numel() for a in p.values() if a.is_floating_point());assert price==2310*width+48428 and execution<1e-10
 rows.append(dict(width=width,dense_replay=replay,gradient_norms=[float(g.norm()) for g in gradients],export_replay=execution,stored_floats=price))
(P/'DUAL_GEOMETRY_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
