from pathlib import Path
import torch,json
from global_mixed_source_graph import export,source_reads,component_scalars
from source_sobolev import SourceSobolev
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);oldmeta=json.loads((P/'MIXED_PRODUCT_NATIVE_FIT_V1.json').read_text());old=torch.load(P/'MIXED_PRODUCT_NATIVE_PROGRAMS_V1.pt',weights_only=True);checks=[]
for row in oldmeta['records']:
 p=old[row['key']];phi=component_scalars(d['z'],d['h'],p);errors=[]
 for j,pair in enumerate(d['pairs']):
  truth=pair['truth'][d['indices']];errors.append(float((phi[d['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
 replay=max(abs(a-b) for a,b in zip(errors,row['per_mode_errors']));assert replay<1e-8;checks.append(dict(key=row['key'],scalar_metric_replay=replay))
g=torch.Generator().manual_seed(834);L=torch.randn(1152,383,generator=g,dtype=torch.double);R=torch.randn(1152,383,generator=g,dtype=torch.double);L/=L.norm(dim=0);R/=R.norm(dim=0);_,W=SourceSobolev(d['teacher'],torch.eye(1152,dtype=L.dtype),0).loss(L,R);p=export(L,R,W,d);z=d['z'][:32];raw=torch.einsum('ir,ro,jr->oij',p['left_reader'],p['product_weights'],p['right_reader']);Q=(raw+raw.transpose(-1,-2))/2;dense=torch.einsum('ni,oij,nj->no',z,Q,z)+z@p['source_linear']+p['source_bias'];replay=float((dense-source_reads(z,p)).norm()/dense.norm());assert replay<1e-8
floats=sum(v.numel() for v in p.values());assert floats==896262
out=dict(historical_checks=checks,new383_dense_replay=replay,source_products=383,stored_floats=floats,scalar_shape=list(component_scalars(z,d['h'][:32],p).shape),scope='Exporter/executor preflight only; new random383program is not trained or promoted. Reuses historical global384program format; all3components present, no privatebranch.')
(P/'GLOBAL_SOURCE_EXPORT_PREFLIGHT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
