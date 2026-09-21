"""Invertible output-metric checks, including fixed-direction readout invariance."""
from pathlib import Path
import torch,json
from source_sobolev import SourceSobolev
from shared_quadratic_products import materialize_mixed
P=Path(__file__).resolve().parent;torch.set_num_threads(2);records=[]
for seed in range(5):
 g=torch.Generator().manual_seed(823+seed);T=torch.randn(4,8,8,generator=g,dtype=torch.double);T=(T+T.transpose(-1,-2))/2
 L=torch.randn(8,4,generator=g,dtype=torch.double,requires_grad=True);R=torch.randn(8,4,generator=g,dtype=torch.double,requires_grad=True)
 e,U=torch.linalg.eigh(T.flatten(1)@T.flatten(1).T);power=[0,.25,.5,.75,1][seed];B=(U*e.pow(-power/2))@U.T;inverse=(U*e.pow(power/2))@U.T;Tb=torch.einsum('ab,bij->aij',B,T);metric=SourceSobolev(Tb,torch.eye(8,dtype=T.dtype),0)
 loss,W=metric.loss(L,R);explicit=metric.explicit(materialize_mixed(L,R,W),W);g1=torch.autograd.grad(loss,[L,R],retain_graph=True);loss2,_=metric.loss(L,R,detach=False);g2=torch.autograd.grad(loss2,[L,R]);_,W0=SourceSobolev(T,torch.eye(8,dtype=T.dtype),0).loss(L,R)
 replay=float((loss-explicit).abs().detach());gradient=max(float((a-b).abs().max()) for a,b in zip(g1,g2));readout=float(((inverse@W)-W0).norm().detach()/W0.norm().detach());assert max(replay,gradient,readout)<1e-10
 records.append(dict(seed=seed,power=power,dense_replay=replay,envelope_gradient_error=gradient,untransformed_readout_error=readout))
with torch.no_grad():
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);T=d['teacher'][:4];e,U=torch.linalg.eigh(T.flatten(1)@T.flatten(1).T);g=torch.Generator().manual_seed(816);L=torch.randn(1152,256,generator=g,dtype=torch.double);R=torch.randn(1152,256,generator=g,dtype=torch.double);L/=L.norm(dim=0);R/=R.norm(dim=0);_,W0=SourceSobolev(T,torch.eye(1152,dtype=T.dtype),0).loss(L,R);native=[]
 for power in [.5,1]:
  B=(U*e.pow(-power/2))@U.T;inverse=(U*e.pow(power/2))@U.T;Tb=torch.einsum('ab,bij->aij',B,T);m=SourceSobolev(Tb,torch.eye(1152,dtype=T.dtype),0);loss,W=m.loss(L,R);replay=float(abs(loss-m.explicit(materialize_mixed(L,R,W),W)));undo=float((inverse@W-W0).norm()/W0.norm());assert max(replay,undo)<1e-10;native.append(dict(power=power,initial_loss=float(loss),dense_replay=replay,untransformed_readout_error=undo))
out=dict(toys=records,native=native,all_pass=True,scope='Loss, gradient and coordinate-transform verification. These are not five new optimizer-recovery runs; native target output Gram is full rank. Fixed-direction optimal readouts are invariant after undoing invertible output weighting, so new effects must come from direction optimization.')
(P/'OUTPUT_BALANCE_PREFLIGHT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
