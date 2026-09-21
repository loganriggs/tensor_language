"""Screen cross corrections built from already-computed shared/private activations."""
from pathlib import Path
import json,torch
from pairwise_reader_graph import GROUPS
from quadratic_pair_blocks import compile_pair
def problem():
 P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'PAIRWISE_SUBSPACE_NATIVE_V1.json').read_text());states=torch.load(P/'PAIRWISE_SUBSPACE_NATIVE_STATES_V1.pt',weights_only=True);params=states[meta['primary']];S=torch.linalg.inv(d['inverse_root']);items=[];outside=0.
 for j,(a,b) in enumerate(GROUPS):
  T=S@torch.stack(d['pairs'][j]['Qs'])@S;U,R=torch.linalg.qr(torch.cat([params[a],params[b]],1),mode='reduced');V=torch.linalg.qr(params[3+j]-U@(U.T@params[3+j]),mode='reduced').Q;Z=torch.cat([U,V],1);core=Z.T@T@Z;B=core[:,:128,128:];C=core[:,128:,128:]
  E=torch.linalg.lstsq(torch.cat(list(C),1).T,torch.cat(list(B),1).T,driver='gelsd').solution.T;res=B-E@C;cp=compile_pair(C[0],C[1])['input_transform']
  # res=R K cp.T. These coordinates correspond to products of an existing
  # shared dictionary activation and an existing compiled private activation.
  left=R/R.norm(dim=0);right=cp/cp.norm(dim=0);theta=torch.linalg.solve(right,torch.linalg.solve(left,res).transpose(-1,-2)).transpose(-1,-2)
  replay=float((left@theta@right.T-res).norm()/res.norm());assert replay<1e-8
  norm=T.square().sum();outside+=float((T-Z@core@Z.T).square().sum()/(3*norm));items.append(dict(left=left,right=right,theta=theta,residual=res,norm=norm,replay=replay))
 return items,outside,meta
