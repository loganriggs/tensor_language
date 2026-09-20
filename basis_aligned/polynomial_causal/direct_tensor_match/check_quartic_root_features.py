import itertools,json
from pathlib import Path
import torch
from quartic_root_features import root_entries
from implicit_quartic import entries
P=Path(__file__).resolve().parent;torch.set_default_dtype(torch.float64);torch.manual_seed(1717);d=4;C,L,R,D,A,B=[torch.randn(*s) for s in [(3,5),(5,6),(5,6),(6,7),(7,d),(7,d)]];idx=torch.tensor(list(itertools.product(range(d),repeat=4)));roots=root_entries(L,R,D,A,B,idx);target=entries(C,L,R,D,A,B,idx);replay=float((roots@C.T-target).norm()/target.norm());x=torch.randn(9,d);h=((x@A.T)*(x@B.T))@D.T;direct=((h@L.T)*(h@R.T));query=torch.einsum('aijkl,ni,nj,nk,nl->na',roots.T.reshape(5,d,d,d,d),x,x,x,x);err=float((query-direct).norm()/direct.norm());assert max(replay,err)<1e-11
out=dict(readout_replay_relative_error=replay,dense_polynomial_relative_error=err,scope='Native quadratic-product root features, exact symmetric entry queries and direct polynomial replay. Fixed features, no optimization claim.')
(P/'QUARTIC_ROOT_FEATURE_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
