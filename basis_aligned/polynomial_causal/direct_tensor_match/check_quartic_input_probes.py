import itertools,json
from pathlib import Path
import torch
from implicit_quartic import entries
from quartic_input_probes import input_probes
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1806);d=5;o=3;teacher=[torch.randn(*s) for s in [(o,4),(4,5),(4,5),(5,6),(6,d),(6,d)]];idx=torch.tensor(list(itertools.product(range(d),repeat=4)));H=entries(*teacher,idx).T.reshape(o,d,d,d,d);b,c,e=[torch.randn(11,d) for _ in range(3)];g=torch.randn(11,o);got=input_probes(teacher,b,c,e,g);ref=torch.einsum('vijkl,nv,nj,nk,nl->ni',H,g,b,c,e);error=float((got-ref).norm()/ref.norm());assert error<1e-11
ix=torch.tensor(list(itertools.product(range(o),range(d),range(d),range(d))));v,j,k,l=ix.T;allprobes=input_probes(teacher,torch.eye(d)[j],torch.eye(d)[k],torch.eye(d)[l],torch.eye(o)[v]);unfold=H.permute(1,0,2,3,4).reshape(d,-1);G=unfold@unfold.T;replay=float((allprobes.T@allprobes-G).norm()/G.norm());assert replay<1e-11
out=dict(directional_adjoint_relative_error=error,exhaustive_unfolding_gram_error=replay,scope='Unbiased first-input Gram estimator uses independent isotropic probes in three inputslots and output. Finite-probe rank tail is not automatically a certified full-tensor bound.')
(P/'QUARTIC_INPUT_PROBE_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
