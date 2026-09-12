from pathlib import Path
import json,itertools,torch,numpy as np
from composed_quartic_contraction_v1 import contract
from quartic_repeated_input_v1 import dense_trace,repeated_inner,square_traces,native_traces
from coupled_quartic_writer_v1 import gram
P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(91720)
d=3;w=[torch.randn(4,d),torch.randn(4,d),torch.randn(d,4),torch.randn(5,d),torch.randn(5,d),torch.randn(2,5)];scale=1.3
eye=torch.eye(d);ids=torch.tensor(list(itertools.product(range(d),repeat=4)));t=contract(eye[ids],*w,scale).T.reshape(2,d,d,d,d)
a=dense_trace(t);fast=native_traces(*w,scale);trace_error=float((a-fast).norm()/a.norm());k=t.flatten(1)@t.flatten(1).T;analytic=repeated_inner(k,a,a)
nodes,weights=np.polynomial.hermite.hermgauss(5);nodes=torch.tensor(nodes)*2**.5;weights=torch.tensor(weights)/np.pi**.5
ids=torch.tensor(list(itertools.product(range(5),repeat=d)));x=nodes[ids];qw=weights[ids].prod(-1);values=torch.einsum('oabcd,na,nb,nc,nd->no',t,x,x,x,x);exact=values.T@(qw[:,None]*values)
quad_error=float((analytic-exact).norm()/exact.norm())
b=torch.randn(3,d,2);nu=torch.randn(3,2);q=torch.einsum('kdi,ki,kei->kde',b,nu,b);sq=(torch.einsum('kab,kcd->kabcd',q,q)+torch.einsum('kac,kbd->kabcd',q,q)+torch.einsum('kad,kbc->kabcd',q,q))/3
st=square_traces(b,nu);square_error=float((st-dense_trace(sq)).norm()/st.norm());sg=repeated_inner(gram(b,nu),st,st);sv=(torch.einsum('na,kab,nb->nk',x,q,x)).square();sg_exact=sv.T@(qw[:,None]*sv);square_metric_error=float((sg-sg_exact).norm()/sg_exact.norm())
r=dict(pred_a=trace_error<=1e-10,pred_b=quad_error<=1e-10,pred_c=max(square_error,square_metric_error)<=1e-10,native_trace_relative_error=trace_error,quadrature_relative_error=quad_error,square_trace_relative_error=square_error,square_metric_relative_error=square_metric_error)
p=P/'QUARTIC_REPEATED_INPUT_V1_CONTROL.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(r);assert r['pred_a'] and r['pred_b'] and r['pred_c']
