"""Compare trace-free coefficient metrics to explicit symmetric tensors."""
import itertools,json
from pathlib import Path
import torch
from coupled_quartic_writer_v1 import gram
from quartic_harmonic_v1 import lower
from quartic_harmonic_metric_v1 import metric

torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(91861)
d=5
def sym(t):return sum(t.permute(*order) for order in itertools.permutations(range(4)))/24
def harmonic(t):
    trace=torch.einsum('abcc->ab',t);matrix,constant=lower(trace)
    eye=torch.eye(d)
    return t-sym(torch.einsum('ab,cd->abcd',eye,matrix))-constant*sym(torch.einsum('ab,cd->abcd',eye,eye))
b=torch.linalg.qr(torch.randn(3,d,2),mode='reduced')[0];n=torch.randn(3,2)
q=torch.einsum('kdi,ki,kei->kde',b,n,b)
f=torch.stack([sym(torch.einsum('ab,cd->abcd',m,m)) for m in q])
t=torch.stack([sym(torch.randn(d,d,d,d)) for _ in range(2)])
hf=torch.stack([harmonic(z) for z in f]);ht=torch.stack([harmonic(z) for z in t])
c=f.flatten(1)@t.flatten(1).T;k=gram(b,n)
hk,hc=metric(k,c,b,n,torch.einsum('mabcc->mab',t))
dk=hf.flatten(1)@hf.flatten(1).T;dc=hf.flatten(1)@ht.flatten(1).T
error=max(float((hk-dk).norm()/dk.norm()),float((hc-dc).norm()/dc.norm()))
traceerror=float(torch.einsum('mabcc->mab',ht).abs().max())
r=dict(pred_a=max(error,traceerror)<=1e-10,metric_relative_error=error,harmonic_trace_error=traceerror)
out=Path(__file__).resolve().parent/'QUARTIC_HARMONIC_METRIC_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(r);assert r['pred_a']
