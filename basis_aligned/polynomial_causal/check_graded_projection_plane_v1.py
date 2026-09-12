"""Verify exact one-plane search escapes the strict nonglobal planted minimum."""
from pathlib import Path
import json,hashlib
import torch
from graded_source_projection_v1 import graded_norms,balanced_loss
from graded_projection_plane_v1 import search
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
    torch.set_num_threads(2);dtype=torch.float64;alpha=.8;root=torch.eye(2,dtype=dtype)
    forms=torch.stack([torch.diag(torch.tensor([1.,-alpha**.5],dtype=dtype)),torch.diag(torch.tensor([1.,alpha**.5],dtype=dtype))]);wg=torch.ones(1,1,dtype=dtype);full=graded_norms(forms,root,wg)
    def value(theta):
        t=torch.tensor(theta,dtype=dtype);p=torch.stack([t.sin(),t.cos()])[:,None]
        return float(balanced_loss(forms,root,p,wg,full))
    result=search(value);expected=alpha**2/(1+alpha**2)
    assert result['interpolation_max_error']<1e-10 and abs(result['value']-expected)<1e-10 and result['improvement']>.2
    result.update(expected_global_plane_minimum=expected,source_sha=hashlib.sha256((P/'graded_projection_plane_v1.py').read_bytes()).hexdigest(),scope='Chosen-plane global solution on an analytic 2D counterexample; not a native optimization outcome.')
    out=P/'GRADED_PROJECTION_PLANE_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
