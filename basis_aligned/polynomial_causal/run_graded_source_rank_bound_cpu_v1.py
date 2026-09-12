"""CPU native fixed-rank bound; no text, no competing GPU work."""
from pathlib import Path
import json,time,hashlib
import torch
from graded_source_rank_bound_v1 import bound
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);tic=time.perf_counter();dtype=torch.float64
    alpha=.8;a=torch.stack([torch.diag(torch.tensor([1.,-alpha**.5],dtype=dtype)),torch.diag(torch.tensor([1.,alpha**.5],dtype=dtype))])
    toy=bound(a,torch.eye(2,dtype=dtype),torch.ones(1,1,dtype=dtype),1)
    assert abs(toy['loss_lower_bound']-alpha**2/(1+alpha**2))<1e-12
    path=P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_INPUTS.pt';receipt=json.loads((P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_RESULT.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert sha(path)==receipt['artifact_sha']
    data=torch.load(path,weights_only=True,map_location='cpu');root=data['root'];root=root/root.square().sum().div(len(root)).sqrt()
    native=bound(data['forms'],root,data['writer_gram'],128)
    assert abs(native['trace']-1)<1e-10 and native['minimum_eigenvalue']>=-1e-10
    result=dict(toy=toy,native=native,inputs_sha=sha(path),source_sha=sha(Path(__file__)),helper_sha=sha(P/'graded_source_rank_bound_v1.py'),seconds=time.perf_counter()-tic,
                scope='All rank128 common projectors under the declared formal graded coefficient metric; not a lower bound for arbitrary circuits or behavioral error.')
    out=P/'GRADED_SOURCE_RANK_BOUND_CPU_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
