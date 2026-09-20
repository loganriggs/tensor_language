import io
import torch
from shared_response_runtime import compile_runtime,execute
from projected_bilinear_response import prepare_context,evaluate_prepared,prepare_readout,readout_prepared


def test_shared_product_runtime_serializes_and_matches_full_symmetric_cores():
    torch.manual_seed(655);r,d,batch=4,9,7
    programs=[];contexts=[];scales=[]
    for _ in range(3):
        core=torch.randn(r,r,r,dtype=torch.float64)/20
        P=torch.randn(d,r,dtype=torch.float64)/5
        p=dict(core=(core+core.transpose(-1,-2))/2,mixed=torch.randn(r,r,d,dtype=torch.float64)/20,
               carry=torch.eye(r,dtype=torch.float64),geometry=P.T@P,input_basis=P,width=d)
        h=torch.randn(batch,d,dtype=torch.float64)
        c=prepare_context(p,h,torch.randn(batch,r,dtype=torch.float64)/20,1e-7)
        programs.append(p);contexts.append(c);scales.append(torch.tensor(.8,dtype=torch.float64))
    fixed,final=prepare_readout(h,P,torch.randn(2,d,dtype=torch.float64),1e-7)
    runtime=compile_runtime(programs,scales,fixed)
    assert sum(b['coefficients'].numel() for b in runtime['blocks'])==3*r*r*(r+1)//2
    assert all('mixed' not in b and 'input_basis' not in b for b in runtime['blocks'])
    storage=io.BytesIO();torch.save(runtime,storage);storage.seek(0)
    loaded=torch.load(storage,weights_only=True)
    initial=torch.randn(batch,r,dtype=torch.float64)/4
    for amplitude in [-.5,0.,.5,1.,1.5]:
        z=initial*amplitude
        for p,c,scale in zip(programs,contexts,scales):z=evaluate_prepared(p,z*scale,c)
        expected=readout_prepared(fixed,z,final)
        actual=execute(loaded,initial*amplitude,contexts,final)
        torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
