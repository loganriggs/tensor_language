import pytest
import torch
import module_write_mediation_contract as c


class Attention(torch.nn.Module):
    def forward(self,x): return x+1,"state"

def test_capture_replace_and_hook_removal():
    attn,mlp=Attention(),torch.nn.Identity(); sites={"attn12":(attn,"attention"),"mlp12":(mlp,"mlp")}; x=torch.zeros(1,2,3)
    output,writes=c.capture_writes(sites,lambda:mlp(attn(x)[0]))
    assert torch.equal(output,torch.ones_like(x)); assert set(writes)==set(sites)
    absolute={name:torch.full_like(x,4 if name.startswith("attn") else 7) for name in sites}
    changed=c.execute_with_writes(sites,absolute,lambda:mlp(attn(x)[0]))
    assert torch.equal(changed,torch.full_like(x,7)); assert torch.equal(mlp(attn(x)[0]),torch.ones_like(x))

def test_prefix_hybrid_and_bad_contracts():
    off=torch.zeros(1,3,2); on=torch.ones_like(off)
    assert torch.equal(c.prefix_hybrid(off,on,(1,)),torch.tensor([[[1.,1.],[1.,1.],[0.,0.]]]))
    with pytest.raises(c.ModuleWriteError): c.prefix_hybrid(off,on,(3,))
    with pytest.raises(c.ModuleWriteError): c.hybrid_bank({"a":off},{"a":on},(),(1,))
    with pytest.raises(c.ModuleWriteError): c.execute_with_writes({"a":(torch.nn.Identity(),"mlp")},{},lambda:off)
