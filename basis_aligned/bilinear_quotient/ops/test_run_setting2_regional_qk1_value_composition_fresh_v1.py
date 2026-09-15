import torch


def test_joint_routing_value_identity_and_shapes():
    torch.manual_seed(150906)
    batch,query,key,dim=3,7,7,11
    pattern=torch.randn(batch,query,key)
    removed=torch.randn(batch,query,key)
    value=torch.randn(batch,key,dim)
    delta=torch.randn(batch,key,dim)
    route=-torch.einsum("bqk,bkd->bqd",removed,value)
    value_effect=torch.einsum("bqk,bkd->bqd",pattern,delta)
    interaction=-torch.einsum("bqk,bkd->bqd",removed,delta)
    composed=route+value_effect+interaction
    direct=torch.einsum("bqk,bkd->bqd",pattern-removed,value+delta)-torch.einsum("bqk,bkd->bqd",pattern,value)
    assert composed.shape==(batch,query,dim)
    assert torch.allclose(composed,direct,atol=2e-6,rtol=2e-6)
