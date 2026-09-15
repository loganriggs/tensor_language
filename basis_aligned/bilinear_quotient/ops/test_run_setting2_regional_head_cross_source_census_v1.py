import torch


def test_source_axis_sums_to_cross_term_without_broadcasting():
    torch.manual_seed(150907)
    batch,query,key,width=3,7,7,11
    removed=torch.randn(batch,query,key)
    delta_value=torch.randn(batch,key,width)
    per_source=-removed[:,-1,:,None]*delta_value
    direct=-torch.einsum("bk,bkd->bd",removed[:,-1],delta_value)
    assert per_source.shape==(batch,key,width)
    assert torch.allclose(per_source.sum(1),direct,atol=1e-6,rtol=1e-6)


def test_variable_length_mask_indexes_only_unpadded_tokens():
    padded=torch.randn(21,128)
    mask=torch.tensor([True,False]*10)
    selected=padded[:mask.numel()][mask]
    assert selected.shape==(10,128)
