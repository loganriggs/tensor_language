import torch

from squared_attention_head_tools import projected_head_writes


def test_head_writes_sum_to_flat_projection_for_multiple_batch_sizes():
    for batch,tokens,heads,width,out in ((1,5,3,4,7),(3,6,2,5,9)):
        torch.manual_seed(batch+tokens)
        factors=[torch.randn(batch,tokens,heads,width) for _ in range(5)]
        weight=torch.randn(out,heads*width)
        writes=projected_head_writes(*factors,weight)
        q,k,q2,k2,value=factors
        score1=torch.einsum("bqhd,bkhd->bhqk",q,k)/width
        score2=torch.einsum("bqhd,bkhd->bhqk",q2,k2)/width
        mask=torch.ones(tokens,tokens,dtype=torch.bool).tril()
        z=torch.einsum("bhqk,bkhd->bhqd",(score1*score2).masked_fill(~mask,0),value)
        reference=torch.nn.functional.linear(z.transpose(1,2).contiguous().reshape(batch,tokens,heads*width),weight)
        assert writes.shape==(batch,heads,tokens,out)
        assert torch.allclose(writes.sum(1),reference,atol=2e-5,rtol=2e-5)


def test_bfloat16_projection_rounding_is_explicitly_accounted_for():
    torch.manual_seed(19)
    batch,tokens,heads,width,out=2,7,3,8,11
    factors=[torch.randn(batch,tokens,heads,width,dtype=torch.bfloat16) for _ in range(5)]
    weight=torch.randn(out,heads*width,dtype=torch.bfloat16)
    writes=projected_head_writes(*factors,weight).double()
    q,k,q2,k2,value=factors
    score1=torch.einsum("bqhd,bkhd->bhqk",q,k)/width
    score2=torch.einsum("bqhd,bkhd->bhqk",q2,k2)/width
    mask=torch.ones(tokens,tokens,dtype=torch.bool).tril()
    z=torch.einsum("bhqk,bkhd->bhqd",(score1*score2).masked_fill(~mask,0),value)
    reference=torch.nn.functional.linear(
        z.transpose(1,2).contiguous().reshape(batch,tokens,heads*width),weight
    ).double()
    rounding=reference-writes.sum(1)
    assert torch.equal(writes.sum(1)+rounding,reference)
    assert float(rounding.norm()/reference.norm()) <= .005
