"""Align individual token fields before crossing contexts with changed answers."""
def align(fields,differentials,donor_tokens,recipient_tokens):
    import torch
    source=donor_tokens.flatten(1);target=recipient_tokens.flatten(1)
    matches=target[:,:,None]==source[:,None,:]
    if not bool((matches.sum(-1)==1).all()):raise ValueError('Token field mapping is not unique/complete')
    index=matches.long().argmax(-1)
    index=torch.cat([index,torch.full((len(index),1),fields.shape[-1]-1,device=index.device,dtype=index.dtype)],dim=1)
    return fields.gather(1,index),differentials.gather(1,index[:,:,None].expand(-1,-1,differentials.shape[-1]))
