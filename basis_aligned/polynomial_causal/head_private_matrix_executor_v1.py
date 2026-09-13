"""Execute shared and head-private factors without materializing their sum."""
import torch

def execute(head_writes,factors):
    shared=(head_writes@factors['shared_right'].T)@factors['shared_left'].T
    heads,_,width=factors['private_right'].shape
    inputs=head_writes.reshape(*head_writes.shape[:-1],heads,width)
    hidden=torch.einsum('...hi,hri->...hr',inputs,factors['private_right'])
    return shared+torch.einsum('...hr,hor->...o',hidden,factors['private_left'])
