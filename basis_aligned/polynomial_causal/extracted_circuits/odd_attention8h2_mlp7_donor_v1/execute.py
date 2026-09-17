"""Conditional head8 write with donor state generated through native MLP7."""
import torch
import donor
import native

def execute(p, current, donor_g7, recipient_token, donor_token, city, destination):
    ids=p['donor']['token_ids'].tolist()
    if donor_token not in ids:
        raise ValueError('Unknown donor token')
    initial=p['donor']['initial_table'][ids.index(donor_token)].unsqueeze(0)
    state=donor.generate(p['donor'],donor_g7,initial)
    return native.execute(p['head'],current,state,recipient_token,donor_token,city,destination)
