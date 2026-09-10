"""Fold a saved block-live head interface; native input/router dependencies remain."""
import torch


def fold(artifact, state):
    blocks={}
    for key,q in artifact['q'].items():
        layer,kind=key
        if kind!='heads' or q.shape[1]!=1:raise ValueError('Only saved rank-one head blocks supported')
        units=[u for u in artifact['units'] if int(u.split(':')[1])==layer]
        if len(units)*128!=q.shape[0]:raise ValueError('Saved unit concatenation mismatch')
        prefix=f'transformer.h.{layer}.attn.'
        local_v=state[prefix+'c_v.weight'].double()
        first_v=state['transformer.h.0.attn.c_v.weight'].double()
        output=state[prefix+'c_proj.weight'].double()
        lam=float(state[prefix+'lamb'])
        full_q=torch.zeros(1152,dtype=torch.float64);ports=[]
        for index,unit in enumerate(units):
            h=int(unit.rsplit(':',1)[1]);reader=q[index*128:(index+1)*128,0].double()
            span=slice(h*128,(h+1)*128);full_q[span]=reader
            ports.append({'unit':unit,'head':h,'head_reader':reader,
                          'local_value_reader':(1-lam)*(reader@local_v[span]),
                          'first_value_reader':lam*(reader@first_v[span])})
        blocks[layer]={'units':units,'ports':ports,'writer':output@full_q,
                       'full_head_reader':full_q,'value_mix_lambda':lam}
    return blocks


def scalar(block,patterns,local_inputs,first_inputs):
    """patterns maps native head ID to [batch,source] routing at one query.

    Inputs [batch,source,1152] are the actual normalized local and first-layer
    attention inputs. The caller retains source mask and complete normalized
    two-QK routing. This is a conditional read port, not an upstream extractor.
    """
    result=torch.zeros(local_inputs.shape[0],dtype=local_inputs.dtype,device=local_inputs.device)
    for port in block['ports']:
        payload=local_inputs@port['local_value_reader'].to(local_inputs)
        payload=payload+first_inputs@port['first_value_reader'].to(first_inputs)
        result=result+(patterns[port['head']]*payload).sum(-1)
    return result


def write_delta(block,donor_scalar,live_scalar):
    return (donor_scalar-live_scalar)[...,None]*block['writer'].to(live_scalar)
