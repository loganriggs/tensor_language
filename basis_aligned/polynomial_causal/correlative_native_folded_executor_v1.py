"""Conditional scalar producer/writer executor with complete native background."""
import torch
import circuit_unit_greedy as g
import correlative_route_read_write_v1 as C


def forward(backend,batch,blocks,donor_scalars=None):
    inputs={};scalars={};cache={};errors={};handles=[];originals={}
    device=backend.device;idx=torch.arange(len(batch.row_ids),device=device)
    positions=torch.tensor(batch.semantic_positions,device=device)
    def input_hook(layer):
        def hook(_module,args):inputs[layer]=args[0]
        return hook
    def route_reader(layer,original):
        def wrapped(q,k,v,q2,k2):
            y=original(q,k,v,q2,k2)
            block=blocks[layer];patterns={};native=torch.zeros(len(idx),device=device,dtype=torch.float64)
            source=torch.arange(q.shape[1],device=device)[None,:]
            for port in block['ports']:
                h=port['head']
                first=torch.einsum('bd,btd->bt',q[idx,positions,h,:].float(),k[:,:,h,:].float())/128
                second=torch.einsum('bd,btd->bt',q2[idx,positions,h,:].float(),k2[:,:,h,:].float())/128
                patterns[h]=((first*second).masked_fill(source>positions[:,None],0)).double()
                z=y[idx,h,positions,:]
                native=native+z.double()@port['head_reader'].double().to(device)
                for i,rid in enumerate(batch.row_ids):cache[(rid,port['unit'])]=z[i].detach().clone()
            folded=C.scalar(block,patterns,inputs[layer].double(),inputs[0].double())
            delta=(folded-native).abs()
            errors[layer]={'maxabs':float(delta.max()),
                           'scaled_error':float((delta/(1e-4+1e-5*native.abs())).max())}
            scalars[layer]=folded
            return y
        return wrapped
    def output_hook(layer):
        def hook(_module,args,out):
            if donor_scalars is None:return out
            changed=out.clone()
            delta=C.write_delta(blocks[layer],donor_scalars[layer].to(device),scalars[layer])
            changed[idx,positions,:]=changed[idx,positions,:]+delta.to(changed)
            return changed
        return hook
    try:
        for layer in sorted(set(blocks)|{0}):
            attn=backend.model.transformer.h[layer].attn
            handles.append(attn.register_forward_pre_hook(input_hook(layer)))
            if layer in blocks:
                originals[layer]=attn.squared_attention
                attn.squared_attention=route_reader(layer,originals[layer])
                handles.append(attn.c_proj.register_forward_hook(output_hook(layer)))
        af,logits=g.forward_units(backend,batch,return_logits=True)
        return af,logits,{'scalars':scalars,'cache':cache,'errors':errors}
    finally:
        for handle in handles:handle.remove()
        for layer,original in originals.items():backend.model.transformer.h[layer].attn.squared_attention=original
