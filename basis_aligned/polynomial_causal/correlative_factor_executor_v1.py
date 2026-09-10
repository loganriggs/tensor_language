"""Expose routing and value operands of the fixed native block-scalar interface."""
import torch
import circuit_unit_greedy as g
from correlative_route_read_write_v1 import write_delta


def forward(backend,batch,blocks,donor_factors=None,arm='joint'):
    if arm not in ('joint','route','value'):raise ValueError(arm)
    inputs={};scalars={};factors={};errors={};terms={};handles=[];originals={}
    device=backend.device;idx=torch.arange(len(batch.row_ids),device=device)
    positions=torch.tensor(batch.semantic_positions,device=device)
    def input_hook(layer):
        def hook(_m,args):inputs[layer]=args[0]
        return hook
    def routed(layer,original):
        def wrapped(q,k,v,q2,k2):
            y=original(q,k,v,q2,k2);block=blocks[layer];ps=[];us=[]
            native=torch.zeros(len(idx),device=device,dtype=torch.float64)
            source=torch.arange(q.shape[1],device=device)[None,:]
            for port in block['ports']:
                h=port['head']
                a=torch.einsum('bd,btd->bt',q[idx,positions,h,:].float(),k[:,:,h,:].float())/128
                b=torch.einsum('bd,btd->bt',q2[idx,positions,h,:].float(),k2[:,:,h,:].float())/128
                ps.append(((a*b).masked_fill(source>positions[:,None],0)).double())
                u=inputs[layer].double()@port['local_value_reader'].to(device)
                u=u+inputs[0].double()@port['first_value_reader'].to(device)
                us.append(u)
                native+=y[idx,h,positions,:].double()@port['head_reader'].double().to(device)
            p=torch.stack(ps,1);u=torch.stack(us,1);s=(p*u).sum((1,2))
            factors[layer]={'route':p,'value':u};scalars[layer]=s
            errors[layer]=float(((s-native).abs()/(1e-4+1e-5*native.abs())).max())
            if donor_factors is not None:
                pd=donor_factors[layer]['route'].to(device);ud=donor_factors[layer]['value'].to(device)
                if pd.shape!=p.shape or ud.shape!=u.shape:raise ValueError('Unaligned factor operands')
                dr=((pd-p)*u).sum((1,2));dv=(p*(ud-u)).sum((1,2));cross=((pd-p)*(ud-u)).sum((1,2))
                terms[layer]={'route':dr,'value':dv,'cross':cross,'joint':(pd*ud).sum((1,2))-s}
            return y
        return wrapped
    def output_hook(layer):
        def hook(_m,args,out):
            if donor_factors is None:return out
            change=terms[layer][arm]
            # These are operand swaps at this block's CURRENT recipient state.
            changed=out.clone();changed[idx,positions,:]+=write_delta(blocks[layer],scalars[layer]+change,scalars[layer]).to(changed)
            return changed
        return hook
    try:
        for layer in sorted(set(blocks)|{0}):
            attn=backend.model.transformer.h[layer].attn;handles.append(attn.register_forward_pre_hook(input_hook(layer)))
            if layer in blocks:
                originals[layer]=attn.squared_attention;attn.squared_attention=routed(layer,originals[layer])
                handles.append(attn.c_proj.register_forward_hook(output_hook(layer)))
        af,z=g.forward_units(backend,batch,return_logits=True)
        return af,z,{'scalars':scalars,'factors':factors,'errors':errors,'terms':terms}
    finally:
        for h in handles:h.remove()
        for layer,original in originals.items():backend.model.transformer.h[layer].attn.squared_attention=original
