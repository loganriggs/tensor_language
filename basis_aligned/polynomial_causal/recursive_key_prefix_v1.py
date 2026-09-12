"""Run selected updates recursively; omitted writes use a fixed zero trajectory.

No saved native prefix states are consumed. Shared first-layer values are still
generated from the actual token input when attention0 itself is omitted.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model,prefixes,supports,layers=(8,9,13)):
    last=max(layers);device=model.transformer.wte.weight.device
    zero_updates={};zero_calls=0
    for length in sorted(set(map(len,prefixes))):
        x0=torch.zeros(1,length,1152,device=device);x=x0;v=None;updates=[]
        for j,block in enumerate(model.transformer.h[:last]):
            x=block.lambdas[0]*x+block.lambdas[1]*x0
            attention,v=block.attn(F.rms_norm(x,(1152,)),v);x=x+attention
            mlp=block.mlp(F.rms_norm(x,(1152,)));x=x+mlp
            updates.append((attention,mlp));zero_calls+=2
        zero_updates[length]=updates
    states={};counts={name:dict(attention=0,mlp=0,first_value_only=0) for name in supports}
    for prefix in prefixes:
        x0=F.rms_norm(model.transformer.wte(torch.tensor([prefix],device=device)),(1152,))
        for name,support in supports.items():
            support=set(support);x=x0;v=None
            for j,block in enumerate(model.transformer.h[:last+1]):
                x=block.lambdas[0]*x+block.lambdas[1]*x0
                if j in layers:states[(tuple(prefix),j,name)]=F.rms_norm(x,(1152,))[0,-1].clone()
                if j==last:break
                if 2*j in support:
                    attention,v=block.attn(F.rms_norm(x,(1152,)),v);counts[name]['attention']+=1
                else:
                    if j==0:
                        v=block.attn.c_v(F.rms_norm(x,(1152,))).reshape(1,len(prefix),9,128)
                        counts[name]['first_value_only']+=1
                    attention=zero_updates[len(prefix)][j][0]
                x=x+attention
                if 2*j+1 in support:
                    mlp=block.mlp(F.rms_norm(x,(1152,)));counts[name]['mlp']+=1
                else:mlp=zero_updates[len(prefix)][j][1]
                x=x+mlp
    for name,support in supports.items():
        assert counts[name]['attention']==len(prefixes)*sum(k%2==0 for k in support)
        assert counts[name]['mlp']==len(prefixes)*sum(k%2==1 for k in support)
    price=dict(prefixes=len(prefixes),zero_update_calls=zero_calls,live_calls=counts,
               cached_zero_scalars=sum(v.numel() for updates in zero_updates.values() for pair in updates for v in pair),
               key_state_scalars=sum(v.numel() for v in states.values()),
               scope='Cache-building uses native zero-input weights. Counts exclude embeddings, RMS, residual reentry, key projections, native queries and downstream background; this is not total program pricing.')
    return states,price
