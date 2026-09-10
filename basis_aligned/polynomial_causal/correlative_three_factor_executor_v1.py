"""Live QK1/QK2/value interchange through a fixed head-output partition.

Score factors retain native normalization/position semantics. Stored QK labels
are exchangeable within each head; this does not identify canonical semantics.
"""
import torch
import circuit_unit_greedy as g


def forward(backend,batch,artifact,donor=None,mask=7,branch='P'):
    if mask not in range(8) or branch not in ('P','R'):raise ValueError((mask,branch))
    blocks=g.blocks_of(artifact['units']);factors={};cache={};errors={};originals={}
    device=backend.device;idx=torch.arange(len(batch.row_ids),device=device)
    pos=torch.tensor(batch.semantic_positions,device=device)
    def wrapper(layer,units,original):
        qb=artifact['q'][(layer,'heads')].double().to(device)
        def wrapped(q,k,v,q2,k2):
            y=original(q,k,v,q2,k2);aa=[];bb=[];vv=[];native=[]
            visible=torch.arange(q.shape[1],device=device)[None,:]<=pos[:,None]
            for unit in units:
                h=int(unit.rsplit(':',1)[1])
                a=torch.einsum('bd,btd->bt',q[idx,pos,h].float(),k[:,:,h].float())/128
                b=torch.einsum('bd,btd->bt',q2[idx,pos,h].float(),k2[:,:,h].float())/128
                aa.append(a.double().masked_fill(~visible,0));bb.append(b.double());vv.append(v[:,:,h].double())
                native.append(y[idx,h,pos].double())
                for i,rid in enumerate(batch.row_ids):cache[(rid,unit)]=y[i,h,pos[i]].detach().clone()
            a=torch.stack(aa,1);b=torch.stack(bb,1);value=torch.stack(vv,1)
            live=torch.cat(native,dim=-1)
            reconstruct=torch.einsum('bht,bhtd->bhd',a*b,value).flatten(1)
            errors[layer]={'relative_l2':float((reconstruct-live).norm()/live.norm().clamp_min(1e-30)),
                           'max_scaled':float((reconstruct-live).abs().max()/(1e-3+1e-5*live.abs().max()))}
            factors[layer]={'qk1':a,'qk2':b,'value':value}
            if donor is None or mask==0:return y
            da,db,dv=(donor[layer][key].to(device) for key in ('qk1','qk2','value'))
            assert da.shape==a.shape and db.shape==b.shape and dv.shape==value.shape
            chosen_a=da if mask&1 else a;chosen_b=db if mask&2 else b;chosen_v=dv if mask&4 else value
            mixed=torch.einsum('bht,bhtd->bhd',chosen_a*chosen_b,chosen_v).flatten(1)
            change=mixed-live;projected=(change@qb)@qb.T
            if branch=='R':projected=change-projected
            changed=y.clone()
            for j,unit in enumerate(units):
                h=int(unit.rsplit(':',1)[1]);changed[idx,h,pos]+=projected[:,128*j:128*(j+1)].to(changed)
            return changed
        return wrapped
    try:
        for (layer,kind),units in blocks.items():
            assert kind=='heads';attn=backend.model.transformer.h[layer].attn
            originals[layer]=attn.squared_attention;attn.squared_attention=wrapper(layer,units,originals[layer])
        af,z=g.forward_units(backend,batch,return_logits=True)
        return af,z,{'factors':factors,'cache':cache,'errors':errors}
    finally:
        for layer,original in originals.items():backend.model.transformer.h[layer].attn.squared_attention=original
