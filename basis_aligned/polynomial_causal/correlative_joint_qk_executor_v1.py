"""Joint-product feature interventions with live native attention semantics."""
import torch
import circuit_unit_greedy as g

def feature(a,b):return (a.double()[..., :,None]*b.double()[...,None,:]).flatten(-2)
def basis_of(delta):
    flat=delta.reshape(-1,delta.shape[-1]);_,s,vh=torch.linalg.svd(flat,full_matrices=False)
    rank=int((s>s[0]*1e-6).sum()) if len(s) and s[0]>1e-12 else 0
    basis=vh[:rank].T.contiguous();res=flat-(flat@basis)@basis.T
    return basis,dict(rank=rank,orthogonality=float((basis.T@basis-torch.eye(rank,device=flat.device)).abs().max()) if rank else 0.,fit_relative=float(res.norm()/flat.norm().clamp_min(1e-30)))
def build_spaces(base,donor):
    spaces={};diagnostics={}
    for key,c in base['factors'].items():
        d=donor['factors'][key];spaces[key]={};diagnostics[key]={}
        for kind,a,b in [('query','q1','q2'),('key','k1','k2')]:
            delta=feature(d[a],d[b])-feature(c[a],c[b]);spaces[key][kind],diagnostics[key][kind]=basis_of(delta)
    return spaces,diagnostics

def forward(backend,batch,artifact,donor=None,space=None,mode='native',branch='P'):
    assert mode in ['native','value','full','space'];device=backend.device
    blocks=g.blocks_of(artifact['units']);idx=torch.arange(len(batch.row_ids),device=device);pos=torch.tensor(batch.semantic_positions,device=device)
    originals={};factors={};cache={};errors={};coordinate_error=0.
    def wrapper(layer,units,original):
        qb=artifact['q'][(layer,'heads')].double().to(device)
        def wrapped(q,k,v,q2,k2):
            nonlocal coordinate_error
            y=original(q,k,v,q2,k2);live=[];mixed=[]
            visible=torch.arange(q.shape[1],device=device)[None,:]<=pos[:,None]
            for unit in units:
                h=int(unit.rsplit(':',1)[1]);c=dict(q1=q[idx,pos,h].detach().clone(),q2=q2[idx,pos,h].detach().clone(),k1=k[:,:,h].detach().clone(),k2=k2[:,:,h].detach().clone(),value=v[:,:,h].detach().clone());factors[unit]=c
                fq=feature(c['q1'],c['q2']);fk=feature(c['k1'],c['k2']);native=y[idx,h,pos].double();live.append(native)
                score=torch.einsum('bd,btd->bt',fq,fk).masked_fill(~visible,0)/128**2
                reconstructed=torch.einsum('bt,btd->bd',score,c['value'].double());err=reconstructed-native
                errors[unit]=dict(relative_l2=float(err.norm()/native.norm().clamp_min(1e-30)),max_scaled=float(err.abs().max()/(1e-3+1e-5*native.abs().max())))
                for i,rid in enumerate(batch.row_ids):cache[(rid,unit)]=y[i,h,pos[i]].detach().clone()
                if mode=='native':continue
                d=donor['factors'][unit]
                if mode in ['full','space']:
                    dq=feature(d['q1'],d['q2'])-fq;dk=feature(d['k1'],d['k2'])-fk
                    if mode=='space':
                        bq=space[unit]['query'];bk=space[unit]['key'];dq=(dq@bq)@bq.T;dk=(dk@bk)@bk.T
                        if bq.shape[1]:
                            coeff=torch.einsum('bi,ijr,bj->br',c['q1'].double(),bq.reshape(128,128,-1),c['q2'].double())
                            coordinate_error=max(coordinate_error,float((coeff-fq@bq).norm()/(fq@bq).norm().clamp_min(1e-30)))
                    fq=fq+dq;fk=fk+dk
                score=torch.einsum('bd,btd->bt',fq,fk).masked_fill(~visible,0)/128**2
                mixed.append(torch.einsum('bt,btd->bd',score,d['value'].double()))
            if mode=='native':return y
            live=torch.cat(live,-1);change=torch.cat(mixed,-1)-live;projected=(change@qb)@qb.T
            if branch=='R':projected=change-projected
            changed=y.clone()
            for j,unit in enumerate(units):changed[idx,int(unit.rsplit(':',1)[1]),pos]+=projected[:,128*j:128*(j+1)].to(changed)
            return changed
        return wrapped
    try:
        for (layer,kind),units in blocks.items():
            assert kind=='heads';attn=backend.model.transformer.h[layer].attn;originals[layer]=attn.squared_attention;attn.squared_attention=wrapper(layer,units,originals[layer])
        af,z=g.forward_units(backend,batch,return_logits=True)
        return af,z,dict(factors=factors,cache=cache,errors=errors,coordinate_error=coordinate_error)
    finally:
        for layer,original in originals.items():backend.model.transformer.h[layer].attn.squared_attention=original
