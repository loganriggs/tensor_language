#!/usr/bin/env python3
# BQGATE:6 body forwards;32 sequences lengths7-11;5 suffix arms;no fit;180sec.
"""pred_a native producer and saved first-branch write replay<=1e-5;
pred_b pure-token routing branch write relative error<=.1;
pred_c pure-token regional signed effect error<=.1 plus preserved controls.
Null: contextual routing input cannot be replaced by normalized token embeddings alone.
Two query/key hybrids are diagnostic, with both QK factors always moved together.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from folded_normalized_router_v1 import rotary
from shared_cubic_source_projection_v1 import cross_factors
from common_quadratic_ports_v1 import source_ports,private_writers
from factorial_source_ports_v1 import write
from jacclust.tt_model import apply_rotary_emb
from regional_cue_row_check_v1 import validate
STEM='REGIONAL_TOKEN_ROUTING_V2'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'REGIONAL_BEHAVIOR_CONTROLS_V2_ROWS.json').read_text())['rows'];validate(rows);buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==32 and len(buckets)==5 and max(buckets)==11
    batches=[(length,ids[off:off+8]) for length,ids in sorted(buckets.items()) for off in range(0,len(ids),8)];assert len(batches)==6
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    for name in ('FACTORIAL_SOURCE_PORTS_V1_CONTROL.json','COMMON_QUADRATIC_PORTS_V1_CONTROL.json'):
        c=json.loads((P/name).read_text());assert max(v for v in c.values() if isinstance(v,(int,float)))<=1e-10
    assert json.loads((P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json').read_text())['relative_replay_error']<=1e-12
    assert json.loads((P/'RESIDUAL_PAYLOAD_UNROLL_V1_CONTROL.json').read_text())['relative_replay_error']<=1e-12
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('6 body forwards32sequences7-11tokens;5suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].cuda()
    pre=torch.empty(32,1152,device='cuda');writes=torch.zeros(32,5,1152,dtype=torch.float64,device='cuda');replays=[];count=0
    folds=torch.load(P/'REGIONAL_PRODUCER_OV_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    C=torch.load(P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['current_readers'].cuda()
    producer_errors=[]
    handles=[model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))]
    def producer_hook(layer,module,args,out):
        inp=args[0];n,length,_=inp.shape
        cos,sin=module.rotary(inp.reshape(n,length,9,128))
        def qk(x):
            return [apply_rotary_emb(F.rms_norm(getattr(module,name)(x).reshape(n,length,9,128),(128,)),cos,sin).double() for name in ('c_q','c_k','c_q2','c_k2')]
        nq,nk,nq2,nk2=qk(inp);tq,tk,tq2,tk2=qk(captured['token_first'])
        mask=torch.ones(length,length,dtype=torch.bool,device='cuda').tril()[None,None]
        def route(q,k,q2,k2):return (torch.einsum('nthd,nshd->nhts',q,k)/128)*(torch.einsum('nthd,nshd->nhts',q2,k2)/128)*mask
        routes=[route(nq,nk,nq2,nk2),route(tq,tk,tq2,tk2),route(nq,tk,nq2,tk2),route(tq,nk,tq2,nk2)]
        fold=folds[str(layer)];E=fold['current_value_readers'].cuda();H=fold['first_value_readers'].cuda()
        current_values=torch.einsum('nsd,ahd->nsha',inp.double(),E);first_values=torch.einsum('nsd,ahd->nsha',captured['token_first'].double(),H)
        donor=torch.arange(n,device='cuda')^1;delta=first_values[donor,1]-first_values[:,1]
        variants=torch.stack([torch.einsum('nht,nha->nta',routing[:,:,:,1],delta) for routing in routes],1)
        replay=torch.einsum('nhts,nsha->nta',routes[0],current_values+first_values)
        reference=float(fold['residual_scale'])*(out[0].double()@C.T)
        producer_errors.append(float((replay-reference).norm()/reference.norm()))
        captured['producer_variants'].append(variants)

    for layer in (8,9,13):
        handles.append(model.transformer.h[layer].attn.register_forward_hook(lambda module,args,out,layer=layer:producer_hook(layer,module,args,out)))
    try:
        for length,ids in batches:
            assert len(ids)%2==0 and all(ids[j]^1==ids[j+1] for j in range(0,len(ids),2));donor_indices=torch.arange(len(ids),device='cuda')^1
            captured['producer_variants']=[]
            assert all(rows[i]['ids'][1]!=rows[i^1]['ids'][1] and rows[i]['ids'][:1]+rows[i]['ids'][2:]==rows[i^1]['ids'][:1]+rows[i^1]['ids'][2:] for i in ids)
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            initial=model.transformer.h[0].lambdas;captured['token_first']=F.rms_norm(initial[0]*x0+initial[1]*x0,(1152,))
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];count+=1
            rho=(r.double().square().mean(-1,keepdim=True)+eps).sqrt();producer_variants=sum(captured['producer_variants'])
            current=current.double();first=captured['first'].double();query=current[:,-1];qr=rotary(length-1,128).cuda();qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2);native_sum=torch.zeros_like(query)
            for pos in range(length):
                source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
                gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt());rotation=qr.T@rotary(pos,128).cuda()
                score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb);vv=torch.einsum('nd,hkd->nhk',source,value);native_sum+=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output)
                kr1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kr2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1);weights=(q1,kr1,q2,kr2,value,output*g0[None,:,None])
                parent,children=source_ports(source,atoms);writers=private_writers(query,atoms,cross_factors(atoms,*weights))
                variants=[children]+[children+producer_variants[:,mask,pos]/rho[:,pos] for mask in range(4)]
                writes[ids]+=torch.stack([write(gate/g0[None],parent,u,writers) for u in variants],1)
            replays.append(float((native_sum-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:
        for handle in handles:handle.remove()
    assert count==6
    def margins(delta):
        z=pre+delta.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    arms=torch.stack([margins(writes[:,j]-writes[:,0]) for j in range(5)])
    prior=torch.load(P/'REGIONAL_BEHAVIOR_CONTROLS_V2_ARTIFACT.pt',weights_only=True,map_location='cpu');reference=prior['write_vertices'].cuda();native_delta=writes[:,1]-writes[:,0]
    replay=float((native_delta-(reference[:,1]-reference[:,0])).norm()/(reference[:,1]-reference[:,0]).norm())
    effects=arms[1:,:,0]-arms[0,None,:,0];reg=[i for i,row in enumerate(rows) if row['family']==0];regref=effects[0,reg];cells=[]
    for arm,name in enumerate(('token_queries_token_keys','native_queries_token_keys','token_queries_native_keys'),2):
        effect=effects[arm-1];write_error=float((writes[:,arm]-writes[:,0]-native_delta).norm()/native_delta.norm());effect_error=float((effect[reg]-regref).norm()/regref.norm());control_cells=[]
        for family in (1,2,3):
            ids=[i for i,row in enumerate(rows) if row['family']==family];base=arms[0,ids,0];edited=arms[arm,ids,0]
            control_cells.append(dict(family=family,native_margin_rms=float(base.square().mean().sqrt()),effect_rms=float(effect[ids].square().mean().sqrt()),effect_meanabs=float(effect[ids].abs().mean()),capable_flips=int(((base>0)&(edited<=0)).sum())))
        cells.append(dict(name=name,write_relative_error=write_error,regional_effect_relative_error=effect_error,control_cells=control_cells))
    instrument=max(producer_errors+replays+[replay])<=1e-5;candidate=cells[0];regscale=float(regref.square().mean().sqrt())
    result={'pred_a':instrument,'pred_b':instrument and candidate['write_relative_error']<=.1,'pred_c':instrument and candidate['write_relative_error']<=.1 and candidate['regional_effect_relative_error']<=.1 and all(c['effect_rms']<=.05*c['native_margin_rms'] and c['effect_meanabs']<=.25*regscale and c['capable_flips']==0 for c in candidate['control_cells'])}
    torch.save(dict(pre=pre.cpu(),write_vertices=writes.cpu(),arm_margins=arms.cpu()),art)
    result.update(native_branch_replay_error=replay,producer_replay_errors=producer_errors,native_attention_replay_errors=replays,cells=cells,body_forwards=count,seconds=time.perf_counter()-tic,artifact_sha=digest(art),source_shas=binding,scope='Frozen embedding-only QK routing baseline and query/key hybrids. BothQK factorsjoint; projectedfirstvalues andremainingdownstreamcontextfixed. No datafit. Failure rejects this rawtoken-routing approximation, not all context-independent or weight-based circuits.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
