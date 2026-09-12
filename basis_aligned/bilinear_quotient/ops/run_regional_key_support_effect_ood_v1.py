#!/usr/bin/env python3
# BQGATE:10 body forwards;48 sequences lengths6-14;5 suffix arms;no fit;180sec.
"""pred_a native producer and saved first-branch write replay<=1e-5;
pred_b forward13-key branch write relative error<=.1 in each geographic family;
pred_c forward13-key signed effect error<=.1 eachfamily,6/6transfer signs match and unrelated<=.5regional.
Null: the frozen13-update support fails downstream first-value write/effect fidelity.
Backward13 is diagnostic and all-update keys are the replay control; native queries fixed.
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
STEM='REGIONAL_KEY_SUPPORT_EFFECT_OOD_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json').read_text())['rows'];validate(rows);buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==48 and len(buckets)==8 and max(buckets)==14
    batches=[(length,ids[off:off+8]) for length,ids in sorted(buckets.items()) for off in range(0,len(ids),8)];assert len(batches)==10
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    for name in ('FACTORIAL_SOURCE_PORTS_V1_CONTROL.json','COMMON_QUADRATIC_PORTS_V1_CONTROL.json'):
        c=json.loads((P/name).read_text());assert max(v for v in c.values() if isinstance(v,(int,float)))<=1e-10
    assert json.loads((P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json').read_text())['relative_replay_error']<=1e-12
    assert json.loads((P/'RESIDUAL_PAYLOAD_UNROLL_V1_CONTROL.json').read_text())['relative_replay_error']<=1e-12
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('10 body forwards48sequences6-14tokens;5suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].cuda()
    pre=torch.empty(48,1152,device='cuda');writes=torch.zeros(48,5,1152,dtype=torch.float64,device='cuda');replays=[];count=0
    folds=torch.load(P/'REGIONAL_PRODUCER_OV_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    C=torch.load(P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['current_readers'].cuda()
    supports=json.loads((P/'REGIONAL_KEY_SUPPORT_V1_FROZEN.json').read_text())['supports'];supports['all']=list(range(26))
    records=torch.load(P/'REGIONAL_KEY_PREFIX_V3_ARTIFACT.pt',weights_only=True,map_location='cpu');support_states={}
    for rec in records:
        for name,support in supports.items():
            keep=[k for k in support if k<len(rec['parts'])]
            raw=rec['anchor'].double()+rec['parts'][keep].double().sum(0)
            support_states[(tuple(rec['prefix']),rec['layer'],name)]=F.rms_norm(raw.float(),(1152,)).cuda()
    producer_errors=[]
    handles=[model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))]
    def producer_hook(layer,module,args,out):
        inp=args[0];n,length,_=inp.shape
        cos,sin=module.rotary(inp.reshape(n,length,9,128))
        def qk(x):
            return [apply_rotary_emb(F.rms_norm(getattr(module,name)(x).reshape(n,length,9,128),(128,)),cos,sin).double() for name in ('c_q','c_k','c_q2','c_k2')]
        nq,nk,nq2,nk2=qk(inp)
        mask=torch.ones(length,length,dtype=torch.bool,device='cuda').tril()[None,None]
        def route(q,k,q2,k2):return (torch.einsum('nthd,nshd->nhts',q,k)/128)*(torch.einsum('nthd,nshd->nhts',q2,k2)/128)*mask
        routes=[route(nq,nk,nq2,nk2)]
        for name in ('forward','backward','all'):
            approx=inp.clone()
            for i,prefix in enumerate(captured['cue_prefixes']):approx[i,captured['cue_positions'][i]]=support_states[(prefix,layer,name)]
            _,ak,_,ak2=qk(approx);routes.append(route(nq,ak,nq2,ak2))
        fold=folds[str(layer)];E=fold['current_value_readers'].cuda();H=fold['first_value_readers'].cuda()
        current_values=torch.einsum('nsd,ahd->nsha',inp.double(),E);first_values=torch.einsum('nsd,ahd->nsha',captured['token_first'].double(),H)
        donor=torch.arange(n,device='cuda')^1;cue=torch.tensor(captured['cue_positions'],device='cuda');delta=first_values[donor,cue]-first_values[torch.arange(n,device='cuda'),cue]
        variants=torch.stack([torch.einsum('nht,nha->nta',torch.stack([routing[i,:,:,c] for i,c in enumerate(captured['cue_positions'])]),delta) for routing in routes],1)
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
            changes=[[j for j,(a,b) in enumerate(zip(rows[i]['ids'],rows[i^1]['ids'])) if a!=b] for i in ids];assert all(len(c)==1 for c in changes);captured['cue_positions']=[c[0] for c in changes]
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            captured['token_embeddings']=x0;captured['cue_prefixes']=[tuple(rows[i]['ids'][:c+1]) for i,c in zip(ids,captured['cue_positions'])]
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
    assert count==10
    def margins(delta):
        z=pre+delta.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    arms=torch.stack([margins(writes[:,j]-writes[:,0]) for j in range(5)])
    prior=torch.load(P/'REGIONAL_PRODUCER_VALUE_STREAM_OOD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['write_vertices'].cuda()
    native_delta=writes[:,1]-writes[:,0];reference_delta=prior[:,2]-prior[:,0]
    replay=float((native_delta-reference_delta).norm()/reference_delta.norm())
    effects=arms-arms[0,None];cells=[]
    for arm,name in enumerate(('forward13_keys','backward13_keys','all_update_keys'),2):
        for family in range(4):
            ids=[i for i,r in enumerate(rows) if r['family']==family];uk=[i for i in ids if rows[i]['cue']=='British'];us=[i+1 for i in uk]
            ref=effects[1,ids,0];effect=effects[arm,ids,0]
            transfer=(effects[arm,us,0]-effects[arm,uk,0])/2;native_transfer=(effects[1,us,0]-effects[1,uk,0])/2
            cells.append(dict(name=name,family=family,write_relative_error=float((writes[ids,arm]-writes[ids,0]-native_delta[ids]).norm()/native_delta[ids].norm()),regional_effect_relative_error=float((effect-ref).norm()/ref.norm()),native_transfer_mean=float(native_transfer.mean()),transfer_mean=float(transfer.mean()),transfer_sign_matches=int((transfer*native_transfer>0).sum()),unrelated_meanabs=float(effects[arm,ids,1].abs().mean()),regional_meanabs=float(effect.abs().mean())))
    instrument=max(producer_errors+replays+[replay]+[c['write_relative_error'] for c in cells if c['name']=='all_update_keys'])<=1e-5
    candidate=[c for c in cells if c['name']=='forward13_keys'];faithful=instrument and all(c['write_relative_error']<=.1 for c in candidate)
    result={'pred_a':instrument,'pred_b':faithful,'pred_c':faithful and all(c['regional_effect_relative_error']<=.1 and c['transfer_sign_matches']==6 and c['unrelated_meanabs']<=.5*c['regional_meanabs'] for c in candidate)}
    torch.save(dict(pre=pre.cpu(),write_vertices=writes.cpu(),arm_margins=arms.cpu()),art)
    result.update(native_branch_replay_error=replay,producer_replay_errors=producer_errors,native_attention_replay_errors=replays,cells=cells,body_forwards=count,frozen_supports=supports,seconds=time.perf_counter()-tic,artifact_sha=digest(art),source_shas=binding,scope='Frozen shared13 key support propagated through first-value branch with native queries and remaining background. Geographic48 rows/8key prefixes excluded from original support selection. BothQK factorsjoint; native prefix generators/queries/background retained; conditional template/geographic validation, not new corpus or independent extraction.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
