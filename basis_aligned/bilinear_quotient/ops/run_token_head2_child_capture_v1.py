#!/usr/bin/env python3
# BQGATE:9 body forwards;48 sequences lengths13-19;7 suffix arms;no fit;180sec.
"""pred_a native/prior andoldFP32write<=1e-5/effect<=1e-3.
pred_b tokenFP32write<=1e-5/effect<=1e-3 EACHfamily.
pred_c childreference<=1e-5 andchildpartition<=1e-6.
9bodybatches48rows13–19tokens7suffixarms180sec. No fit; closesfirststateonly.
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
from compiled_shared_head_v1 import execute_head
from compiled_token_shared_head_v1 import execute_token_head
STEM='TOKEN_HEAD2_CHILD_CAPTURE_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'REGIONAL_VALUE_FRESH_GEO_V1_ROWS.json').read_text())['rows'];validate(rows);buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==48 and len(buckets)==7 and max(buckets)==19
    batches=[(length,ids[off:off+8]) for length,ids in sorted(buckets.items()) for off in range(0,len(ids),8)];assert len(batches)==9
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    for name in ('FACTORIAL_SOURCE_PORTS_V1_CONTROL.json','COMMON_QUADRATIC_PORTS_V1_CONTROL.json'):
        c=json.loads((P/name).read_text());assert max(v for v in c.values() if isinstance(v,(int,float)))<=1e-10
    assert json.loads((P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json').read_text())['relative_replay_error']<=1e-12
    assert json.loads((P/'RESIDUAL_PAYLOAD_UNROLL_V1_CONTROL.json').read_text())['relative_replay_error']<=1e-12
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('9 body forwards48sequences13-19tokens;7suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].cuda()
    pre=torch.empty(48,1152,device='cuda');writes=torch.zeros(48,6,1152,dtype=torch.float64,device='cuda');replays=[];count=0
    program={k:v.cuda() for k,v in torch.load(P/'COMPILED_SHARED_HEAD2_V1_ARTIFACT.pt',weights_only=True,map_location='cpu').items()}
    program32={k:v.float() for k,v in program.items()}
    token_program={k:v.cuda() for k,v in torch.load(P/'COMPILED_TOKEN_SHARED_HEAD2_V1_ARTIFACT.pt',weights_only=True,map_location='cpu').items()}
    child_replays=[];worst=None
    handles=[model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))]
    try:
        for length,ids in batches:
            assert len(ids)%2==0 and all(ids[j]^1==ids[j+1] for j in range(0,len(ids),2));donor_indices=torch.arange(len(ids),device='cuda')^1
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];count+=1
            rho=(r.double().square().mean(-1,keepdim=True)+eps).sqrt()
            current=current.double();first=captured['first'].double();query=current[:,-1];qr=rotary(length-1,128).cuda();qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2);native_sum=torch.zeros_like(query)
            for pos in range(length):
                source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
                gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt());rotation=qr.T@rotary(pos,128).cuda()
                score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb);vv=torch.einsum('nd,hkd->nhk',source,value);native_sum+=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output)
                kr1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kr2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1);weights=(q1,kr1,q2,kr2,value,output*g0[None,:,None])
                parent,children=source_ports(source,atoms);writers=private_writers(query,atoms,cross_factors(atoms,*weights))
                mask=torch.zeros_like(gate);mask[:,2]=gate[:,2]
                ref=write(mask/g0[None],parent,children,writers)
                compiled=execute_head(query.float(),source.float(),rotation.float(),program32).double()
                compiled32=execute_token_head(query.float(),current[:,pos].float(),tokens[:,pos],rotation.float(),token_program).double()
                child_writes=[execute_token_head(query.float(),current[:,pos].float(),tokens[:,pos],rotation.float(),token_program,mask).double() for mask in ((1.,0.),(0.,1.))]
                for j in (0,1):
                    cc=children.clone();cc[:,1-j]=0;child_ref=write(mask/g0[None],parent,cc,writers)
                    error=float((child_writes[j]-child_ref).norm()/child_ref.norm().clamp_min(1e-30));child_replays.append(error)
                    if worst is None or error>worst['relative_error']:
                        old_child=dict(program32);old_child['children']=program32['children'].clone();old_child['children'][1-j]=0
                        old_write=execute_head(query.float(),source.float(),rotation.float(),old_child).double()
                        worst=dict(relative_error=error,child=j,position=pos,row_ids=ids,
                            reference_norm=float(child_ref.norm()),error_norm=float((child_writes[j]-child_ref).norm()),
                            old_fp32_relative_error=float((old_write-child_ref).norm()/child_ref.norm().clamp_min(1e-30)),
                            query=query.cpu(),source=source.cpu(),rotation=rotation.cpu(),token_ids=tokens[:,pos].cpu(),
                            reference=child_ref.cpu(),actual=child_writes[j].cpu(),old_fp32=old_write.cpu())
                writes[ids]+=torch.stack([write(gate/g0[None],parent,children,writers),ref,compiled,compiled32,*child_writes],1)
            replays.append(float((native_sum-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:
        for handle in handles:handle.remove()
    assert count==9
    def margins(delta):
        z=pre+delta.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    baseline=margins(torch.zeros_like(pre));arms=torch.stack([margins(-writes[:,j]) for j in range(6)]);effects=baseline[None]-arms
    prior=torch.load(P/'REGIONAL_PRIVATE_CONSUMERS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['write_vertices'][:,1].cuda()
    def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
    prior_replay=relative(writes[:,1],prior);cells=[]
    for arm,name in ((2,'old_fp32'),(3,'token_fp32')):
        for family in range(4):
            ids=[i for i,row in enumerate(rows) if row['family']==family]
            cells.append(dict(arm=name,family=family,write_relative_error=relative(writes[ids,arm],writes[ids,1]),effect_relative_error=relative(effects[arm,ids,0],effects[1,ids,0])))
    aa=max(replays+[prior_replay])<=1e-5 and all(c['write_relative_error']<=1e-5 and c['effect_relative_error']<=1e-3 for c in cells if c['arm']=='old_fp32')
    bb=aa and all(c['write_relative_error']<=1e-5 and c['effect_relative_error']<=1e-3 for c in cells if c['arm']=='token_fp32')
    partition=relative(writes[:,4]+writes[:,5],writes[:,2]);cc=aa and max(child_replays)<=1e-5 and partition<=1e-6
    interactions=[]
    for family in range(4):
        ids=[i for i,row in enumerate(rows) if row['family']==family]
        interaction=effects[2,ids,0]-effects[4,ids,0]-effects[5,ids,0]
        interactions.append(dict(family=family,interaction_norm=float(interaction.norm()),relative_to_joint_effect=float(interaction.norm()/effects[2,ids,0].norm())))
    result={'pred_a':aa,'pred_b':bb,'pred_c':cc,'cells':cells,'prior_replay':prior_replay,'child_partition_error':partition,'child_reference_max_error':max(child_replays),'interactions':interactions}
    torch.save(dict(pre=pre.cpu(),write_vertices=writes.cpu(),baseline_margins=baseline.cpu(),removal_margins=arms.cpu(),worst=worst),art)
    result['worst_case']={k:v for k,v in worst.items() if not torch.is_tensor(v)}
    result.update(seconds=time.perf_counter()-tic,body_forwards=count,artifact_sha=digest(art),source_shas=binding,
                  scope='Token lookup package executes on query/current/tokenID/R, without first-state inputs. OldFP32 and tokenFP32 compare to original factors. All-vocab first-state lookup closes that dependency; current/query generators and final suffix remain native.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
