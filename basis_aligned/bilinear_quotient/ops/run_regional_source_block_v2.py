#!/usr/bin/env python3
# BQGATE:7 body forwards;32 sequences lengths4-12;8 batched suffix arms;no fit;180sec.
"""pred_a native replay<=1e-5, block collapse<=.1, eachfamily native cue mean>=.1 with>=5/8positive;
pred_b removal reduces native regional cue effect>=10% and positive>=5/8 eachfamily;
pred_c reduction exceeds norm-matched paired-direction control by>=5%native,
 and unrelated contrast differential meanabs<=.5regional meanabs, eachfamily.
Null: frozen block is not a selective regional-spelling mechanism on these templates.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from folded_normalized_router_v1 import rotary
from shared_cubic_source_projection_v1 import cross_factors
from cubic_source_mixture_execute_v1 import execute
STEM='REGIONAL_SOURCE_BLOCK_V2'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];buckets={}
    for i,r in enumerate(rows):buckets.setdefault(len(r['ids']),[]).append(i)
    assert len(rows)==32 and len(buckets)==7 and max(buckets)<=12 and all(len(v)<=8 for v in buckets.values())
    assert json.loads((P/'COMMON_QUADRATIC_NATIVE_V1_RESULT.json').read_text())['pred_a']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('7 body forwards32sequences4-12tokens;8suffix arms;no fitting');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1));eps=torch.finfo(torch.float32).eps
    pack=torch.load(P/'CUBIC_SECANT_BLOCK_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['exact_secant'];simple=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms']
    programs=dict(secant=(pack['components'].cuda(),pack['mix'].cuda()),collapsed=(simple.cuda(),torch.eye(16,dtype=torch.float64,device='cuda')))
    pre=torch.empty(32,1152,device='cuda');writes={name:torch.zeros(32,1152,dtype=torch.float64,device='cuda') for name in ('secant','collapsed','secant_all','collapsed_all')};replays=[];count=0
    handle=model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))
    try:
        for length,ids in sorted(buckets.items()):
            tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1);pre[ids]=(r+attention)[:,-1];count+=1
            current=current.double();first=captured['first'].double();query=current[:,-1];qr=rotary(length-1,128).cuda();qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2);native_sum=torch.zeros_like(query)
            for pos in range(length):
                source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
                gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt());rotation=qr.T@rotary(pos,128).cuda()
                score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb);vv=torch.einsum('nd,hkd->nhk',source,value);native_sum+=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output)
                kr1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kr2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1);weights=(q1,kr1,q2,kr2,value,output*g0[None,:,None])
                for name,(components,matrix) in programs.items():
                    terms=execute(query,source,components,matrix,cross_factors(components,*weights));change=(terms[:,:,-2:]*(gate/g0[None])[:,:,None,None]).sum((1,2));writes[name+'_all'][ids]+=change
                    if pos in (0,length-2):writes[name][ids]+=change
            replays.append(float((native_sum-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:handle.remove()
    assert count==7
    def margins(removal):
        z=pre-removal.float();h=z+last.mlp(F.rms_norm(z,(1152,)));logits=30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
        return torch.stack([torch.stack([logits[i,r['uk_id']]-logits[i,r['us_id']],logits[i,r['control_ids'][0]]-logits[i,r['control_ids'][1]]]) for i,r in enumerate(rows)]).double()
    coverages=[];saved_arms={}
    for coverage,candidate_key,reference_key in [('two_sources','collapsed','secant'),('all_sources','collapsed_all','secant_all')]:
        candidate=writes[candidate_key];swap=candidate[torch.arange(32,device='cuda')^1];swap=swap*candidate.norm(dim=-1,keepdim=True)/swap.norm(dim=-1,keepdim=True).clamp_min(1e-30)
        arms=torch.stack([margins(torch.zeros_like(pre)),margins(candidate),margins(swap),margins(writes[reference_key])],1);cells=[]
        for family in (0,1):
            indices=[i for i,r in enumerate(rows) if r['family']==family and r['cue']=='British'];native=arms[indices,0]-arms[[i+1 for i in indices],0];deleted=arms[indices,1]-arms[[i+1 for i in indices],1];controlled=arms[indices,2]-arms[[i+1 for i in indices],2];reduction=native-deleted;control_reduction=native-controlled
            cells.append(dict(family=family,native_mean=float(native[:,0].mean()),native_positive=int((native[:,0]>0).sum()),removal_mean_reduction=float(reduction[:,0].mean()),removal_positive=int((reduction[:,0]>0).sum()),control_mean_reduction=float(control_reduction[:,0].mean()),regional_meanabs_reduction=float(reduction[:,0].abs().mean()),unrelated_meanabs_reduction=float(reduction[:,1].abs().mean())))
        collapse=float((candidate-writes[reference_key]).norm()/writes[reference_key].norm());instrument=max(replays)<=1e-5 and collapse<=.1;capability=all(c['native_mean']>=.1 and c['native_positive']>=5 for c in cells)
        verdict={'pred_a':instrument and capability,'pred_b':instrument and capability and all(c['removal_mean_reduction']>=.1*c['native_mean'] and c['removal_positive']>=5 for c in cells),'pred_c':instrument and capability and all(c['removal_mean_reduction']-c['control_mean_reduction']>=.05*c['native_mean'] and c['unrelated_meanabs_reduction']<=.5*c['regional_meanabs_reduction'] for c in cells)}
        saved_arms[coverage]=arms.cpu();coverages.append(dict(coverage=coverage,**verdict,instrument_pass=instrument,native_capability_pass=capability,cells=cells,relative_block_collapse_error=collapse))
    torch.save(dict(pre=pre.cpu(),writes={k:v.cpu() for k,v in writes.items()},arm_margins=saved_arms),art)
    result=dict(coverages=coverages,native_replay_errors=replays,arm_order=['native','delete_block','delete_paired_context_direction_normmatched','delete_secant'],arm_margins={k:v.tolist() for k,v in saved_arms.items()},body_forwards=count,seconds=time.perf_counter()-tic,artifact_sha=digest(art),source_shas=binding,scope='V2 article repair and separately declared all-source program. Original V1 failures retained. No fit or four-property promotion.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','arm_margins')}),flush=True)
if __name__=='__main__':main()
