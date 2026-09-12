#!/usr/bin/env python3
# BQGATE:8 body forwards;64 sequences of9 tokens;6 batched suffix arms;zero fitting;180sec.
"""pred_a mixture controls and native attention replay<=1e-5 relative;
pred_b collapsed full and block write errors<=.1 at EACH source0/7;
pred_c full-logit removal effect relativeerror<=.1 and reference effectnorm>=1e-6.
Null: coefficient simplification does not preserve native gated removal effects.
Fixed cached FineWeb prefixes, not untouched OOD. All native background retained.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from folded_normalized_router_v1 import rotary
from shared_cubic_source_projection_v1 import cross_factors
from cubic_source_mixture_execute_v1 import execute
STEM='COMMON_QUADRATIC_NATIVE_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/'CUBIC_SOURCE_MIXTURE_EXECUTE_V1_CONTROL.json').read_text());assert max(control.values())<=1e-10
    tokens=torch.load(P/(STEM+'_ROWS.pt'),weights_only=True,map_location='cpu');assert tokens.shape==(64,9)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('8 body forwards;64 cached FineWeb9-token prefixes;no fit;6 suffix arms');return
    out=P/(STEM+'_RESULT.json');art=P/(STEM+'_PORTS.pt');assert not out.exists() and not art.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];captured={};ports={k:[] for k in ('current','first','residual','attention')}
    handle=model.transformer.h[0].attn.register_forward_pre_hook(lambda module,args:captured.update(first=args[0].detach()))
    count=0
    try:
        for off in range(0,64,8):
            x=F.rms_norm(model.transformer.wte(tokens[off:off+8].cuda()),(1152,));x0=x;v1=None
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            r=last.lambdas[0]*x+last.lambdas[1]*x0;current=F.rms_norm(r,(1152,));attention,_=last.attn(current,v1)
            for key,value in dict(current=current,first=captured['first'],residual=r[:,8],attention=attention[:,8]).items():ports[key].append(value.cpu())
            count+=1
    finally:handle.remove()
    assert count==8;ports={k:torch.cat(v) for k,v in ports.items()}
    def w(key):return model.state_dict()[key].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mixvalue=float(last.attn.lamb);value=torch.cat([(1-mixvalue)*w('transformer.h.17.attn.c_v.weight'),mixvalue*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    output=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128);g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1))
    query=ports['current'][:,8].double().cuda();first=ports['first'].double().cuda();current=ports['current'].double().cuda();qr=rotary(8,128).cuda();eps=torch.finfo(torch.float32).eps
    qa=torch.einsum('nd,hkd->nhk',query,q1);qb=torch.einsum('nd,hkd->nhk',query,q2)
    pack=torch.load(P/'CUBIC_SECANT_BLOCK_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['exact_secant'];simple=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms']
    programs=dict(secant=(pack['components'].cuda(),pack['mix'].cuda()),collapsed=(simple.cuda(),torch.eye(16,dtype=torch.float64,device='cuda')))
    sums={name:torch.zeros(64,2,1152,dtype=torch.float64,device='cuda') for name in programs};cells=[];native_sum=torch.zeros_like(query);native_selected=torch.zeros_like(query)
    for pos in range(9):
        source=torch.cat([current[:,pos],first[:,pos]],-1);ka=torch.einsum('nd,hkd->nhk',current[:,pos],k1);kb=torch.einsum('nd,hkd->nhk',current[:,pos],k2)
        gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt())
        rotation=qr.T@rotary(pos,128).cuda();score1=torch.einsum('nhk,kl,nhl->nh',qa,rotation,ka);score2=torch.einsum('nhk,kl,nhl->nh',qb,rotation,kb)
        vv=torch.einsum('nd,hkd->nhk',source,value);native=torch.einsum('nhk,ohk->no',vv*(score1*score2*gate)[...,None],output);native_sum+=native
        if pos not in (0,7):continue
        native_selected+=native
        krot1=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);krot2=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1)
        weights=(q1,krot1,q2,krot2,value,output*g0[None,:,None]);evaluated={}
        for name,(components,matrix) in programs.items():
            terms=execute(query,source,components,matrix,cross_factors(components,*weights));terms*= (gate/g0[None])[:,:,None,None]
            total=terms.sum((1,2));children=terms[:,:,-2:].sum(1);block=children.sum(1);sums[name]+=children;evaluated[name]=(total,block)
        errors=[float((evaluated['collapsed'][i]-evaluated['secant'][i]).norm()/evaluated['secant'][i].norm().clamp_min(1e-30)) for i in (0,1)]
        cells.append(dict(source_position=pos,full_projection_error=errors[0],block_error=errors[1],secant_block_norm=float(evaluated['secant'][1].norm()),native_source_norm=float(native.norm())))
    replay=float((native_sum-ports['attention'].double().cuda()).norm()/ports['attention'].double().norm());pre=(ports['residual']+ports['attention']).cuda()
    def logits(removal):
        z=pre-removal.float();h=z+last.mlp(F.rms_norm(z,(1152,)));return 30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
    zero=torch.zeros_like(pre);base_logits=logits(zero);ref=sums['secant'].sum(1);candidate=sums['collapsed'].sum(1)
    ref_effect=(logits(ref)-base_logits).double();candidate_effect=(logits(candidate)-base_logits).double();child0=(logits(sums['collapsed'][:,0])-base_logits).double();child1=(logits(sums['collapsed'][:,1])-base_logits).double()
    effectnorm=float(ref_effect.norm());effecterror=float((candidate_effect-ref_effect).norm()/ref_effect.norm().clamp_min(1e-30));interaction=float((candidate_effect-child0-child1).norm()/candidate_effect.norm().clamp_min(1e-30))
    magnitudes,indices=candidate_effect.abs().topk(5,dim=-1);signed=candidate_effect.gather(1,indices)
    torch.save(dict(**ports,secant_children=sums['secant'].cpu(),collapsed_children=sums['collapsed'].cpu(),token_ids=tokens),art)
    result={'pred_a':replay<=1e-5,'pred_b':replay<=1e-5 and all(c['full_projection_error']<=.1 and c['block_error']<=.1 for c in cells),'pred_c':replay<=1e-5 and effecterror<=.1 and effectnorm>=1e-6}
    result.update(native_attention_replay_error=replay,cells=cells,removal_effect_norm=effectnorm,removal_effect_relative_error=effecterror,child_logit_nonadditivity=interaction,top_changed_token_ids=indices.cpu().tolist(),top_signed_logit_changes=signed.cpu().tolist(),native_selected_write_norm=float(native_selected.norm()),secant_block_write_norm=float(ref.norm()),body_forwards=count,seconds=time.perf_counter()-tic,artifact_sha=digest(art),source_shas=binding,scope='Frozen cached FineWeb function/removal screen. Actual normalization and full final suffix retained. Whole fitted projection is not the whole attention/model; no semantic selectivity or OOD claim. Parent deletion removes two children together, no refit.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','top_changed_token_ids','top_signed_logit_changes')}),flush=True)
if __name__=='__main__':main()
