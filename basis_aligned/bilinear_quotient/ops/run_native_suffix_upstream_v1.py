#!/usr/bin/env python3
# BQGATE: 16 body forwards, 128 sequences lengths7-9, 832 tail rows, no fitting.
"""Trace new suffix reader fold and screen attention versus incoming residual.
A bound sources and CPU fold identities passed; B native/cached/fold replay<=1e-5 relative,
  previous whole-input private margin replay<=1e-4 nats.
C attention-only mediated swap meanabs effect>=20%bothswap and positive mean
  EACH intended task/direction. D residual/attention margin nonadditivity<=10%
  bothswap meanabs EACHtaskfamily. Individualhead effects descriptive.
Null: incoming residual supplies most change; then trace producers instead of QK fitting.
Mediation changes only the frozen component, not attention's direct output route.
Native routing uses BOTH QK factors, actual RoPE/norms and signed value mix.
Price 45 products per conditional evaluation; 48 native and48 folded readers,
  full native input/background/routing retained. No compression adoption.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from native_relation_split_v1 import evaluate
from native_suffix_attention_fold_v1 import scalar_from_reads
STEM='NATIVE_SUFFIX_UPSTREAM_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    fold_receipt=json.loads((P/'NATIVE_SUFFIX_ATTENTION_FOLD_V1.json').read_text());assert fold_receipt['pred_a']
    authority=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text());rows=authority['rows']
    prior=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_RESULT.json').read_text());assert prior['pred_a'] and prior['pred_b']
    old=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')
    assert digest(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt')==prior['cache_sha256']
    sequences=[];buckets={}
    for row in rows:
        for side in ('base','donor'):
            ids=row[side+'_ids'];assert row[side+'_prediction_position']==len(ids)-1
            j=len(sequences);sequences.append(ids);buckets.setdefault(len(ids),[]).append(j)
    assert len(rows)==64 and len(sequences)==128 and sum((len(v)+7)//8 for v in buckets.values())==16 and set(buckets)=={7,8,9}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=16,sequences=128,tail_rows=832,fitting=False)));return
    output=P/(STEM+'_RESULT.json');cachepath=P/(STEM+'_PORTS.pt');assert not output.exists() and not cachepath.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];o=last.attn.c_proj.weight.float()
    program=torch.load(P/'NATIVE_RELATION_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    program={k:([{a:b.cuda() if isinstance(b,torch.Tensor) else b for a,b in c.items()} for c in v] if k=='components' else v.cuda()) for k,v in program.items()}
    fold=torch.load(P/'NATIVE_SUFFIX_ATTENTION_FOLD_V1.pt',weights_only=True,map_location='cpu')
    reader=fold['readers'].cuda();folded=fold['folded_output'].cuda()
    split=torch.load(P/'NATIVE_RELATION_OUTPUT_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    writer=split['splits']['ambient']['private_writers'].cuda()
    saved={k:torch.empty(128,1152) for k in ('residual','head_values','mlp16_output')};captured={};count=[0,0]
    def output_hook(module,args,out):captured['mlp16']=out[:,-1].detach()
    def head_hook(module,args):captured['head_values']=args[0][:,-1].detach()
    def count_hook(module,args):
        count[0]+=1;count[1]+=len(args[0]);assert count[0]<=16 and args[0].shape[1]<=9
    handles=[model.transformer.h[16].mlp.register_forward_hook(output_hook),last.attn.c_proj.register_forward_pre_hook(head_hook),
        model.transformer.h[0].attn.register_forward_pre_hook(count_hook)]
    errors=[]
    try:
        for length,indices in sorted(buckets.items()):
            for off in range(0,len(indices),8):
                selected=indices[off:off+8];tokens=torch.tensor([sequences[j] for j in selected],device='cuda')
                x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
                for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
                r=last.lambdas[0]*x+last.lambdas[1]*x0
                attention,v1=last.attn(F.rms_norm(r,(1152,)),v1)
                pre=r[:,-1]+attention[:,-1];xin=F.rms_norm(pre,(1152,))
                y=captured['head_values']
                saved['residual'][selected]=r[:,-1].cpu();saved['head_values'][selected]=y.cpu();saved['mlp16_output'][selected]=captured['mlp16'].cpu()
                for actual,key in [(pre,'pre'),(xin,'input')]:
                    ref=old['ports'][key][selected].cuda();errors.append(float((actual-ref).norm()/ref.norm()))
                rho=(pre.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
                reads=(r[:,-1].double()@reader.T+y.double()@folded.T)/rho
                scalar=scalar_from_reads(program,reads,pre.double().square().sum(-1)/rho[:,0].square())
                a,b=evaluate(program,xin.double());errors.append(float((scalar-a-b).norm()/(a+b).norm()))
                reconstructed=F.linear(y,o);errors.append(float((reconstructed-attention[:,-1]).norm()/attention[:,-1].norm()))
    finally:
        for handle in handles:handle.remove()
    assert count==[16,128]
    r=saved['residual'].cuda();y=saved['head_values'].cuda();attention=F.linear(y,o)
    pre=old['ports']['pre'].cuda();h=(old['ports']['pre']+old['ports']['native_output']).cuda()
    records=[];replay=[]
    for i,row in enumerate(rows):
        b,d=2*i,2*i+1
        inputs=[pre[b],pre[d],pre[b]+r[d]-r[b],pre[b]+attention[d]-attention[b]]
        for head in range(9):
            sl=slice(128*head,128*(head+1));change=F.linear(y[d,sl]-y[b,sl],o[:,sl])
            inputs.append(pre[b]+change)
        a,bpart=evaluate(program,F.rms_norm(torch.stack(inputs),(1152,)).double());scalar=a+bpart
        states=h[b]+((scalar-scalar[:1])@writer.T).float()
        logits=30*torch.tanh(model.lm_head(F.rms_norm(states,(1152,)))/30)
        margins=(logits[:,row['donor_answer_id']]-logits[:,row['donor_foil_id']]).double();effects=margins-margins[0]
        replay.append(abs(float(effects[1])-prior['records'][i]['arms'][1]['swap_effect']))
        records.append(dict(row_id=row['row_id'],family=row['family'],direction=row['direction'],effects=effects.tolist(),
            residual_attention_cross=float(effects[1]-effects[2]-effects[3])))
    cells=[]
    for c in prior['cells']:
        local=[v for v in records if v['family']==c['family'] and v['direction']==c['direction']]
        effects=torch.tensor([v['effects'] for v in local],dtype=torch.float64)
        cells.append(dict(family=c['family'],direction=c['direction'],n=len(local),mean_effects=effects.mean(0).tolist(),
            meanabs_effects=effects.abs().mean(0).tolist(),attention_abs_fraction=float(effects[:,3].abs().mean()/effects[:,1].abs().mean())))
    compositions=[]
    for family in ('A1','A2'):
        local=[v for v in records if v['family']==family]
        cross=sum(abs(v['residual_attention_cross']) for v in local)/len(local)
        whole=sum(abs(v['effects'][1]) for v in local)/len(local)
        compositions.append(dict(family=family,meanabs_cross=cross,meanabs_whole=whole,relative_cross=cross/whole))
    b=max(errors)<=1e-5 and max(replay)<=1e-4
    torch.save(dict(ports=saved,rows_sha256=digest(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json'),
        block17_lambdas=last.lambdas.detach().cpu(),native_endpoint_cache='NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt'),cachepath)
    result={'pred_a':fold_receipt['pred_a'],'pred_b':b,
        'pred_c':b and all(c['attention_abs_fraction']>=.2 and c['mean_effects'][3]>0 for c in cells if c['family'] in ('A1','A2')),
        'pred_d':b and all(c['relative_cross']<=.1 for c in compositions),'cells':cells,'compositions':compositions,
        'records':records,'relative_replay_errors':errors,'max_old_margin_replay_nats':max(replay),
        'arm_order':['base','both','residual','attention']+['head_'+str(j) for j in range(9)],
        'price':dict(body_forwards=16,sequences=128,tail_rows=832),'seconds':time.perf_counter()-started,
        'cache_sha256':digest(cachepath),'binding_sha256':digest(P/(STEM+'_BINDING.json')),
        'scope':'Frozen component-mediated input swaps only; direct attention output route held native. Descriptive individual heads, no new circuit promotion or upstream extraction.'}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('records','relative_replay_errors')},indent=2))


if __name__=='__main__':main()
