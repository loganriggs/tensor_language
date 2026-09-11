#!/usr/bin/env python3
# BQGATE: 10 body forwards, 80 sequences length7-8, 320 tail rows, no fitting.
"""Frozen suffix component neighboring-inflection removal screen.
A bound sources, finite and native/wholezero physical relative logit replay<=1e-5.
B native answer>foil>=.85 each family/side, without filtering.
C wholezero mean absolute endpoint CE<=.05 nats AND absolute contrast attenuation
  divided by positive native gap<=.05 eachfamily.
D leading and remainder zero each mean absolute endpoint CE<=.05 eachfamily.
Exact native three-readout-span zero is a descriptive scope control.
Null: nearby behaviors are changed, or native capability invalidates the screen.
Price: 45 products,48 dense readers,3writers plus radius and native background.
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
STEM='NATIVE_RELATION_NEIGHBOR_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(path)==sha for path,sha in binding.items())
    authority=json.loads((P/(STEM+'_ROWS.json')).read_text());rows=authority['rows']
    assert len(rows)==32 and authority['authority_sha256']=='05422b5aa3bbe014de3c19c9ab78a7c2a5d653650b5bd0723e305739a6b544f4'
    assert hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()==authority['authority_sha256']
    sequences=[];buckets={}
    for row in rows:
        for side in ('base','donor'):
            ids=row[side+'_ids'];assert row[side+'_prediction_position']==len(ids)-1
            index=len(sequences);sequences.append(ids);buckets.setdefault(len(ids),[]).append(index)
    assert len(sequences)==64 and sum((len(v)+7)//8 for v in buckets.values())+2==10 and set(buckets)=={7,8}
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,**authority['price'],fitting=False)));return
    output=P/(STEM+'_RESULT.json');cachepath=P/(STEM+'_ENDPOINTS.pt')
    assert not output.exists() and not cachepath.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17]
    program=torch.load(P/'NATIVE_RELATION_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    program={k:([{a:b.cuda() if isinstance(b,torch.Tensor) else b for a,b in c.items()} for c in v] if k=='components' else v.cuda()) for k,v in program.items()}
    def writes(x):
        lead,rest=evaluate(program,x.double())
        return lead@program['writers'].T,rest@program['writers'].T
    def logits(h):return 30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
    def prefix(tokens):
        x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
        for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
        x=last.lambdas[0]*x+last.lambdas[1]*x0
        attn,v1=last.attn(F.rms_norm(x,(1152,)),v1)
        pre=x+attn;xin=F.rms_norm(pre,(1152,))
        return xin,pre,last.mlp(xin)
    counts=[0,0]
    def counter(module,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=10 and args[0].shape[1]<=8
    handle=model.transformer.h[0].attn.register_forward_pre_hook(counter)
    ports={k:torch.empty(64,1152) for k in ('input','pre','native_output')};controls=[]
    try:
        first=True
        for length,indices in sorted(buckets.items()):
            for off in range(0,len(indices),8):
                selected=indices[off:off+8];tokens=torch.tensor([sequences[i] for i in selected],device='cuda')
                xin,pre,native=prefix(tokens)
                for name,value in zip(ports,(xin,pre,native)):ports[name][selected]=value[:,-1].cpu()
                if first:
                    assert len(selected)==8
                    for remove in (False,True):
                        captured={}
                        def head_hook(module,args,value):captured['raw']=value[:,-1].detach().clone()
                        def mlp_hook(module,args,value):
                            captured['input_error']=float((args[0]-xin).norm()/xin.norm())
                            lead,rest=writes(args[0])
                            return value-(lead+rest).float() if remove else value
                        hooks=[model.lm_head.register_forward_hook(head_hook),last.mlp.register_forward_hook(mlp_hook)]
                        try:model(tokens,tokens)
                        finally:
                            for hook in hooks:hook.remove()
                        lead,rest=writes(xin[:,-1]);out=native[:,-1]-(lead+rest).float() if remove else native[:,-1]
                        manual=logits(pre[:,-1]+out);physical=30*torch.tanh(captured['raw']/30)
                        controls.append(dict(remove=remove,input_error=captured['input_error'],logit_error=float((manual-physical).norm()/physical.norm())))
                    first=False
    finally:handle.remove()
    assert counts==[10,80]
    x=ports['input'].cuda();native=ports['native_output'].cuda();h=(ports['pre']+ports['native_output']).cuda()
    lead,rest=writes(x);exact=native.double()@program['readouts'].T@program['writers'].T
    records=[]
    for i,row in enumerate(rows):
        endpoint=[]
        for offset,side in enumerate(('base','donor')):
            j=2*i+offset
            states=torch.stack([h[j],h[j]-lead[j].float(),h[j]-rest[j].float(),h[j]-(lead[j]+rest[j]).float(),h[j]-exact[j].float()])
            z=logits(states);ans=row[side+'_answer_id'];foil=row[side+'_foil_id']
            ce=F.cross_entropy(z.double(),torch.full((5,),ans,device='cuda'),reduction='none')
            # All margins have a common orientation: inflected minus base form.
            margin=(z[:,row['donor_answer_id']]-z[:,row['donor_foil_id']]).double()
            endpoint.append(dict(side=side,native_correct=bool(z[0,ans]>z[0,foil]),ce=ce.tolist(),margin=margin.tolist()))
        records.append(dict(row_id=row['row_id'],family=row['family'],verb=row['verb'],endpoints=endpoint))
    families={}
    for family in ('past','progressive'):
        local=[r for r in records if r['family']==family]
        ce=torch.tensor([[e['ce'] for e in r['endpoints']] for r in local],dtype=torch.float64)
        margin=torch.tensor([[e['margin'] for e in r['endpoints']] for r in local],dtype=torch.float64)
        gap=(margin[:,1]-margin[:,0]).mean(0);attenuation=gap[0]-gap
        families[family]=dict(n=len(local),capability=[sum(r['endpoints'][i]['native_correct'] for r in local)/len(local) for i in (0,1)],
            mean_abs_ce=(ce-ce[:,:,:1]).abs().mean((0,1)).tolist(),mean_signed_ce=(ce-ce[:,:,:1]).mean((0,1)).tolist(),
            contrast_gap=gap.tolist(),contrast_attenuation=attenuation.tolist(),
            relative_contrast_attenuation=(attenuation/gap[0]).tolist(),native_gap_positive=bool(gap[0]>0))
    a=all(c['input_error']<=1e-6 and c['logit_error']<=1e-5 for c in controls) and bool(torch.isfinite(lead+rest).all())
    b=a and all(min(f['capability'])>=.85 for f in families.values())
    torch.save(dict(ports=ports,rows_sha256=digest(P/(STEM+'_ROWS.json'))),cachepath)
    result={'pred_a':a,'pred_b':b,
        'pred_c':b and all(f['mean_abs_ce'][3]<=.05 and f['native_gap_positive'] and abs(f['relative_contrast_attenuation'][3])<=.05 for f in families.values()),
        'pred_d':b and all(max(f['mean_abs_ce'][1:3])<=.05 for f in families.values()),
        'families':families,'records':records,'physical_controls':controls,'price':authority['price'],
        'arm_order':['native','zero_leading','zero_remainder','zero_whole','zero_exact_span'],
        'cache_sha256':digest(cachepath),'binding_sha256':digest(P/(STEM+'_BINDING.json')),
        'seconds':time.perf_counter()-started,'scope':authority['scope']+' Frozen component; native background remains. No ordinary replacement or broad circuit promotion.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
