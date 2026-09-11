#!/usr/bin/env python3
# BQGATE: 18 body forwards, 144 sequences length7-9, 704 tail rows, no fitting.
"""Fresh lexical/construction validation of frozen ambient output branches.
A bound rows, physical relative replay<=1e-5 and CE identity<=1e-9.
B native capability>=.85 eachfamily/side/direction.
C private swap>=80%whole and>=.05 eachtaskdirection.
D private target contrast attenuation>=80%whole and>=10%nativegap eachtaskdirection.
E private neighbor meanabsfullCE<=.05 and abscontrastattenuation<=5%nativegap eachfamily.
F private neighbor binaryCEmeanabs<=.02 and pairmassRMS/fullCEchangeRMS>=.75 eachfamily.
G private/shared margin nonadditivity<=10%meanabswhole eachtaskfamily.
Null: frozen separation fails new lexemes/constructions. No fitting or outcome filtering.
45products,48readers,6912materialized writerfloats, nativebackground retained.
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
STEM='NATIVE_RELATION_OUTPUT_FRESH_V1'


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
    assert len(rows)==64 and authority['authority_sha256']=='4d3a27fd8906a873d2d2e193d8a037ce0312ace815de8a8c47135f56b7c9f2b0'
    assert hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()==authority['authority_sha256']
    sequences=[];buckets={}
    for row in rows:
        for side in ('base','donor'):
            ids=row[side+'_ids'];assert row[side+'_prediction_position']==len(ids)-1
            index=len(sequences);sequences.append(ids);buckets.setdefault(len(ids),[]).append(index)
    assert len(sequences)==128 and sum((len(v)+7)//8 for v in buckets.values())+2==18 and set(buckets)=={7,8,9}
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
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=18 and args[0].shape[1]<=9
    handle=model.transformer.h[0].attn.register_forward_pre_hook(counter)
    ports={k:torch.empty(128,1152) for k in ('input','pre','native_output')};controls=[]
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
    assert counts==[18,144]
    split=torch.load(P/'NATIVE_RELATION_OUTPUT_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    x=ports['input'].double().cuda();lead,rest=evaluate(program,x);scalar=lead+rest
    matrices=[program['writers'],split['splits']['ambient']['private_writers'].cuda(),split['splits']['ambient']['shared_writers'].cuda()]
    writes=[scalar@v.T for v in matrices];h=(ports['pre']+ports['native_output']).cuda()
    records=[];identities=[];tail_rows=0
    for i,row in enumerate(rows):
        b,d=2*i,2*i+1;states=[h[b],h[d]]
        for w in writes:states.extend([h[b]+(w[d]-w[b]).float(),h[b]-w[b].float(),h[d]-w[d].float()])
        z=logits(torch.stack(states)).double();tail_rows+=len(states)
        targets=torch.tensor([row['donor_answer_id'] if j==1 or (j>=2 and (j-2)%3==2) else row['base_answer_id'] for j in range(11)],device='cuda')
        foils=torch.tensor([row['donor_foil_id'] if j==1 or (j>=2 and (j-2)%3==2) else row['base_foil_id'] for j in range(11)],device='cuda')
        za=z.gather(1,targets[:,None]).squeeze(1);zf=z.gather(1,foils[:,None]).squeeze(1)
        ce=F.cross_entropy(z,targets,reduction='none');binary=F.softplus(zf-za)
        mass=torch.logsumexp(z,-1)-torch.logaddexp(za,zf)
        identities.append(float((ce-binary-mass).abs().max()))
        margin=z[:,row['donor_answer_id']]-z[:,row['donor_foil_id']]
        arms=[]
        for arm in range(3):
            k=2+3*arm
            arms.append(dict(swap_effect=float(margin[k]-margin[0]),
                zero_ce=[float(ce[k+1]-ce[0]),float(ce[k+2]-ce[1])],
                zero_binary=[float(binary[k+1]-binary[0]),float(binary[k+2]-binary[1])],
                zero_pairmass=[float(mass[k+1]-mass[0]),float(mass[k+2]-mass[1])],
                contrast_attenuation=float((margin[1]-margin[0])-(margin[k+2]-margin[k+1]))))
        records.append(dict(row_id=row['row_id'],family=row['family'],direction=row['direction'],lexeme=row['lexeme'],
            native_correct=[bool(za[j]>zf[j]) for j in (0,1)],native_gap=float(margin[1]-margin[0]),arms=arms,
            margin_nonadditivity=arms[0]['swap_effect']-arms[1]['swap_effect']-arms[2]['swap_effect']))
    assert tail_rows==704
    cells=[]
    for family in ('A1','A2','past','progressive'):
        directions=('base_to_suffix','suffix_to_base') if family in ('A1','A2') else ('base_to_inflected',)
        for direction in directions:
            local=[r for r in records if r['family']==family and r['direction']==direction]
            gap=sum(r['native_gap'] for r in local)/len(local)
            arms=[]
            for j in range(3):
                full=torch.tensor([v for r in local for v in r['arms'][j]['zero_ce']],dtype=torch.float64)
                binary=torch.tensor([v for r in local for v in r['arms'][j]['zero_binary']],dtype=torch.float64)
                mass=torch.tensor([v for r in local for v in r['arms'][j]['zero_pairmass']],dtype=torch.float64)
                arms.append(dict(swap_effect=sum(r['arms'][j]['swap_effect'] for r in local)/len(local),
                    zero_meanabs_ce=float(full.abs().mean()),zero_meanabs_binary=float(binary.abs().mean()),
                    zero_meanabs_pairmass=float(mass.abs().mean()),pairmass_relative_rms=float(mass.norm()/full.norm()),
                    contrast_attenuation=sum(r['arms'][j]['contrast_attenuation'] for r in local)/len(local)))
            cells.append(dict(family=family,direction=direction,n=len(local),native_gap=gap,arms=arms,
                capability=[sum(r['native_correct'][j] for r in local)/len(local) for j in (0,1)]))
    target=[c for c in cells if c['family'] in ('A1','A2')];control=[c for c in cells if c['family'] in ('past','progressive')]
    composition=[]
    for family in ('A1','A2'):
        local=[r for r in records if r['family']==family]
        cross=sum(abs(r['margin_nonadditivity']) for r in local)/len(local)
        whole=sum(abs(r['arms'][0]['swap_effect']) for r in local)/len(local)
        composition.append(dict(family=family,meanabs_cross=cross,meanabs_whole=whole,held=cross<=.1*whole))
    a=all(c['input_error']<=1e-6 and c['logit_error']<=1e-5 for c in controls) and max(identities)<=1e-9 and bool(torch.isfinite(scalar).all())
    a=a and all(torch.isfinite(torch.tensor([v for r in records for arm in r['arms'] for v in arm['zero_ce']])).tolist())
    b=a and all(min(c['capability'])>=.85 for c in cells)
    torch.save(dict(ports=ports,rows_sha256=digest(P/(STEM+'_ROWS.json'))),cachepath)
    result={'pred_a':a,'pred_b':b,
        'pred_c':b and all(c['arms'][0]['swap_effect']>0 and c['arms'][1]['swap_effect']>=max(.05,.8*c['arms'][0]['swap_effect']) for c in target),
        'pred_d':b and all(c['native_gap']>0 and c['arms'][0]['contrast_attenuation']>0 and c['arms'][1]['contrast_attenuation']>=max(.8*c['arms'][0]['contrast_attenuation'],.1*c['native_gap']) for c in target),
        'pred_e':b and all(c['native_gap']>0 and c['arms'][1]['zero_meanabs_ce']<=.05 and abs(c['arms'][1]['contrast_attenuation'])<=.05*c['native_gap'] for c in control),
        'pred_f':b and all(c['arms'][1]['zero_meanabs_binary']<=.02 and c['arms'][1]['pairmass_relative_rms']>=.75 for c in control),
        'pred_g':b and all(c['held'] for c in composition),
        'cells':cells,'composition':composition,'records':records,'physical_controls':controls,'max_probability_identity_error':max(identities),
        'arm_order':['whole','ambient_private','ambient_shared'],'price':authority['price'],
        'cache_sha256':digest(cachepath),'binding_sha256':digest(P/(STEM+'_BINDING.json')),'seconds':time.perf_counter()-started,
        'scope':authority['scope']+' Frozen output branches and full native background; previous failures remain.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
