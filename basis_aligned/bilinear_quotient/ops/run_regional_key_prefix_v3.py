#!/usr/bin/env python3
# BQGATE:112 existing rows through14blocks;12 short prefixes plus zero anchors;180sec;no fit;compact artifact.
"""pred_a prefix joint-key and residual-unroll relative replay <=2e-5.
pred_b pred_a and attention-update-only joint-key error <=.1 at all3layers.
pred_c pred_a and MLP-update-only joint-key error <=.1 at all3layers.
Null: neither update family independently closes contextual joint keys.
Price: at most40 batches through14blocks, no logits or optimization; <10MB artifact.
"""
import os, sys, json, time, signal
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
P = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F
from jacclust.tt_model import apply_rotary_emb
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM = 'REGIONAL_KEY_PREFIX_V3'
LAYERS = (8, 9, 13)


@torch.no_grad()
def main():
    binding = json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k) == v for k,v in binding.items())
    rows, prefixes = [], set()
    for stem in ['REGIONAL_SOURCE_BLOCK_V2','REGIONAL_SOURCE_BLOCK_OOD_V1','REGIONAL_BEHAVIOR_CONTROLS_V2']:
        panel = json.loads((P/(stem+'_ROWS.json')).read_text())['rows']; validate(panel)
        for i in range(0,len(panel),2):
            a,b = panel[i]['ids'],panel[i+1]['ids']
            positions = [j for j,(x,y) in enumerate(zip(a,b)) if x != y]
            assert len(a)==len(b) and len(positions)==1
            c=positions[0]
            for ids in (a,b):
                prefix=tuple(ids[:c+1]); prefixes.add(prefix); rows.append((ids,c,prefix))
    prefixes=sorted(prefixes); assert len(rows)==112 and len(prefixes)==12
    buckets={}
    for row in rows: buckets.setdefault(len(row[0]),[]).append(row)
    batches=[v[o:o+8] for v in buckets.values() for o in range(0,len(v),8)]
    lengths=sorted(set(map(len,prefixes)))
    assert len(batches)+len(prefixes)+len(lengths)<=40
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(full_batches=len(batches),prefixes=len(prefixes),zero_anchors=len(lengths),blocks=14))); return
    out=P/(STEM+'_RESULT.json'); art=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not art.exists()
    signal.alarm(180); tic=time.perf_counter(); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval(); calls=0
    def run(ids=None,length=None):
        nonlocal calls
        x0=torch.zeros(1,length,1152,device='cuda') if ids is None else F.rms_norm(model.transformer.wte(torch.tensor(ids,device='cuda')),(1152,))
        x=x0; v=None; raw={}; updates=[]
        for j,block in enumerate(model.transformer.h[:14]):
            x=block.lambdas[0]*x+block.lambdas[1]*x0
            if j in LAYERS: raw[j]=x.clone()
            attention,v=block.attn(F.rms_norm(x,(1152,)),v)
            x=x+attention; mlp=block.mlp(F.rms_norm(x,(1152,))); x=x+mlp
            updates.append((attention.clone(),mlp.clone()))
        calls+=1
        return x0,raw,updates
    def keys(raw,j):
        module=model.transformer.h[j].attn
        inp=F.rms_norm(raw.float(),(1152,)); n,t,_=inp.shape
        cos,sin=module.rotary(inp.reshape(n,t,9,128))
        return tuple(apply_rotary_emb(F.rms_norm(getattr(module,name)(inp).reshape(n,t,9,128),(128,)),cos,sin).double() for name in ('c_k','c_k2'))
    def joint(k): return k[0][..., :,None]*k[1][...,None,:]
    def rel(x,y): return float((x-y).norm()/y.norm().clamp_min(1e-30))
    zeros={t:run(length=t) for t in lengths}
    cache={}; cells=[]; replay=[]; saved=[]
    for prefix in prefixes:
        x0,raw,updates=run([list(prefix)]); _,zero,zupdates=zeros[len(prefix)]
        cache[prefix]={j:tuple(k[0,-1] for k in keys(raw[j],j)) for j in LAYERS}
        for j in LAYERS:
            e=x0.new_tensor(1.)
            for block in model.transformer.h[:j+1]: e=block.lambdas[0]*e+block.lambdas[1]
            base=zero[j].double()+e.double()*x0.double(); da=torch.zeros_like(base); dm=da.clone(); parts=[]
            for k in range(j):
                scale=torch.ones((),device='cuda',dtype=torch.float64)
                for block in model.transformer.h[k+1:j+1]: scale*=block.lambdas[0].double()
                pa=scale*(updates[k][0].double()-zupdates[k][0].double())
                pm=scale*(updates[k][1].double()-zupdates[k][1].double())
                da+=pa; dm+=pm; parts.extend([pa[0,-1].cpu().float(),pm[0,-1].cpu().float()])
            replay.append(rel(base+da+dm,raw[j].double()))
            reference=joint(cache[prefix][j])
            for name,state in [('anchor',base),('attention',base+da),('mlp',base+dm),('all',base+da+dm)]:
                predicted=joint(tuple(k[0,-1] for k in keys(state,j)))
                cells.append(dict(prefix=list(prefix),layer=j,arm=name,error=rel(predicted,reference)))
            saved.append(dict(prefix=list(prefix),layer=j,anchor=base[0,-1].cpu().float(),native=raw[j][0,-1].cpu(),parts=torch.stack(parts)))
    prefix_errors=[]
    for batch in batches:
        _,raw,_=run([r[0] for r in batch])
        for j in LAYERS:
            k=keys(raw[j],j)
            for i,(_,c,prefix) in enumerate(batch):
                prefix_errors.append(rel(joint((k[0][i,c],k[1][i,c])),joint(cache[prefix][j])))
    maxima={arm:max(c['error'] for c in cells if c['arm']==arm) for arm in ['anchor','attention','mlp','all']}
    a=max(prefix_errors+replay+[maxima['all']])<=2e-5
    result={'pred_a':a,'pred_b':a and maxima['attention']<=.1,'pred_c':a and maxima['mlp']<=.1}
    result.update(
                max_prefix_joint_key_error=max(prefix_errors),max_raw_unroll_error=max(replay),max_errors=maxima,cells=cells,
                calls=calls,seconds=time.perf_counter()-tic,source_shas=binding,
                scope='Frozen conditional joint-key sufficiency of upstream update families. All original upstream updates are computed natively; family removal is at the projected residual port, not recursive network ablation. No behavioral or independent extraction claim. Prefix counts are finite panel coverage.')
    torch.save(saved,art); result['artifact_sha']=digest(art)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','source_shas')},indent=2),flush=True)


if __name__=='__main__': main()
