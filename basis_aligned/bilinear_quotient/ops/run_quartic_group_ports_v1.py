#!/usr/bin/env python3
# BQGATE:16bodyforwards,128sequences, lengths7-9; native source cache only, no fit.
"""pred_a bound rows/program, pred_b oldMLP17/MLP16output replay<=1e-5, pred_c16forwards128sequences.
Capture normalizedMLP16input for frozen quartic group validation. Developmental existing panel.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
STEM='QUARTIC_GROUP_PORTS_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    group=json.loads((P/'QUARTIC_GROUP_PROGRAM_V1.json').read_text());assert group['pred_a'] and digest(P/'QUARTIC_GROUP_PROGRAM_V1.pt')==group['artifact_sha256']
    rowsfile=P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json';rows=json.loads(rowsfile.read_text())['rows'];sequences=[];buckets={}
    for row in rows:
        for side in ('base','donor'):
            ids=row[side+'_ids'];assert row[side+'_prediction_position']==len(ids)-1
            j=len(sequences);sequences.append(ids);buckets.setdefault(len(ids),[]).append(j)
    assert len(sequences)==128 and sum((len(v)+7)//8 for v in buckets.values())==16
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=16,sequences=128,maximum_length=9,fitting=False)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PORTS.pt');assert not out.exists() and not ap.exists();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];old=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports']
    upfile=P/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt';upreceipt=json.loads((P/'NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json').read_text());assert digest(upfile)==upreceipt['cache_sha256']
    upstream=torch.load(upfile,weights_only=True,map_location='cpu')['ports'];saved=torch.empty(128,1152);captured={};count=[0,0];errors=[];started=time.perf_counter()
    def pre_hook(module,args):captured['x16']=args[0][:,-1].detach()
    def out_hook(module,args,out):captured['y16']=out[:,-1].detach()
    def count_hook(module,args):count[0]+=1;count[1]+=len(args[0]);assert count[0]<=16 and args[0].shape[1]<=9
    handles=[model.transformer.h[16].mlp.register_forward_pre_hook(pre_hook),model.transformer.h[16].mlp.register_forward_hook(out_hook),model.transformer.h[0].attn.register_forward_pre_hook(count_hook)]
    try:
        for length,ids in sorted(buckets.items()):
            for off in range(0,len(ids),8):
                selected=ids[off:off+8];tokens=torch.tensor([sequences[i] for i in selected],device='cuda')
                x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
                for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
                r=last.lambdas[0]*x+last.lambdas[1]*x0;attention,v1=last.attn(F.rms_norm(r,(1152,)),v1)
                pre=r[:,-1]+attention[:,-1];xin=F.rms_norm(pre,(1152,));native=last.mlp(xin)
                saved[selected]=captured['x16'].cpu()
                for actual,key in [(pre,'pre'),(xin,'input'),(native,'native_output')]:
                    ref=old[key][selected].cuda();errors.append(float((actual-ref).norm()/ref.norm()))
                ref=upstream['mlp16_output'][selected].cuda();errors.append(float((captured['y16']-ref).norm()/ref.norm()))
    finally:
        for h in handles:h.remove()
    assert torch.isfinite(saved).all();torch.save(dict(input16=saved,rows_sha256=digest(rowsfile),group_sha256=group['artifact_sha256']),ap)
    result={'pred_a':True,'pred_b':max(errors)<=1e-5,'pred_c':count==[16,128],'maximum_native_replay_error':max(errors),'counts':count,
        'artifact_sha256':digest(ap),'wall_seconds_excluding_load':time.perf_counter()-started,'scope':'Actual normalizedMLP16inputs on existing developmental morphology rows, no new discovery data or fresh/OOD claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_b'] and result['pred_c']


if __name__=='__main__':main()
