#!/usr/bin/env python3
# BQGATE:72bodyforwards;8prefixes;300seconds;no fitting.
"""pred_a source/native anchors<=1e-5abs/1e-6relative; pred_b carry<=.35.
pred_c carry+MLP7<=.20; pred_d exact8corner expansion<=1e-12.
Both head-key factors/RMS remain live; opened-panel sufficiency screen only.
"""
import hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v2 import measure
from run_even_value_factorial_native_v1 import setup
D=P/'extracted_circuits/odd_attention8h2_typed_face_v1'
spec=importlib.util.spec_from_file_location('native',D/'native.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
STEM='TYPED_FACE_KEY_SOURCE_EDIT_V1';ARMS=['native']+[f'source:{i}' for i in range(8)]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
    fold=json.loads((P/'TYPED_FACE_MLP7_KEY_FOLD_V1_RESULT.json').read_text());assert all(fold[k] for k in ['pred_a','pred_b','pred_c'])
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert len(groups)==8 and len(ARMS)==9
        print('72bodyforwards;8prefixes;carry/MLP7/initial full factorial;native fold verified');return
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');b7=model.transformer.h[7];b8=model.transformer.h[8]
    program={k:v.to('cuda') for k,v in torch.load(D/'program.pt',weights_only=True).items()}
    state={};sources=[];source_errors=[]
    def before7(module,args):
        if len(sources)==len(groups):return
        x,_,x0=args;state['u']=module.lambdas[0]*x+module.lambdas[1]*x0;state['e']=x0
    def after7attn(module,args,result):
        if len(sources)<len(groups):state['g']=state['u']+result[0]
    def after7mlp(module,args,result):
        if len(sources)<len(groups):state['m']=result
    def before8attn(module,args):
        if len(sources)==len(groups):return
        city=groups[len(sources)]['city_position'];l0,l1=b8.lambdas
        parts=torch.stack([l0*state['g'][:,city],l0*state['m'][:,city],l1*state['e'][:,city]])
        rebuilt=F.rms_norm(parts.sum(0),(1152,));target=args[0][:,city]
        source_errors.append(float((rebuilt-target).norm()/target.norm()));sources.append(parts.detach().clone())
    handles=[b7.register_forward_pre_hook(before7),b7.attn.register_forward_hook(after7attn),b7.mlp.register_forward_hook(after7mlp),b8.attn.register_forward_pre_hook(before8attn)]
    group_index={(row['context_id'],row['cue']):i for i,row in enumerate(groups)}
    def write(arm,row,donor_row,current,donor,mask):
        i=group_index[(row['context_id'],row['cue'])];bits=int(arm.split(':')[1]);city=row['city_position']
        raw=sum(sources[i^1][j] if bits&(1<<j) else sources[i][j] for j in range(3))
        city_state=F.rms_norm(raw,(1152,))
        return native.execute(program,current,city_state,row['ids'][city],donor_row['ids'][city],city,mask)
    try:measured=measure(model,graph,groups,ARMS,write)
    finally:
        for handle in handles:handle.remove()
    v=expand(measured['values'],mapping,6)
    parent=torch.load(P/'REGIONAL_ENDPOINT_BATCHING_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    composition=torch.load(P/'TYPED_FACE_WRITE_COMPOSITION_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)
    reference=torch.stack([parent[0],composition['values'][composition['arms'].index('0:2')],parent[6]])
    anchor=v[[0,1,8]]-reference;anchor_abs=float(anchor.abs().max());anchor_rel=float(anchor.norm()/reference.norm())
    corners=v[1:]-v[1:2];target=corners[7,:,0]
    cells={cue:[i for i,r in enumerate(rows) if r['cue']==cue] for cue in ['British','American']}
    cells.update({f'endpoint_{e}':[i for i,r in enumerate(rows) if r['endpoint']==e] for e in range(6)})
    metrics={name:{str(mask):float((corners[mask,ids,0]-target[ids]).norm()/target[ids].norm().clamp_min(1e-8)) for mask in range(1,7)} for name,ids in cells.items()}
    dividends=corners.clone()
    for bit in range(3):
        for mask in range(8):
            if mask&(1<<bit):dividends[mask]-=dividends[mask^(1<<bit)]
    closure=float((dividends.sum(0)-corners[7]).abs().max())
    pred_a=max(source_errors)<=1e-5 and anchor_abs<=1e-5 and anchor_rel<=1e-6 and float(target.square().mean().sqrt())>=1e-5
    pred_b=all(m['1']<=.35 for m in metrics.values());pred_c=all(m['3']<=.20 for m in metrics.values());pred_d=closure<=1e-12
    term_stats={str(mask):{'target_rms_logits':float(dividends[mask,:,0].square().mean().sqrt()),'aligned_fraction':float(dividends[mask,:,0]@target/(target@target)),
                'readout_rms_logits':dividends[mask].square().mean(0).sqrt().tolist()} for mask in range(1,8)}
    result={'pred_a':pred_a,'pred_b':pred_b,'pred_c':pred_c,'pred_d':pred_d,'cell_prediction_errors':metrics,'term_stats':term_stats,
      'full_key_increment_rms_logits':float(target.square().mean().sqrt()),'anchor_max_abs':anchor_abs,'anchor_relative':anchor_rel,
      'max_source_current_error':max(source_errors),'mobius_closure':closure,'body_forwards':measured['body_forwards'],'seconds':time.perf_counter()-start,
      'panel_status':'opened four-context causal source screen','source_shas':binding,
      'scope':'All source corners recompute residual/head-key normalizers and both QK factors. Inherited value held donor. No fresh source identification, selective source claim or omission adoption.'}
    torch.save({'values':v,'dividends':dividends,'source_arrays':torch.stack(sources).cpu()},P/(STEM+'_ARTIFACT.pt'))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
