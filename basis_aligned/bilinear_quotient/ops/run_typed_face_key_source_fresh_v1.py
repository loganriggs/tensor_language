#!/usr/bin/env python3
# BQGATE:192bodyforwards;16prefixes;300seconds;no fitting.
"""pred_a source/native anchors<=1e-5abs/1e-6relative; pred_b carry<=.35.
pred_c carry+MLP7<=.20; pred_d frozen key norms<=.10; pred_e closure<=1e-12.
Fresh eight-context sufficiency screen; 192 forwards, two native state ports.
Null hypotheses: source omission or frozen norms exceed their registered errors.
No source-selectivity claim without a matched directional null.
"""
import hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
import torch
import tiktoken
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import typed_face_key_norm_v1 as keynorm
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v2 import measure
from run_even_value_factorial_native_v1 import setup
D=P/'extracted_circuits/odd_attention8h2_typed_face_v1'
spec=importlib.util.spec_from_file_location('native',D/'native.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
STEM='TYPED_FACE_KEY_SOURCE_FRESH_V1';ARMS=['native']+[f'source:{i}' for i in range(8)]+['direct_inherited','direct_full','norm_frozen']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    row_doc=json.loads((P/(STEM+'_ROWS.json')).read_text());rows=row_doc['rows'];groups,mapping=group_rows(rows)
    row_control=json.loads((P/(STEM+'_CPU_CONTROL.json')).read_text());assert row_control['pred_a'] and row_control['prior_source_overlap']==0 and row_control['prior_sequence_overlap']==0
    encoder=tiktoken.get_encoding('gpt2');token_ids=sorted({encoder.encode(city)[0] for pair in row_doc['city_pairs'] for city in pair});assert len(token_ids)==20
    fold=json.loads((P/'TYPED_FACE_MLP7_KEY_FOLD_V1_RESULT.json').read_text());assert all(fold[k] for k in ['pred_a','pred_b','pred_c'])
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert len(groups)==16 and len(ARMS)==12
        print('192bodyforwards;16prefixes;fresh source and normalization rules;20native token generators');return
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');b7=model.transformer.h[7];b8=model.transformer.h[8]
    program={k:v.to('cuda') for k,v in torch.load(D/'program.pt',weights_only=True).items()}
    original_tokens=program['token_ids'].tolist()
    ids=torch.tensor([token_ids],device='cuda')
    initial=F.rms_norm(model.transformer.wte(ids),(1152,));b0=model.transformer.h[0]
    first_input=F.rms_norm((b0.lambdas[0]+b0.lambdas[1])*initial,(1152,))
    table=b0.attn.c_v(first_input)[0,:,256:384].detach().clone()
    old_table_errors=[float((table[token_ids.index(token)]-program['first_table'][i]).norm()/program['first_table'][i].norm()) for i,token in enumerate(original_tokens)]
    program['token_ids']=torch.tensor(token_ids,device='cuda');program['first_table']=table
    program_path=P/(STEM+'_PROGRAM.pt');assert not program_path.exists()
    torch.save({k:v.detach().cpu().clone() for k,v in program.items()},program_path)
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
        city=row['city_position']
        if arm in ['direct_inherited','direct_full','norm_frozen']:
            city_state=current[:,city] if arm=='direct_inherited' else donor[:,city]
            return keynorm.execute(program,current,city_state,row['ids'][city],donor_row['ids'][city],city,mask,freeze_mask=3 if arm=='norm_frozen' else 0)
        i=group_index[(row['context_id'],row['cue'])];bits=int(arm.split(':')[1])
        raw=sum(sources[i^1][j] if bits&(1<<j) else sources[i][j] for j in range(3))
        city_state=F.rms_norm(raw,(1152,))
        return native.execute(program,current,city_state,row['ids'][city],donor_row['ids'][city],city,mask)
    try:measured=measure(model,graph,groups,ARMS,write)
    finally:
        for handle in handles:handle.remove()
    v=expand(measured['values'],mapping,6)
    reference=v[[9,10]]
    anchor=v[[1,8]]-reference;anchor_abs=float(anchor.abs().max());anchor_rel=float(anchor.norm()/reference.norm())
    corners=v[1:9]-v[1:2];target=corners[7,:,0]
    cells={cue:[i for i,r in enumerate(rows) if r['cue']==cue] for cue in ['British','American']}
    cells.update({f'endpoint_{e}':[i for i,r in enumerate(rows) if r['endpoint']==e] for e in range(6)})
    metrics={name:{str(mask):float((corners[mask,ids,0]-target[ids]).norm()/target[ids].norm().clamp_min(1e-8)) for mask in range(1,7)} for name,ids in cells.items()}
    dividends=corners.clone()
    for bit in range(3):
        for mask in range(8):
            if mask&(1<<bit):dividends[mask]-=dividends[mask^(1<<bit)]
    closure=float((dividends.sum(0)-corners[7]).abs().max())
    pred_a=max(source_errors)<=1e-5 and anchor_abs<=1e-5 and anchor_rel<=1e-6 and float(target.square().mean().sqrt())>=1e-5
    native_pair=v[0,::2,0]-v[0,1::2,0];capable=int((native_pair>=.1).sum())
    pred_a=pred_a and capable>=36 and max(old_table_errors)<=1e-5
    pred_b=all(m['1']<=.35 for m in metrics.values());pred_c=all(m['3']<=.20 for m in metrics.values())
    face=v[10,:,0]-v[0,:,0];frozen=v[11,:,0]-v[0,:,0]
    norm_errors={name:float((frozen[ids]-face[ids]).norm()/face[ids].norm().clamp_min(1e-8)) for name,ids in cells.items()}
    pred_d=all(e<=.1 for e in norm_errors.values());pred_e=closure<=1e-12 and row_control['prior_source_overlap']==0 and row_control['prior_sequence_overlap']==0 and bool(torch.isfinite(v).all())
    term_stats={str(mask):{'target_rms_logits':float(dividends[mask,:,0].square().mean().sqrt()),'aligned_fraction':float(dividends[mask,:,0]@target/(target@target)),
                'readout_rms_logits':dividends[mask].square().mean(0).sqrt().tolist()} for mask in range(1,8)}
    result={'pred_a':pred_a,'pred_b':pred_b,'pred_c':pred_c,'pred_d':pred_d,'pred_e':pred_e,'native_capable_cells':capable,'native_pair_cells':48,'normalizer_cell_errors':norm_errors,'old_table_relative_errors':old_table_errors,'newly_tested_city_tokens':sorted({r['ids'][r['city_position']] for r in rows}-set(original_tokens)),'expanded_program_sha256':digest(program_path),'cell_prediction_errors':metrics,'term_stats':term_stats,
      'all_readout_key_increment_rms_logits':corners[7].square().mean(0).sqrt().tolist(),'all_readout_full_face_rms_logits':(v[10]-v[0]).square().mean(0).sqrt().tolist(),'full_key_increment_rms_logits':float(target.square().mean().sqrt()),'anchor_max_abs':anchor_abs,'anchor_relative':anchor_rel,
      'max_source_current_error':max(source_errors),'mobius_closure':closure,'body_forwards':measured['body_forwards'],'seconds':time.perf_counter()-start,
      'panel_status':'fresh eight-source-context screen with city shifted to offset8; cached corpus not pretraining-disjoint','source_shas':binding,
      'scope':'Prospective transfer of frozen source/norm rules to8unused regional sources with shifted city offset and20native token generators. Both QK factors retained. No source-selectivity, simplicity or complete-circuit certification.'}
    torch.save({'values':v,'dividends':dividends,'source_arrays':torch.stack(sources).cpu()},P/(STEM+'_ARTIFACT.pt'))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
