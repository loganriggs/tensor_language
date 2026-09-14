#!/usr/bin/env python3
# BQGATE:288bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a source sum<=1e-10 and native/post/self/all replay<=1e-5.
pred_b framing error<=.35 and clause error>=.50 EACHfamily.
pred_c clause error<=.35 and framing error>=.50 EACHfamily (opposing).
pred_d winning source control/target RMS<=.50 EACHfamily; neither means fail.
Null: neither frozen semantic source class alone carries O's cue effect.
Price288bodyforwards,48prefixes,180seconds; all native weights retained.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import source_channels,select_sources
from odd_semantic_positions_v1 import semantic_masks
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_SEMANTIC_POSITIONS_NATIVE_V1'
ARM_ORDER=['native','remove_framing','remove_clause','remove_post_union','remove_self','remove_all_odd']


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows'];assert len(rows)==48;validate(rows);masks=semantic_masks(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert not json.loads((P/'ODD_SEMANTIC_POSITIONS_V1_CPU_CONTROL.json').read_text())['pred_a']
        assert json.loads((P/'ODD_SEMANTIC_POSITIONS_V1_CPU_CONTROL_V2.json').read_text())['pred_a']
        print('288bodyforwards;48prefixes; semantic-source CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={};checks=[]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        arm=context['arm']
        if not arm:return output
        current=args[0];first=args[1];channels=source_channels(graph,current,first)
        pieces={k:select_sources(channels,v) for k,v in context['masks'].items()}
        reference=graph.state(current,first_values=first.reshape(*current.shape[:2],9,128)[:,:,8])[2];checks.append(rel(sum(pieces.values()),reference))
        selected={1:pieces['framing'],2:pieces['clause'],3:pieces['framing']+pieces['clause'],4:pieces['self'],5:sum(pieces.values())}[arm]
        return output[0]-(selected@graph.p['output'].double().T).to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook);values=torch.zeros(6,48,2,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for arm in range(6):
                context.update(arm=arm,masks={k:v[None].to('cuda') for k,v in masks[i].items()})
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                values[arm,i,0]=(scores[row['uk_id']]-scores[row['us_id']]).cpu();values[arm,i,1]=(scores[row['control_ids'][0]]-scores[row['control_ids'][1]]).cpu();count+=1
    finally:handle.remove()
    assert count==288
    prior=torch.load(P/'ODD_CONTEXTUAL_POSITIONS_NATIVE_V1_ARTIFACT.pt',weights_only=True)['values']
    anchors={'native':rel(values[0],prior[0]),'post':rel(values[3],prior[3]),'self':rel(values[4],prior[4]),'all_odd':rel(values[5],prior[6])};effect=values-values[0];cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family];cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];control=effect[:,ix,1]
        errors={'framing':rel(cue[1],cue[5]),'clause':rel(cue[2],cue[5]),'post':rel(cue[3],cue[5]),'self':rel(cue[4],cue[5])}
        ratios={name:float(control[arm].square().mean().sqrt()/cue[arm].square().mean().sqrt().clamp_min(1e-8)) for name,arm in [('framing',1),('clause',2)]}
        framing=errors['framing']<=.35 and errors['clause']>=.5;clause=errors['clause']<=.35 and errors['framing']>=.5
        cells.append({'family':family,'cue_errors_vs_all':errors,'control_to_target_rms':ratios,'framing_pass':framing,'clause_pass':clause,'winner_selectivity_pass':(framing and ratios['framing']<=.5) or (clause and ratios['clause']<=.5)})
    torch.save({'values':values,'effects':effect},artifact)
    result={'pred_a':max(anchors.values())<=1e-5 and max(checks)<=1e-10,'pred_b':all(c['framing_pass'] for c in cells),'pred_c':all(c['clause_pass'] for c in cells),'pred_d':all(c['winner_selectivity_pass'] for c in cells),'anchors':anchors,'max_source_recomposition_error':max(checks),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARM_ORDER,'scope':'Frozen framing-versus-quoted-clause O source screen on corrected rows. Native upstream context and suffix retained; no unseen-template, unique semantic unit, compression or quantization claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)


if __name__=='__main__':main()
