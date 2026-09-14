#!/usr/bin/env python3
# BQGATE:336bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a native/all-O anchors<=1e-5 and four-source sum<=1e-10.
pred_b post+self cue error versus all-O<=.20 EACHfamily.
pred_c post error<=.35 and self error>=.50 EACHfamily (relay).
pred_d self error<=.35 and post error>=.50 EACHfamily (consolidation).
pred_e post+self control/target RMS<=.50 EACHfamily.
Null: earlier sources matter or neither later source class dominates.
Price336bodyforwards,48prefixes,180seconds; all native weights retained.
"""
import json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from run_even_value_factorial_native_v1 import setup,cpu_control
from odd_source_positions_v1 import source_channels,select_sources
from odd_contextual_positions_v1 import contextual_masks
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
STEM='ODD_CONTEXTUAL_POSITIONS_NATIVE_V1'
ARM_ORDER=['native','remove_pre','remove_city','remove_post','remove_self','remove_after','remove_all_odd']


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows']
    assert len(rows)==48;validate(rows);masks=contextual_masks(rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert json.loads((P/'ODD_CONTEXTUAL_POSITIONS_V1_CPU_CONTROL.json').read_text())['pred_a']
        print('336bodyforwards;48prefixes; contextual-source CPUcontrol',cpu_control());return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda');context={};checks=[]
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-8))
    def hook(module,args,output):
        arm=context['arm']
        if not arm:return output
        current=args[0];first=args[1];channels=source_channels(graph,current,first)
        pieces={k:select_sources(channels,v) for k,v in context['masks'].items()}
        reference=graph.state(current,first_values=first.reshape(*current.shape[:2],9,128)[:,:,8])[2]
        checks.append(rel(sum(pieces.values()),reference))
        selected={1:pieces['pre'],2:pieces['city'],3:pieces['post'],4:pieces['self'],5:pieces['post']+pieces['self'],6:sum(pieces.values())}[arm]
        return output[0]-(selected@graph.p['output'].double().T).to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook)
    values=torch.zeros(7,48,2,dtype=torch.float64);count=0
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['ids']],device='cuda')
            for arm in range(7):
                context.update(arm=arm,masks={k:v[None].to('cuda') for k,v in masks[i].items()})
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                values[arm,i,0]=(scores[row['uk_id']]-scores[row['us_id']]).cpu()
                values[arm,i,1]=(scores[row['control_ids'][0]]-scores[row['control_ids'][1]]).cpu();count+=1
    finally:handle.remove()
    assert count==336
    old=torch.load(P/'SRO_ARTICLE_CORRECTION_V1_ARTIFACT.pt',weights_only=True)['cube']
    anchors={'native':rel(values[0],old[0]),'all_odd':rel(values[6],old[4])};effect=values-values[0];cells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(rows) if r['family']==family]
        cue=effect[:,ix[::2],0]-effect[:,ix[1::2],0];control=effect[:,ix,1]
        errors={name:rel(cue[arm],cue[6]) for name,arm in [('pre',1),('city',2),('post',3),('self',4),('after',5)]}
        target_rms=float(cue[5].square().mean().sqrt());control_rms=float(control[5].square().mean().sqrt());selectivity=control_rms/max(target_rms,1e-8)
        cells.append({'family':family,'cue_errors_vs_all':errors,'cue_norms':[float(x.norm()) for x in cue[1:]],'control_norms':[float(x.norm()) for x in control[1:]],'after_target_rms':target_rms,'after_control_rms':control_rms,'after_control_to_target_rms':selectivity,'after_pass':errors['after']<=.2,'relay_pass':errors['post']<=.35 and errors['self']>=.5,'self_pass':errors['self']<=.35 and errors['post']>=.5,'selectivity_pass':selectivity<=.5})
    torch.save({'values':values,'effects':effect},artifact)
    result={'pred_a':max(anchors.values())<=1e-5 and max(checks)<=1e-10,'pred_b':all(c['after_pass'] for c in cells),'pred_c':all(c['relay_pass'] for c in cells),'pred_d':all(c['self_pass'] for c in cells),'pred_e':all(c['selectivity_pass'] for c in cells),'anchors':anchors,'max_source_recomposition_error':max(checks),'families':cells,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'arm_order':ARM_ORDER,'scope':'Token-defined pre/city/post/self localization of O on corrected rows. Full native context and suffix remain; removal at source read does not erase upstream descendants. No semantic identification, OOD, static compression or quantization claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True);signal.alarm(0)


if __name__=='__main__':main()
