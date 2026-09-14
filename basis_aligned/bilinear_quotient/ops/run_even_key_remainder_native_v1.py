#!/usr/bin/env python3
# BQGATE:208bodyforwards;104prefixes<=247tokens;120seconds;no fitting.
"""pred_a all old/new native anchors and component sum replay<=1e-5.
pred_b paired target effect norm of remainder<=.1scalar EACH3families.
pred_c composed separate effects/even effect<=.05 BOTHreadouts/EACHfamily;
newline CE composition<=.05 EACHhalf.
pred_d remainder meanabscontrol>=scalar EACHfamily; meanabsCE>=2scalar EACHhalf.
Null: nonlinear suffix prevents separating scalar target and added collateral.
Price208bodyforwards,104prefixes,120seconds; existing fixed weights/panel.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
from key_span_reuse_v1 import KeySpanParent
from even_key_value_native_backend_v1 import FullValueComponents
STEM='EVEN_KEY_REMAINDER_NATIVE_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    panel=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())
    regional,natural=panel['regional'],panel['natural']
    validate(regional[:48]);validate(regional[48:],expected_token_differences=2)
    assert len(regional)==72 and len(natural)==32
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('208bodyforwards:104prefixes native+remove full-even minus fixed scalar; old-arm reuse gated by native replay');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(120);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval()
    components=FullValueComponents(torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True),'cuda')
    scalar=KeySpanParent({k:v.cuda() for k,v in torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True).items()})
    writer=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'][1].cuda()
    context={};checks=[];count=0
    def hook(module,args,output):
        if context['arm']==0:return output
        even,_=components(args[0],args[1].reshape(*args[0].shape[:2],9,128)[:,:,8])
        selected=scalar.scalar(args[0],context['tokens'],1)[...,None]*writer
        remainder=even-selected
        checks.append(float((selected+remainder-even).norm()/even.norm().clamp_min(1e-30)))
        return output[0]-remainder.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook)
    def forward(row,arm):
        nonlocal count
        tokens=torch.tensor([row['ids']],device='cuda');context.update(tokens=tokens,arm=arm)
        x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        count+=1
        return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    reg=torch.zeros(72,2,2,dtype=torch.float64);ce=torch.zeros(32,2,dtype=torch.float64)
    try:
        for i,row in enumerate(regional):
            for arm in range(2):
                scores=forward(row,arm)
                reg[i,arm,0]=(scores[row['uk_id']]-scores[row['us_id']]).cpu()
                reg[i,arm,1]=(scores[row['control_ids'][0]]-scores[row['control_ids'][1]]).cpu()
        for i,row in enumerate(natural):
            for arm in range(2):ce[i,arm]=-forward(row,arm).log_softmax(-1)[198].cpu()
    finally:handle.remove()
    assert count==208
    old=torch.load(P/'EVEN_KEY_FULL_VALUE_NATIVE_V1_ARTIFACT.pt',weights_only=True)
    rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
    anchors=dict(regional=rel(reg[:,0],old['regional'][:,0]),ce=rel(ce[:,0],old['newline_ce'][:,0]))
    cells=[]
    for family in range(3):
        ix=[i for i,r in enumerate(regional) if r['family']==family]
        remainder=reg[ix,1]-reg[ix,0]
        selected=old['regional'][ix,5]-old['regional'][ix,0]
        even=old['regional'][ix,1]-old['regional'][ix,0]
        paired_remainder=remainder[::2,0]-remainder[1::2,0]
        paired_selected=selected[::2,0]-selected[1::2,0]
        ratio=float(paired_remainder.norm()/paired_selected.norm().clamp_min(1e-30))
        composition=[rel(remainder[:,j]+selected[:,j],even[:,j]) for j in range(2)]
        collateral=float(remainder[:,1].abs().mean());scalar_control=float(selected[:,1].abs().mean())
        cells.append(dict(family=family,paired_target_ratio=ratio,composition_errors=composition,
            remainder_control_meanabs=collateral,scalar_control_meanabs=scalar_control,
            target_small=ratio<=.1,composition_pass=max(composition)<=.05,collateral_pass=collateral>=scalar_control,
            remainder_target_meanabs=float(remainder[:,0].abs().mean()),
            scalar_target_meanabs=float(selected[:,0].abs().mean())))
    ncells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(natural) if r['family']==family]
        rem=ce[ix,1]-ce[ix,0];selected=old['newline_ce'][ix,5]-old['newline_ce'][ix,0]
        even=old['newline_ce'][ix,1]-old['newline_ce'][ix,0]
        error=rel(rem+selected,even);ra=float(rem.abs().mean());sa=float(selected.abs().mean())
        ncells.append(dict(family=family,composition_error=error,composition_pass=error<=.05,
                           remainder_meanabs_ce=ra,scalar_meanabs_ce=sa,collateral_pass=ra>=2*sa))
    A=max(anchors.values())<=1e-5 and max(checks)<=1e-5
    torch.save(dict(regional=reg,newline_ce=ce),artifact)
    result={'pred_a':A,'pred_b':A and all(c['target_small'] for c in cells),
            'pred_c':A and all(c['composition_pass'] for c in cells+ncells),
            'pred_d':A and all(c['collateral_pass'] for c in cells+ncells),
            'anchor_replay':anchors,'max_component_sum_error':max(checks),
            'regional':cells,'newline':ncells,'body_forwards':count,'seconds':time.perf_counter()-start,
            'artifact_sha256':digest(artifact),'source_shas':binding,
            'scope':'Physical removal of full-even minus earlier scalar at head9.8; reused native panel and validated prior arms. Conditional computation split, not named new semantic circuit, OOD or whole-model compression.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)


if __name__=='__main__':main()
