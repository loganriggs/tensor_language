#!/usr/bin/env python3
# BQGATE:768bodyforwards;104prefixes<=247tokens;300seconds;no fitting.
"""pred_a live recomposition<=1e-5, selfdonor<=1e-4, native capability.
pred_b EACH3families even removal/donor>=.5 with10/12 and20/24positive;
both unrelated ratios<=.5.
pred_c BOTHnaturalhalves meanabsCE<=.02,maxabs<=.1 for even removal.
pred_d historical native/scalar-removal/recomposition score replay<=1e-5.
Null: all-value extension loses selectivity despite exact shared routing.
Price768bodyforwards,104prefixes,300seconds,fullnativeprefix/suffix retained.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from regional_cue_row_check_v1 import validate
from key_span_reuse_v1 import KeySpanParent
from even_key_value_native_backend_v1 import FullValueComponents

STEM='EVEN_KEY_FULL_VALUE_NATIVE_V1'
ARMS=['native','remove_even','remove_odd','remove_full','recompose','remove_scalar','donor_even','selfdonor_even']


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items()),'bound dependency changed'
    panel=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())
    regional,natural=panel['regional'],panel['natural']
    assert len(regional)==72 and len(natural)==32
    validate(regional[:48]);validate(regional[48:],expected_token_differences=2)
    assert max(len(r['ids']) for r in regional+natural)<=247
    for i in range(0,72,2):assert len(regional[i]['ids'])==len(regional[i+1]['ids'])
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('768bodyforwards:72regional8arms+32natural6arms; reused confirmation panel; full-value even/odd');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    tic=time.perf_counter();signal.alarm(300);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval()
    program=torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True)
    components=FullValueComponents(program,'cuda')
    compact={k:v.cuda() for k,v in torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True).items()}
    scalar=KeySpanParent(compact)
    writers=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'].cuda()
    context={};donors={};checks=[];sizes=[];count=0
    attn=model.transformer.h[9].attn
    def hook(module,args,output):
        current=args[0];i=context['row'];arm=context['arm']
        first=args[1].reshape(*current.shape[:2],9,128)[:,:,8]
        even,odd=components(current,first)
        projected=context['preov'].reshape(*current.shape[:2],9,128)[:,:,8]
        full=F.linear(projected,module.c_proj.weight[:,8*128:9*128].to(projected.dtype))
        checks.append(float((even+odd-full.double()).norm()/full.double().norm().clamp_min(1e-30)))
        if arm==0:
            donors[i]=even.clone()
            sizes.append(dict(pool=context['pool'],row=i,even_norm=float(even.norm()),odd_norm=float(odd.norm()),full_norm=float(full.norm())))
            return output
        if arm==1:delta=even
        elif arm==2:delta=odd
        elif arm==3:delta=full
        elif arm==4:delta=full.double()-even-odd
        elif arm==5:
            value=scalar.scalar(current,context['tokens'],1)
            delta=value[...,None]*writers[1]
        elif arm==6:delta=even-donors[i^1]
        elif arm==7:delta=even-donors[i]
        else:raise ValueError(arm)
        return output[0]-delta.to(output[0].dtype),output[1]
    handles=[attn.c_proj.register_forward_pre_hook(lambda m,a:context.update(preov=a[0])),
             attn.register_forward_hook(hook)]
    def forward(row,i,arm,pool):
        nonlocal count
        tokens=torch.tensor([row['ids']],device='cuda')
        context.update(tokens=tokens,row=i,arm=arm,pool=pool)
        x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
        for block in model.transformer.h:x,v1=block(x,v1,x0)
        count+=1
        return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    reg=torch.zeros(72,8,2,dtype=torch.float64)
    ce=torch.zeros(32,6,dtype=torch.float64);margin=torch.zeros_like(ce)
    def regional_score(i,arm):
        row=regional[i];logits=forward(row,i,arm,'regional')
        reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu()
        reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
    try:
        for pair in range(0,72,2):
            donors.clear()
            for i in [pair,pair+1]:regional_score(i,0)
            for i in [pair,pair+1]:
                for arm in range(1,8):regional_score(i,arm)
        donors.clear()
        for i,row in enumerate(natural):
            for arm in range(6):
                logits=forward(row,i,arm,'natural')
                ce[i,arm]=-logits.log_softmax(-1)[198].cpu()
                margin[i,arm]=(logits[198]-logits[11]).cpu()
    finally:
        for h in handles:h.remove()
    assert count==768
    cells=[]
    for family in range(3):
        ix=[i for i,r in enumerate(regional) if r['family']==family]
        m=reg[ix,:,0];contrast=m[::2,0]-m[1::2,0];effect=m-m[:,:1]
        sign=torch.tensor([-1 if regional[i]['cue']=='British' else 1 for i in ix],dtype=torch.float64)
        reduction=effect[1::2,1]-effect[::2,1];directed=effect[:,6]*sign
        coverage=float(reduction.mean()/contrast.mean());transfer=float(directed.mean()/contrast.mean())
        ratios=[float((reg[ix,a,1]-reg[ix,0,1]).abs().mean()/effect[:,a].abs().mean().clamp_min(1e-30)) for a in [1,6]]
        selfdiff=reg[ix,7]-reg[ix,0];selfnorm=float(reg[ix,0].norm())
        selferror=float(selfdiff.norm()/max(selfnorm,1e-30))
        selfpass=selferror<=1e-4 if selfnorm>1e-8 else float(selfdiff.abs().max())<=1e-6
        arm_coverage=[float((effect[1::2,a]-effect[::2,a]).mean()/contrast.mean()) for a in range(8)]
        interaction=effect[:,3]-effect[:,1]-effect[:,2]
        cells.append(dict(family=family,native_mean_contrast=float(contrast.mean()),
            native_positive_pairs=int((contrast>0).sum()),
            native_capability=float(contrast.mean())>=.2 and int((contrast>0).sum())>=10,
            coverage=coverage,positive_removal_pairs=int((reduction>0).sum()),
            donor_transfer=transfer,positive_donor_rows=int((directed>0).sum()),
            unrelated_ratios=ratios,self_error=selferror,self_pass=selfpass,
            passed=coverage>=.5 and int((reduction>0).sum())>=10 and transfer>=.5 and int((directed>0).sum())>=20 and max(ratios)<=.5,
            arm_coverage=arm_coverage,mean_absolute_target_effects=effect.abs().mean(0).tolist(),
            mean_absolute_control_effects=(reg[ix,:,1]-reg[ix,:1,1]).abs().mean(0).tolist(),
            nonlinear_combination_residual_norm=float(interaction.norm())))
    ncells=[]
    for family in range(2):
        ix=[i for i,r in enumerate(natural) if r['family']==family]
        changes=ce[ix]-ce[ix,:1];delta=changes[:,1]
        ncells.append(dict(family=family,native_mean_ce=float(ce[ix,0].mean()),
            native_mean_margin=float(margin[ix,0].mean()),native_positive=int((margin[ix,0]>0).sum()),
            capability=float(ce[ix,0].mean())<=5 and float(margin[ix,0].mean())>=.2 and int((margin[ix,0]>0).sum())>=12,
            meanabs_change=float(delta.abs().mean()),maxabs_change=float(delta.abs().max()),
            passed=float(delta.abs().mean())<=.02 and float(delta.abs().max())<=.1,
            mean_signed_ce_changes=changes.mean(0).tolist(),mean_absolute_ce_changes=changes.abs().mean(0).tolist(),
            full_head_positive_changes=int((changes[:,3]>0).sum())))
    old=torch.load(P/'SCALAR_JOINT_KEY_GRADES_CONFIRMATION_V1_ARTIFACT.pt',weights_only=True)
    rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
    replays=dict(native_regional=rel(reg[:,0],old['regional'][:,0]),
                 native_ce=rel(ce[:,0],old['newline_ce'][:,0]),
                 recompose_regional=rel(reg[:,4],reg[:,0]),recompose_ce=rel(ce[:,4],ce[:,0]),
                 scalar_removal=rel(reg[:,5],old['regional'][:,2]))
    instrument=max(checks)<=1e-5 and all(c['self_pass'] for c in cells)
    capability=all(c['native_capability'] for c in cells) and all(c['capability'] for c in ncells)
    A=instrument and capability and max(replays.values())<=1e-5
    torch.save(dict(regional=reg,newline_ce=ce,newline_margin=margin),artifact)
    result={'pred_a':A,'pred_b':A and all(c['passed'] for c in cells),
            'pred_c':A and all(c['passed'] for c in ncells),'pred_d':max(replays.values())<=1e-5,
            'instrument':instrument,'capability':capability,'max_live_recomposition_error':max(checks),
            'replays':replays,'regional':cells,'newline':ncells,'component_sizes':sizes,
            'arms_regional':ARMS,'arms_natural':ARMS[:6],'body_forwards':count,
            'seconds':time.perf_counter()-tic,'artifact_sha256':digest(artifact),'source_shas':binding,
            'scope':'Frozen full128value even-key component, all current/first values and1152writer. Existing72regional/32natural developmentpanel. Fullnativeprefix/suffix retained; no new OOD/fit/fullmodelcompression claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_shas','component_sizes']}),flush=True)


if __name__=='__main__':main()
