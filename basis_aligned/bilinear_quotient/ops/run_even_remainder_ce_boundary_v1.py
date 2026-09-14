#!/usr/bin/env python3
# BQGATE:128bodyforwards;32prefixes<=247tokens;120seconds;no fitting.
"""pred_a all four old/new CE anchors<=1e-5relative.
pred_b CE interaction decomposition<=1e-10relative.
pred_c downstream raw-score contribution norm<=.2total interaction BOTHhalves.
Null: composition failure already occurs in raw logits, not merely CE/softcap.
Price128bodyforwards,32naturalprefixes,120seconds,full50304rawlogits retained.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from key_span_reuse_v1 import KeySpanParent
from even_key_value_native_backend_v1 import FullValueComponents
STEM='EVEN_REMAINDER_CE_BOUNDARY_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())['natural']
    assert len(rows)==32 and max(len(r['ids']) for r in rows)<=247
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('128bodyforwards:32natural4arms native/scalar/remainder/even; record raw50304logits for CE decomposition');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(120);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval()
    components=FullValueComponents(torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True),'cuda')
    scalar=KeySpanParent({k:v.cuda() for k,v in torch.load(P/'KEY_SPAN_REUSE_V1_PROGRAM.pt',weights_only=True).items()})
    writer=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True)['writers'][1].cuda()
    context={};count=0
    def hook(module,args,output):
        arm=context['arm']
        if arm==0:return output
        selected=scalar.scalar(args[0],context['tokens'],1)[...,None]*writer
        if arm==1:delta=selected
        else:
            even,_=components(args[0],args[1].reshape(*args[0].shape[:2],9,128)[:,:,8])
            delta=even-selected if arm==2 else even
        return output[0]-delta.to(output[0].dtype),output[1]
    handle=model.transformer.h[9].attn.register_forward_hook(hook)
    raw=torch.empty(32,4,50304,dtype=torch.float32);ce=torch.empty(32,4,dtype=torch.float64)
    try:
        for i,row in enumerate(rows):
            for arm in range(4):
                ids=torch.tensor([row['ids']],device='cuda');context.update(tokens=ids,arm=arm)
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
                for block in model.transformer.h:x,v1=block(x,v1,x0)
                scores=model.lm_head(F.rms_norm(x[:,-1],(1152,)))[0]
                raw[i,arm]=scores.cpu();ce[i,arm]=-(30*torch.tanh(scores/30)).log_softmax(-1)[198].cpu();count+=1
    finally:handle.remove()
    assert count==128
    old=torch.load(P/'EVEN_KEY_FULL_VALUE_NATIVE_V1_ARTIFACT.pt',weights_only=True)['newline_ce']
    rem=torch.load(P/'EVEN_KEY_REMAINDER_NATIVE_V1_ARTIFACT.pt',weights_only=True)['newline_ce']
    references=torch.stack([old[:,0],old[:,5],rem[:,1],old[:,1]],1)
    anchor_errors=[float((ce[:,a]-references[:,a]).norm()/references[:,a].norm()) for a in range(4)]
    # Analyze continuous FP64 cap/loss evaluated at native rounded FP32 raw logits.
    r=raw.double();cap=lambda x:30*torch.tanh(x/30)
    loss=lambda x:torch.logsumexp(x,-1)-x[...,198]
    z=cap(r);losses=loss(z)
    raw_add=r[:,1]+r[:,2]-r[:,0];logit_add=z[:,1]+z[:,2]-z[:,0]
    interaction=losses[:,3]-losses[:,1]-losses[:,2]+losses[:,0]
    downstream=losses[:,3]-loss(cap(raw_add))
    softcap=loss(cap(raw_add))-loss(logit_add)
    ce_curve=loss(logit_add)-losses[:,1]-losses[:,2]+losses[:,0]
    terms=torch.stack([downstream,softcap,ce_curve],1)
    replay=float((terms.sum(1)-interaction).norm()/interaction.norm().clamp_min(1e-30))
    cells=[]
    for family in range(2):
        ix=[i for i,row in enumerate(rows) if row['family']==family]
        target=interaction[ix];t=terms[ix];norm=target.norm()
        ratios=(t.norm(dim=0)/norm).tolist();fraction=((t*target[:,None]).sum(0)/target.square().sum()).tolist()
        cells.append(dict(family=family,interaction_norm=float(norm),term_norm_ratios=ratios,
                           signed_projection_fractions=fraction,metric_dominance_pass=ratios[0]<=.2,
                           maxabs_interaction=float(target.abs().max())))
    torch.save(dict(raw_logits=raw,native_ce=ce,fp64_analysis_losses=losses,
                    interaction=interaction,terms=terms),artifact)
    result={'pred_a':max(anchor_errors)<=1e-5,'pred_b':replay<=1e-10,
            'pred_c':all(c['metric_dominance_pass'] for c in cells),
            'anchor_errors':anchor_errors,'decomposition_relative_error':replay,
            'families':cells,'term_order':['downstream_raw_and_final_RMS','softcap_curvature','logsumexp_curvature'],
            'row_terms':terms.tolist(),'row_interactions':interaction.tolist(),
            'arms':['native','remove_scalar','remove_remainder','remove_even'],
            'body_forwards':count,'seconds':time.perf_counter()-start,
            'artifact_bytes':artifact.stat().st_size,'artifact_sha256':digest(artifact),'source_shas':binding,
            'scope':'Post hoc discriminator of preserved failedCEcomposition; FP64 analytic losses of frozen nativeFP32rawlogits. Term split is ordered, signed, not nonnegative attribution shares. Same32developmentprefixes; no fitting/correction/newOOD/wholemodel claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_shas','row_terms','row_interactions']}),flush=True)


if __name__=='__main__':main()
