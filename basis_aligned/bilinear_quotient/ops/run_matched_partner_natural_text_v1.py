#!/usr/bin/env python3
# BQGATE:20bodyforwards,160sequences,128tokens;144naturalendpoints;600sec.
"""pred_a native replay/counts; pred_b ing damage/ing-base>=.002 both corpora;
pred_c other meanabsCE<=.02; pred_d target top20>=50%; pred_e joint CE error<=.02.
Frozen packed branches; no fitting, capability filtering or grammar labels.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from packed_quadratic_branch_v1 import execute
STEM='MATCHED_PARTNER_NATURAL_TEXT_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    provenance=json.loads((P/(STEM+'_ROWS.json')).read_text());assert digest(P/(STEM+'_ROWS.pt'))==provenance['artifact_sha']
    panel=torch.load(P/(STEM+'_ROWS.pt'),weights_only=True);rows=panel['rows'];metadata=panel['metadata'];assert rows.shape==(144,129) and len({m['document_unit'] for m in metadata})==48
    for source in provenance['sources']:
        raw=torch.load(ROOT/source['path'],weights_only=True,mmap=True);raw=raw['rows'] if isinstance(raw,dict) else raw
        for i,m in enumerate(metadata):
            if m['corpus']==source['corpus']:assert torch.equal(rows[i],raw[m['document'],m['target_position']-128:m['target_position']+1])
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=20,sequences=160,input_tokens=128,fitting=False)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PORTS.pt');assert not out.exists() and not ap.exists();signal.alarm(600)
    tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];counts=[0,0];captured={}
    def capture_input(module,args):captured['x16']=args[0].detach()
    def count(module,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=20 and args[0].shape[1]==128
    handles=[model.transformer.h[16].mlp.register_forward_pre_hook(capture_input),model.transformer.h[0].attn.register_forward_pre_hook(count)]
    program=torch.load(P/'MATCHED_PARTNER_EXACT_INPUT_FOLD_V1_PROGRAM.pt',weights_only=True)
    original=torch.load(P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt',weights_only=True)
    l,r=[getattr(model.transformer.h[16].mlp,n).weight.double() for n in ('Left','Right')]
    def logits(h):return 30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
    scores=[];ranks=[];ports={k:[] for k in ('input16','pre','native_output')};coefficients=[];checks=[];write_checks=[]
    try:
        for offset in range(0,144,8):
            tokens=rows[offset:offset+8,:128].cuda();targets=rows[offset:offset+8,-1].cuda()
            x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
            for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
            residual=last.lambdas[0]*x+last.lambdas[1]*x0;attn,v1=last.attn(F.rms_norm(residual,(1152,)),v1)
            pre=residual+attn;xin=F.rms_norm(pre,(1152,));native=last.mlp(xin);xx=captured['x16'][:,-1].double();pp=pre[:,-1];nn=native[:,-1]
            den=pp.double().square().mean(-1)+torch.finfo(torch.float32).eps;writes=execute(program,xx,den)
            hidden=(xx@l.T)*(xx@r.T);values=hidden@original['compiled_producer_readers'][:,[0,3,8]].cuda();alpha=values[:,0,None]*values[:,1:]/den[:,None]
            for j,branch in enumerate((3,8)):
                ref=alpha[:,j,None]*original['output_writers'][:,branch].cuda()[None,:];write_checks.append(float((writes[branch]-ref).norm()/ref.norm()))
            coefficients.append(alpha.cpu());state=pp+nn;base=logits(state);alternatives=[logits(state-writes[3].float()),logits(state-writes[8].float()),logits(state-(writes[3]+writes[8]).float())]
            if offset==0:
                for edited in (False,True):
                    captured_control={}
                    def head_hook(module,args,value):captured_control['raw']=value[:,-1].detach()
                    def mlp_hook(module,args,value):
                        captured_control['input_error']=float((args[0]-xin).norm()/xin.norm())
                        if not edited:return value
                        value=value.clone();value[:,-1]-=writes[8].float();return value
                    extra=[model.lm_head.register_forward_hook(head_hook),last.mlp.register_forward_hook(mlp_hook)]
                    try:model(tokens,rows[offset:offset+8,1:].cuda())
                    finally:
                        for hook in extra:hook.remove()
                    physical=30*torch.tanh(captured_control['raw']/30);manual=alternatives[1] if edited else base
                    checks.append(dict(edited=edited,input_error=captured_control['input_error'],logit_relative_error=float((manual-physical).norm()/physical.norm())))
            ce=F.cross_entropy(base.double(),targets,reduction='none');scores.append(torch.stack([ce]+[F.cross_entropy(a.double(),targets,reduction='none')-ce for a in alternatives],-1).cpu())
            ranks.append((1+(base>base.gather(1,targets[:,None])).sum(1)).cpu())
            for key,value in [('input16',xx),('pre',pp),('native_output',nn)]:ports[key].append(value.float().cpu())
    finally:
        for hook in handles:hook.remove()
    assert counts==[20,160];scores=torch.cat(scores);ranks=torch.cat(ranks);alpha=torch.cat(coefficients);cells=[]
    for corpus in ('fineweb','pile_ood'):
        for group in ('ing','base','other'):
            ids=torch.tensor([i for i,m in enumerate(metadata) if m['corpus']==corpus and m['group']==group]);e=scores[ids,1:]
            cells.append(dict(corpus=corpus,group=group,count=len(ids),native_top20=float((ranks[ids]<=20).double().mean()),native_ce=float(scores[ids,0].mean()),
                              mean_ce_effect=e.mean(0).tolist(),meanabs_ce_effect=e.abs().mean(0).tolist(),joint_additivity_meanabs=float((e[:,2]-e[:,0]-e[:,1]).abs().mean()),
                              branch8_positive_coefficient_fraction=float((alpha[ids,1]>0).double().mean()),branch8_coefficient_mean=float(alpha[ids,1].mean())))
    by={(c['corpus'],c['group']):c for c in cells}
    result={'pred_a':bool(torch.isfinite(scores).all()) and max(write_checks)<=1e-5 and all(c['input_error']<=1e-6 and c['logit_relative_error']<=1e-5 for c in checks),
            'pred_b':all(by[(c,'ing')]['mean_ce_effect'][1]>=.002 and by[(c,'ing')]['mean_ce_effect'][1]-by[(c,'base')]['mean_ce_effect'][1]>=.002 for c in ('fineweb','pile_ood')),
            'pred_c':all(by[(c,'other')]['meanabs_ce_effect'][1]<=.02 for c in ('fineweb','pile_ood')),
            'pred_d':all(by[(c,g)]['native_top20']>=.5 for c in ('fineweb','pile_ood') for g in ('ing','base')),
            'pred_e':all(c['joint_additivity_meanabs']<=.02 for c in cells)}
    torch.save(dict(ports={k:torch.cat(v) for k,v in ports.items()},scores=scores,ranks=ranks,alpha=alpha,metadata=metadata,rows_sha=digest(P/(STEM+'_ROWS.pt'))),ap)
    result.update(cells=cells,checks=checks,max_write_replay=max(write_checks),counts=counts,artifact_sha=digest(ap),source_shas=binding,execution_seconds=time.perf_counter()-tic,
                  scope='Frozen natural spelling-group screen, FineWeb and Pile separately; no data fitting, not POS semantics or full-model OOD replacement.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)

if __name__=='__main__':main()
