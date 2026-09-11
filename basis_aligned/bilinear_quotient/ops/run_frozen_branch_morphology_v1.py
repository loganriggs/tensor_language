#!/usr/bin/env python3
# BQGATE: 20 body forwards, 144 sequences, lengths8-16; frozen morphology rows, no fitting.
"""Test frozen shared-bank subject-verb and count-noun morphology transfer.
A bound rows/ports/finite/physical replay<=1e-5. B native capability per family/direction.
C branch0 direct donor-answer write contrast positive>=75% in A1/A2.
D both-branch margin transfer>=10% of mean native gap and>=.05 nats in each A1/A2 direction.
E mean absolute CE change<=.05 on answer-preserving P and unrelated C.
No new behavior, factor fitting, or fresh OOD claim; mechanical token relations proposed this screen; historical C controls retained.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3]
P=ROOT/'basis_aligned/polynomial_causal';STEM='FROZEN_BRANCH_MORPHOLOGY_V1'
sys.path[:0]=[str(RUNNER.parent),str(ROOT)]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(8<<20),b''):h.update(c)
    return h.hexdigest()


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())
    assert all(digest(path)==sha for path,sha in binding['files'].items())
    authority=json.loads((P/(STEM+'_ROWS.json')).read_text())
    assert authority['authority_sha256']=='4288abe4a68bd804a274db4dc1399000d4816e405d304191c6676d12472fe767'
    rows=authority['rows'];assert len(rows)==64 and len({r['row_id'] for r in rows})==64
    sequences=[];buckets={}
    for i,row in enumerate(rows):
        assert all(row['construction_checks'].values())
        assert row['family'] in ('A1','A2','P','C') and row['group_number']==i//4
        for side in ('base','donor'):
            ids=row[side+'_ids'];index=len(sequences)
            assert row[side+'_prediction_position']==len(ids)-1
            sequences.append(ids);buckets.setdefault(len(ids),[]).append(index)
    expected_forwards=sum((len(v)+7)//8 for v in buckets.values())+2
    assert expected_forwards==20 and len(sequences)==128
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=20,sequences=144,length_range=[min(buckets),max(buckets)],fitting=False)));return
    output=P/(STEM+'_RESULT.json');cache_path=P/(STEM+'_ENDPOINTS.pt')
    assert not output.exists() and not cache_path.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17]
    saved=torch.load(P/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt',weights_only=True,map_location='cpu')
    node=saved['nodes'][1];reader=node['reader'].double().cuda();partners=node['partners'].double().cuda()
    writers=torch.linalg.solve_triangular(saved['output_whitener'].double(),node['writers'].double(),upper=True).cuda()
    def delta(x,branch):
        xx=x.double();return (((xx@reader)*(xx@partners[:,branch]))[...,None]*writers[:,branch]).float()
    def logits(h):return 30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
    def prefix(tokens):
        x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
        for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
        x=last.lambdas[0]*x+last.lambdas[1]*x0
        attention,v1=last.attn(F.rms_norm(x,(1152,)),v1)
        pre=x+attention;xin=F.rms_norm(pre,(1152,))
        return xin,pre,last.mlp(xin)
    counts=[0,0]
    def count(module,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=20 and args[0].shape[1]<=16
    handle=model.transformer.h[0].attn.register_forward_pre_hook(count)
    ports={k:torch.empty(128,1152) for k in ('input','pre','native_output')};controls=[]
    try:
        first=True
        for length,indices in sorted(buckets.items()):
            for off in range(0,len(indices),8):
                selected=indices[off:off+8];tokens=torch.tensor([sequences[i] for i in selected],device='cuda')
                xin,pre,native=prefix(tokens)
                for name,value in zip(ports,(xin,pre,native)):ports[name][selected]=value[:,-1].cpu()
                if first:
                    assert len(selected)==8
                    for branch in (None,0):
                        captured={}
                        def head_hook(module,args,value):captured['raw']=value[:,-1].detach().clone()
                        def mlp_hook(module,args,value):
                            captured['input_error']=float((args[0]-xin).norm()/xin.norm())
                            return value if branch is None else value-delta(args[0],branch)
                        hooks=[model.lm_head.register_forward_hook(head_hook),last.mlp.register_forward_hook(mlp_hook)]
                        try:model(tokens,tokens)
                        finally:
                            for hook in hooks:hook.remove()
                        manual=logits(pre[:,-1]+(native[:,-1] if branch is None else native[:,-1]-delta(xin[:,-1],branch)))
                        physical=30*torch.tanh(captured['raw']/30)
                        controls.append(dict(branch=branch,input_error=captured['input_error'],logit_error=float((manual-physical).norm()/physical.norm())))
                    first=False
    finally:handle.remove()
    assert counts==[20,144]
    x=ports['input'].double().cuda();amp=(x@reader)[:,None]*(x@partners)
    h=(ports['pre']+ports['native_output']).cuda()
    records=[];native_logits=[]
    for i,row in enumerate(rows):
        base,donor=2*i,2*i+1
        base_logits=logits(h[base:base+1]);donor_logits=logits(h[donor:donor+1])
        ba,bf,da,df=[int(row[key]) for key in ('base_answer_id','base_foil_id','donor_answer_id','donor_foil_id')]
        base_ce=F.cross_entropy(base_logits.double(),torch.tensor([ba],device='cuda'))
        donor_margin=float(donor_logits[0,da]-donor_logits[0,df])
        base_margin=float(base_logits[0,da]-base_logits[0,df]);gap=donor_margin-base_margin
        difference=amp[donor]-amp[base]
        writes=[difference[j]*writers[:,j] for j in range(2)]
        shifts=[];cechanges=[]
        for write in (writes[0],writes[1],writes[0]+writes[1]):
            alternative=logits(h[base:base+1]+write.float()[None])
            shifts.append(float(alternative[0,da]-alternative[0,df])-base_margin)
            cechanges.append(float(F.cross_entropy(alternative.double(),torch.tensor([ba],device='cuda'))-base_ce))
        records.append(dict(row_id=row['row_id'],family=row['family'],group=row['group_number'],
            direction=row['direction'],
            direct_branch0_contrast=float((model.lm_head.weight[da].double()-model.lm_head.weight[df].double())@writes[0]),
            native_mlp_only_effect=float((logits((ports['pre'][base]+ports['native_output'][donor]).cuda()[None])[0,da]-logits((ports['pre'][base]+ports['native_output'][donor]).cuda()[None])[0,df]))-base_margin,
            native_base_correct=bool(base_logits[0,ba]>base_logits[0,bf]),
            native_donor_correct=bool(donor_logits[0,da]>donor_logits[0,df]),
            native_gap=gap,margin_shifts=shifts,ce_changes=cechanges,
            base_amplitudes=amp[base].cpu().tolist(),donor_amplitudes=amp[donor].cpu().tolist()))
    cells=[];activation={};collateral={}
    for family in ('A1','A2','P','C'):
        local=[r for r in records if r['family']==family]
        for direction in ('base_to_suffix','suffix_to_base'):
            subset=[r for r in local if r['direction']==direction]
            base_capability=sum(r['native_base_correct'] for r in subset)/len(subset)
            donor_capability=sum(r['native_donor_correct'] for r in subset)/len(subset)
            capability=min(base_capability,donor_capability)
            gap=sum(r['native_gap'] for r in subset)/len(subset)
            shift=sum(r['margin_shifts'][2] for r in subset)/len(subset)
            cells.append(dict(family=family,direction=direction,native_capability=capability,
                native_base_capability=base_capability,native_donor_capability=donor_capability,
                capability_held=capability>=(.75 if family=='C' else .85),mean_native_gap=gap,
                mean_bank_margin_shift=shift,recovery=shift/gap if gap!=0 else None,
                transfer_held=gap>0 and shift>=.05 and shift/gap>=.1))
        if family in ('A1','A2'):
            differences=torch.tensor([r['direct_branch0_contrast'] for r in local],dtype=torch.float64)
            fraction=float((differences>0).double().mean())
            activation[family]=dict(mean_direct_contrast=float(differences.mean()),positive_fraction=fraction,
                held=fraction>=.75)
        else:collateral[family]=sum(abs(r['ce_changes'][2]) for r in local)/len(local)
    a=all(c['input_error']<=1e-6 and c['logit_error']<=1e-5 for c in controls) and all(torch.isfinite(amp).flatten())
    a=bool(a) and all(torch.isfinite(torch.tensor(r['margin_shifts'])).all() for r in records)
    b=a and all(c['capability_held'] for c in cells)
    torch.save(dict(ports=ports,amplitudes=amp.cpu(),rows_sha256=digest(P/(STEM+'_ROWS.json')),
        scope='Frozen factor validation cache, no fitting.'),cache_path)
    result={'pred_a':a,'pred_b':b,'pred_c':b and all(v['held'] for v in activation.values()),
        'pred_d':b and all(c['transfer_held'] for c in cells if c['family'] in ('A1','A2')),
        'pred_e':b and all(v<=.05 for v in collateral.values()),'capability_transfer_cells':cells,
        'activation':activation,'control_mean_abs_ce_change':collateral,'records':records,'physical_controls':controls,
        'price':{'body_forwards':counts[0],'sequences':counts[1],'length_range':[8,16],'native_background_retained':True},
        'seconds':time.perf_counter()-started,'cache_sha256':digest(cache_path),'binding_sha256':digest(P/(STEM+'_BINDING.json')),
        'scope':'Spelling-guided subject agreement and count-noun validation of frozen weight branches. Historical V6 unrelated controls reused. No new grammatical behavior, fitting, fresh OOD, isolated extraction or semantic circuit promotion.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
