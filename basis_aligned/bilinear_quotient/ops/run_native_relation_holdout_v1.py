#!/usr/bin/env python3
# BQGATE: 20 body forwards, 144 sequences, lengths9-16; frozen construction holdout rows, no fitting.
"""Frozen native suffix readout program on new lexemes/constructions.
A bound rows/finite and physical replay<=1e-5; B native capability>=.85 tasks/P,>=.75C eachside/direction.
C approximate/exactspan effect relative RMS<=.25 and sign agreement>=.9 eachtask/direction.
D exact AND approximate swap>=10%nativegap and>=.05 eachtask/direction.
E approximate swap meanabs CE<=.05 P/C; F absolute replacement meanabs CE<=.05 eachfamily.
No fitting, outcome filtering, training-disjoint or full circuit claim; native complement remains.
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
P=ROOT/'basis_aligned/polynomial_causal';STEM='NATIVE_RELATION_HOLDOUT_V1'
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
    assert authority['authority_sha256']=='ac5326448f053c330662e87b41053a93a284eaf1b312513de24e598ad4b7e405'
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
    saved=torch.load(P/'NATIVE_TOKEN_RELATION_FOLD_V1.pt',weights_only=True,map_location='cpu')
    assert saved['labels'][:3]==['add_s','add_es','y_to_ies']
    readouts=saved['physical_readouts'][:3].double().cuda()
    writers=torch.linalg.solve(readouts@readouts.T,readouts).T
    projection_error=float((readouts@writers-torch.eye(3,dtype=torch.float64,device='cuda')).abs().max())
    assert projection_error<=1e-9
    compact=[]
    for j,c in enumerate(saved['compact'][:3]):
        trace=(c['top16_square_coefficients']*c['top16_square_readers'].square().sum(1)).sum()
        compact.append((c['top16_square_readers'].cuda(),c['top16_square_coefficients'].cuda(),
                        (c['radial']-trace/1152).cuda(),saved['folded_bias'][j].cuda()))
    def predict(x):
        xx=x.double()
        return torch.stack([(xx@v.T).square()@coef+radial*xx.square().sum(-1)+bias
                            for v,coef,radial,bias in compact],dim=-1)
    def delta(x,branch):return (predict(x)@writers.T).float()
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
    x=ports['input'].double().cuda();predicted=predict(x)
    phi=ports['native_output'].double().cuda()@readouts.T
    h=(ports['pre']+ports['native_output']).cuda();records=[]
    for i,row in enumerate(rows):
        base,donor=2*i,2*i+1
        base_logits=logits(h[base:base+1]);donor_logits=logits(h[donor:donor+1])
        ba,bf,da,df=[int(row[key]) for key in ('base_answer_id','base_foil_id','donor_answer_id','donor_foil_id')]
        base_ce=F.cross_entropy(base_logits.double(),torch.tensor([ba],device='cuda'))
        donor_margin=float(donor_logits[0,da]-donor_logits[0,df]);base_margin=float(base_logits[0,da]-base_logits[0,df])
        writes=[writers@(phi[donor]-phi[base]),writers@(predicted[donor]-predicted[base]),writers@(predicted[base]-phi[base])]
        effects=[];cechanges=[]
        for write in writes:
            alternative=logits(h[base:base+1]+write.float()[None])
            effects.append(float(alternative[0,da]-alternative[0,df])-base_margin)
            cechanges.append(float(F.cross_entropy(alternative.double(),torch.tensor([ba],device='cuda'))-base_ce))
        records.append(dict(row_id=row['row_id'],family=row['family'],group=row['group_number'],
            construction=row['construction'],direction=row['direction'],
            native_base_correct=bool(base_logits[0,ba]>base_logits[0,bf]),
            native_donor_correct=bool(donor_logits[0,da]>donor_logits[0,df]),
            native_gap=donor_margin-base_margin,effects=effects,ce_changes=cechanges))
    cells=[];collateral={};replacement={}
    for family in ('A1','A2','P','C'):
        local=[r for r in records if r['family']==family]
        collateral[family]=sum(abs(r['ce_changes'][1]) for r in local)/len(local)
        replacement[family]=sum(abs(r['ce_changes'][2]) for r in local)/len(local)
        for direction in ('base_to_suffix','suffix_to_base'):
            subset=[r for r in local if r['direction']==direction]
            bc=sum(r['native_base_correct'] for r in subset)/len(subset);dc=sum(r['native_donor_correct'] for r in subset)/len(subset)
            gap=sum(r['native_gap'] for r in subset)/len(subset)
            exact=torch.tensor([r['effects'][0] for r in subset],dtype=torch.float64)
            approx=torch.tensor([r['effects'][1] for r in subset],dtype=torch.float64)
            error=float((approx-exact).norm()/exact.norm());sign=float((approx.sign()==exact.sign()).double().mean())
            cells.append(dict(family=family,direction=direction,n=len(subset),native_base_capability=bc,native_donor_capability=dc,
                capability_held=min(bc,dc)>=(.75 if family=='C' else .85),native_gap=gap,
                exact_mean=float(exact.mean()),approx_mean=float(approx.mean()),relative_effect_error=error,sign_agreement=sign,
                exact_recovery=float(exact.mean())/gap if gap!=0 else None,approx_recovery=float(approx.mean())/gap if gap!=0 else None,
                approximation_held=error<=.25 and sign>=.9,
                transfer_held=gap>0 and min(float(exact.mean()),float(approx.mean()))>=max(.05,.1*gap)))
    a=all(c['input_error']<=1e-6 and c['logit_error']<=1e-5 for c in controls) and bool(torch.isfinite(predicted).all())
    a=bool(a) and all(torch.isfinite(torch.tensor(r['effects'])).all() for r in records)
    b=a and all(c['capability_held'] for c in cells)
    torch.save(dict(ports=ports,readout_values=phi.cpu(),predicted_readout_values=predicted.cpu(),
        rows_sha256=digest(P/(STEM+'_ROWS.json')),scope='Frozen holdout validation cache, no fitting.'),cache_path)
    result={'pred_a':a,'pred_b':b,'pred_c':b and all(c['approximation_held'] for c in cells if c['family'] in ('A1','A2')),
        'pred_d':b and all(c['transfer_held'] for c in cells if c['family'] in ('A1','A2')),
        'pred_e':b and all(collateral[f]<=.05 for f in ('P','C')),'pred_f':b and all(v<=.05 for v in replacement.values()),
        'cells':cells,'swap_mean_abs_ce':collateral,'replacement_mean_abs_ce':replacement,'records':records,
        'physical_controls':controls,'projection_error':projection_error,
        'price':{'body_forwards':counts[0],'sequences':counts[1],'length_range':[9,16],'square_products':48,'native_background_retained':True},
        'seconds':time.perf_counter()-started,'cache_sha256':digest(cache_path),'binding_sha256':digest(P/(STEM+'_BINDING.json')),
        'effect_order':['exact_span_swap','approx_span_swap','absolute_replacement'],
        'scope':'Frozen trace-corrected native suffix program, new answer lexemes and constructions; old unrelated controls retained. '
        'No fit/outcome filtering, corpus OOD/pretraining disjointness or ordinary replacement adoption implied.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
