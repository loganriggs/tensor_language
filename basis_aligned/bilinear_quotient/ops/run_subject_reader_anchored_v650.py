#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_joint_replay pred_b_conditional_fidelity pred_c_native_fidelity
"""Width8 reader-anchored basis control: two fixed readout plus six response features.

128 original calibration rows, freeze SVD bases, evaluate48 opened position rows.
Fixed random8 matched baseline.12 prefix+16 suffix calls, zero model fits/backwards.
Full tensor contractions and exact RMS; native background ports charged.
"""
import os
import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_joint_chain_v650_result.json'
PREDICTIONS=dict(pred_a_joint_replay='projected local response relative and absolute replay <=1e-4',
 pred_b_conditional_fidelity='calibration SVD8 conditional effect error <=.05 every evaluation cell',
 pred_c_native_fidelity='calibration SVD8 full native effect error <=.10 every evaluation cell')


def main():
    plan=dict(prefix_calls=12,suffix_calls=16,calibration_rows=128,evaluation_rows=48,width=8,
              additional_endpoint_chain_batches=6,basis_selection='two fixed answer readers plus six orthogonal calibration response SVD vectors',controls=['dense','random_seed647'],
              model_fits=0,backwards=0,execution_policy='managed_queue_only',predictions=PREDICTIONS)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import circuit_fast_screen_producer as producer
    import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as original
    import subject_number_sparse_graph_token_extraction_v1 as graph
    from disk_guard import guard_write
    sys.path.insert(0,str(POLY))
    from projected_bilinear_response import compile_response,evaluate
    from conditional_mlp_chain import execute
    if OUT.exists():raise FileExistsError(OUT)
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    decoder=json.loads((POLY/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
    axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64)
    unit=axis/axis.norm();threshold=float(decoder['threshold'])/float(axis.norm())
    eps=torch.finfo(torch.float32).eps
    U=model.lm_head.weight[[318,389]].double()
    counts=dict(prefix_calls=0,suffix_calls=0)
    def suffix(raw,x0,first,pos,frozen=None):
        counts['suffix_calls']+=1;x=raw;batch=torch.arange(len(x),device=x.device)
        result=dict(states=[],backgrounds=[],mlp_writes=[],attention={})
        for layer in range(11,18):
            b=model.transformer.h[layer]
            if layer>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
            if frozen is not None and layer>=12:a=frozen[layer]
            else:a,first=b.attn(F.rms_norm(x,(x.shape[-1],)),first)
            x=x+a
            m=b.mlp(F.rms_norm(x,(x.shape[-1],)))
            if layer>=12:
                result['backgrounds'].append(x[batch,pos].double())
                result['mlp_writes'].append((m[batch,pos]-b.mlp.Down_bias).double())
            x=x+m;result['states'].append(x[batch,pos].double())
            result['attention'][layer]=a
        return result
    def margin(x,direction):
        logits=30*torch.tanh(F.rms_norm(x,(x.shape[-1],),eps=eps)@U.T/30)
        return (logits[:,0]-logits[:,1])*direction
    def capture(entries,with_truth):
        ids=torch.tensor([r['token_ids'] for r in entries],device='cuda')
        editpos=torch.tensor([r['subject_position'] for r in entries],device='cuda')
        readpos=torch.tensor([r['readout_position'] for r in entries],device='cuda')
        answers=torch.tensor([r['answer_ids'] for r in entries],device='cuda')
        batch=torch.arange(len(entries),device='cuda');direction=torch.where(answers[:,0]==318,1.,-1.)
        initial=F.rms_norm(model.transformer.wte(ids),(model.config.n_embd,)).float()
        subject=initial[batch,editpos].double();orth=subject-(subject@unit)[:,None]*unit
        scale=((subject.square().sum(-1)-threshold**2)/orth.square().sum(-1)).sqrt()
        removed=initial.clone();removed[batch,editpos]=(threshold*unit+scale[:,None]*orth).float()
        raw,x0,first,pb,_=graph._capture(model,initial,torch,F)
        _,_,_,pr,_=graph._capture(model,removed,torch,F);counts['prefix_calls']+=2
        edited=raw.clone();edited[batch,editpos]+=sum((r-b)[batch,editpos] for b,r in zip(pb,pr)).float()
        base=suffix(raw,x0,first,readpos);frozen=suffix(edited,x0,first,readpos,base['attention'])
        bm=margin(base['states'][-1],direction)
        conditional=bm-margin(frozen['states'][-1],direction)
        truth=None
        if with_truth:
            true=graph._suffix_margin(model,edited,x0,first,readpos,answers,torch,F);counts['suffix_calls']+=1
            truth=bm-true
        return dict(base=base,frozen=frozen,direction=direction,conditional=conditional,truth=truth,
                    families=[r['family'] for r in entries],initial=x0[batch,readpos].double(),
                    attention_ports=[base['attention'][l][batch,readpos].double() for l in range(12,18)])
    calrows=[]
    for r in original.build_rows():
        ans=r['native_answer_id']
        calrows.append(dict(token_ids=r['token_ids'],subject_position=r['subject_position'],
             readout_position=len(r['token_ids'])-1,answer_ids=[ans,389 if ans==318 else 318],family='calibration'))
    calibration=[capture(calrows[:64],False),capture(calrows[64:],False)]
    bases=[];singular=[]
    anchors=torch.linalg.qr(U.T.cpu()).Q
    for l in range(7):
        deltas=torch.cat([c['frozen']['states'][l]-c['base']['states'][l] for c in calibration])
        residual=deltas.cpu()-(deltas.cpu()@anchors)@anchors.T
        _,s,Vh=torch.linalg.svd(residual,full_matrices=False)
        frame=torch.linalg.qr(torch.cat([anchors,Vh[:6].T],dim=1)).Q.cuda()
        assert float((U-(U@frame)@frame.T).norm()/U.norm())<1e-10
        bases.append(frame);singular.append(s.tolist())
    generator=torch.Generator(device='cpu').manual_seed(647)
    random=[torch.linalg.qr(torch.randn(1152,8,dtype=torch.float64,generator=generator)).Q.cuda() for _ in range(7)]
    programs={};factors=[]
    for b in model.transformer.h[12:18]:
        factors.append((b.mlp.Left.weight.double(),b.mlp.Right.weight.double(),b.mlp.Down.weight.double()))
    for name,frames in [('calibration',bases),('random',random)]:
        programs[name]=[compile_response(*factor,frames[l],frames[l+1]) for l,factor in enumerate(factors)]
    frames_by_name=dict(calibration=bases,random=random)
    checks=[];absolute=[];reports={name:{} for name in programs}
    reports.update(input_projection_only={},final_projection_only={})
    blocks=[dict(left=b.mlp.Left.weight.double(),right=b.mlp.Right.weight.double(),
                 down=b.mlp.Down.weight.double(),bias=b.mlp.Down_bias.double(),lambdas=b.lambdas.double())
            for b in model.transformer.h[12:18]]
    def score(c,panel):
        xb=c['base']['states'][-1]
        initial=c['base']['states'][0]
        start_delta=c['frozen']['states'][0]-initial
        projected=(start_delta@bases[0])@bases[0].T
        logits,_=execute(blocks,initial+projected,c['initial'],c['attention_ports'],U,eps)
        input_margin=(logits[:,0]-logits[:,1])*c['direction']
        final_delta=c['frozen']['states'][-1]-xb
        final_margin=margin(xb+(final_delta@bases[-1])@bases[-1].T,c['direction'])
        for name,pred in [('input_projection_only',margin(xb,c['direction'])-input_margin),
                          ('final_projection_only',margin(xb,c['direction'])-final_margin)]:
            for i,family in enumerate(c['families']):
                key=panel+'|'+family
                r=reports[name].setdefault(key,dict(prediction=[],conditional=[],native=[]))
                r['prediction'].append(float(pred[i]));r['conditional'].append(float(c['conditional'][i]))
                if c['truth'] is not None:r['native'].append(float(c['truth'][i]))
        for name,program in programs.items():
            frames=frames_by_name[name];z=(c['frozen']['states'][0]-c['base']['states'][0])@frames[0]
            for l,p in enumerate(program):
                z=z*model.transformer.h[l+12].lambdas[0].double()
                h=c['base']['backgrounds'][l];L,R,D=factors[l]
                m0=(((h@L.T)*(h@R.T))@D.T/(h.square().mean(-1,keepdim=True)+eps))@frames[l+1]
                got=evaluate(p,z,h,m0,eps)
                # Independent direct contraction at this proposed projected state.
                L,R,D=factors[l];d=z@frames[l].T
                def numerator(x):return ((x@L.T)*(x@R.T))@D.T
                dense=(d+numerator(h+d)/((h+d).square().mean(-1,keepdim=True)+eps)
                         -numerator(h)/(h.square().mean(-1,keepdim=True)+eps))@frames[l+1]
                checks.append(float((got-dense).norm()/dense.norm().clamp_min(1e-20)))
                absolute.append(float((got-dense).abs().max()));z=got
            xb=c['base']['states'][-1];xe=xb+z@frames[-1].T
            pred=margin(xb,c['direction'])-margin(xe,c['direction'])
            for i,family in enumerate(c['families']):
                key=panel+'|'+family
                r=reports[name].setdefault(key,dict(prediction=[],conditional=[],native=[]))
                r['prediction'].append(float(pred[i]));r['conditional'].append(float(c['conditional'][i]))
                if c['truth'] is not None:r['native'].append(float(c['truth'][i]))
    for c in calibration:score(c,'calibration')
    rowfile=POLY/'SUBJECT_POSITION_STRESS_V644_ROWS.json'
    assert hashlib.sha256(rowfile.read_bytes()).hexdigest()=='03dd63a4b5aa3e4bf77f41280217d449084e2ad5121b0919f7ab0e07ce6c4db1'
    rows=json.loads(rowfile.read_text())
    for template in dict.fromkeys(r['template'] for r in rows):
        score(capture([r for r in rows if r['template']==template],True),'evaluation')
    for report in reports.values():
        for r in report.values():
            p=np.array(r['prediction'])
            for target in ['conditional','native']:
                if not r[target]:continue
                y=np.array(r[target]);yn=np.linalg.norm(y)
                r[target+'_metrics']=dict(relative_error=float(np.linalg.norm(p-y)/max(yn,1e-20)),
                    cosine=float(p@y/max(np.linalg.norm(p)*yn,1e-20)),target_rms=float(np.sqrt(np.mean(y*y))))
    a=max(checks)<=1e-4 and max(absolute)<=1e-4 and counts==dict(prefix_calls=12,suffix_calls=16)
    evaluation=[r for k,r in reports['calibration'].items() if k.startswith('evaluation|')]
    b=a and all(r['conditional_metrics']['relative_error']<=.05 for r in evaluation)
    c=a and all(r['native_metrics']['relative_error']<=.10 for r in evaluation)
    values=sum(v.numel() for p in programs['calibration'] for v in p.values() if isinstance(v,torch.Tensor))+bases[-1].numel()+U.numel()+6
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,
        predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),reports=reports,
        max_local_relative_error=max(checks),max_local_absolute_error=max(absolute),fixed_values=values,
        background_ports='six h vectors, six projected bias-free baseline MLP writes, starting delta, final baseline state',
        calibration_singular_values=singular,wall_seconds=time.perf_counter()-tic,
        scope='Fixed reader anchored width8 control, same calibration/evaluation data; final projection is oracle. Data-informed residual basis, no native circuit adoption.')
    payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'joint response chain')
    OUT.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','fixed_values','max_local_relative_error','max_local_absolute_error','wall_seconds']},indent=2));assert a


if __name__=='__main__':main()
