"""Four-corner source prediction, preserving full native downstream execution.
A: 1e-3/1e-5 native,1e-9 algebra,128forwards1120seq. Btotal ANDCinteraction
<=.10 relativefull-logit/margin all16cells; Dbothreferenceslive>1e-8.
No fourthinput inlocalprediction; inheriteddiagnostic only. Allweightscharged.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_total_cue_prediction pred_c_interaction_prediction pred_d_live_reference
import json,os,signal,time
from pathlib import Path
import run_bilin18_mlp4_shared_products_v1 as Q
import semantic_square_bilinear as S
import audit_semantic_square_rows_v1 as A
from circuit_fast_screen_managed_runner import atomic_create_json
R=Q.R;P=Q.P;N=Q.N;C=Q.C;E=Q.E;G=Q.G;POLY=Q.POLY
RUNNER=Path(__file__).resolve();PRIOR=POLY/'BILIN18_MLP4_SEMANTIC_SQUARE_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP4_SEMANTIC_SQUARE_V1_RESULT.json'
FILES={'prior':PRIOR,'square_algebra':Path(S.__file__),'square_builder':Path(A.__file__),'squares':A.OUT,'capture':Path(G.__file__),'parent_runner':Path(R.__file__),'parent_result':R.OUT}
EXPECTED={'prior': 'b18470e457b28c841c43ba826c008d93a3cd8755ce18906987096223b04b225b', 'square_algebra': 'c5f92bb2a4746413d551b488a8d9452f722426fa7de1b5554c4c6be7fe0874bb', 'square_builder': '74d127efceaa931eb8dbc7da7756287b1ff51bda77ebea0a30a2eccb29b840d9', 'squares': '37c486e49dac4406ed966e833212f2f1cf111ed805a0538e891a0c3c19469603', 'capture': 'c0fc4e672e167f9d93fe83ed2f6e4fd2aa09719492c47fb439d74f4b9c0f2217', 'parent_runner': '7919d9bbc758d037cf212027175d88aa4e942fb773f90c8ca1fcfeea03331f24', 'parent_result': '1c8c41842c41f1975386cc41b5327effe0d85d52e7a2b2ed4fb007b4e45eb80d'}
ARMS=('identity','full','transport','local','inherited','exactsum','independent_prediction')

def main():
    observed={k:N.sha(v) for k,v in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in R.FILES.items()}==R.EXPECTED
    assert {k:N.sha(p) for k,p in E.S.FILES.items()}==E.S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    square_manifest=json.loads(A.OUT.read_text());assert P.value_sha(A.build())==P.value_sha(square_manifest)
    row_manifest=json.loads(E.S.ROWS.read_text());splits=row_manifest['splits']
    assert all(P.value_sha(v)==row_manifest['row_sha256'][k] for k,v in splits.items())
    controls=S.controls();assert controls['passed']
    import torch
    t=torch.arange(4,dtype=torch.float64).reshape(1,4);l=torch.eye(4,dtype=torch.float64)
    a=S.partition(t,t+1,t+2,t+3,l,l,l);b=S.partition(t,t+1,t+2,t-5,l,l,l)
    fourth_independent=torch.equal(a['local_product'],b['local_product']) and torch.equal(a['three_corner_input'],b['three_corner_input'])
    assert fourth_independent
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':128,'sequences':1120,'controls':controls,'local_fourth_independent':fourth_independent}));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');model=backend.model;mlp=model.transformer.h[4].mlp;torch.set_num_threads(2)
    left,right,down=(m.weight.detach().double() for m in (mlp.Left,mlp.Right,mlp.Down));bias=mlp.Down_bias.detach().double()
    bilinear=lambda x,y:S.bilinear(x,y,left,right,down)
    counts=[0,0];audits=[];reports={};finite=True;source_masks=True
    def count(_m,args,_y):counts[0]+=1;counts[1]+=len(args[0])
    handle=model.transformer.h[9].attn.register_forward_hook(count)
    def execute(batch,replacement=None,lengths=None):
        from contextlib import nullcontext
        with (C.replace_mlp(mlp,replacement,lengths) if replacement is not None else nullcontext()):
            with G.capture(model) as record:
                hook=mlp.register_forward_pre_hook(lambda _m,args:record.update(n=args[0].detach().clone()))
                try:backend.native(batch,capture=False)
                finally:hook.remove()
        return record,G.endpoint_logits(record['raw_logits'],batch).double()
    parent=json.loads(R.OUT.read_text())
    try:
        with torch.inference_mode():
            for panel,entry in square_manifest['panels'].items():
                if panel.endswith('_fit'):continue
                squares=entry['squares'];rows=splits[panel];rowmap={r['row_id']:r for r in rows};rowindex={r['row_id']:i for i,r in enumerate(rows)}
                lengths=[s['token_length'] for s in squares];native={};batches={};outputs={};local_reports={}
                for context in range(2):
                    for cue in range(2):
                        key=(context,cue);members=[s['context'+str(context)] for s in squares];tokens=tuple(tuple(s['tokens'+str(cue)]) for s in members)
                        answers=tuple(rowmap[s['row_id']][s['side'+str(cue)]+'_answer_id'] for s in members)
                        foils=tuple(rowmap[s['row_id']][s['side'+str(cue)]+'_foil_id'] for s in members)
                        positions=tuple(rowmap[s['row_id']][s['side'+str(cue)]+'_semantic_position'] for s in members)
                        assert all(p==n-1 for p,n in zip(positions,lengths))
                        batch=N.producer.ModelBatch(tuple(s['row_id'] for s in members),'base',tokens,answers,foils,positions);batches[key]=batch
                        native[key],outputs[key]=execute(batch)
                        expected=bilinear(native[key]['n'].double(),native[key]['n'].double())+bias
                        audits.append({'kind':'native_mlp_weight_oracle','panel':panel,'corner':key,**P.agree(expected,native[key]['source'].double())})
                ans=torch.tensor(batches[(0,1)].answer_ids,device='cuda');foil=torch.tensor(batches[(0,0)].answer_ids,device='cuda');ix=torch.arange(len(squares),device='cuda')
                assert torch.equal(ans,torch.tensor(batches[(1,1)].answer_ids,device='cuda')) and torch.equal(foil,torch.tensor(batches[(1,0)].answer_ids,device='cuda'))
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[ix,ans]-z[ix,foil]
                valid=lambda x:torch.cat([x[j,:n] for j,n in enumerate(lengths)])
                cells={}
                for context in range(2):
                    for cue in range(2):
                        direction=f'context{context}_cue{cue}';recv=(context,cue);target=(context,1-cue);otherrecv=(1-context,cue);othertarget=(1-context,1-cue)
                        rs=[native[k] for k in (otherrecv,othertarget,recv,target)]
                        part=S.partition(*(r['n'].double() for r in rs),left,right,down)
                        audits.append({'kind':'fp64_partition','panel':panel,'direction':direction,**P.agree(valid(part['full']),valid(part['local_product']+part['inherited_input']),atol=1e-9,rtol=1e-9)})
                        transport=rs[2]['source'].double()+rs[1]['source'].double()-rs[0]['source'].double()
                        replacements={'identity':rs[2]['source'],'full':rs[3]['source'],'transport':transport,'local':transport+part['local_product'],'inherited':transport+part['inherited_input'],'exactsum':transport+part['local_product']+part['inherited_input'],'independent_prediction':bilinear(part['three_corner_input'],part['three_corner_input'])+bias}
                        for a,b in (('exactsum','full'),('local','independent_prediction')):
                            audits.append({'kind':'local_write_'+a+'_'+b,'panel':panel,'direction':direction,**P.agree(valid(replacements[a]),valid(replacements[b]))})
                        z={}
                        for arm,replacement in replacements.items():
                            record,z[arm]=execute(batches[recv],replacement,lengths)
                            source_masks=source_masks and all(torch.equal(record['source'][j,:n],replacement[j,:n].to(record['source'])) and torch.equal(record['source'][j,n:],rs[2]['source'][j,n:]) for j,n in enumerate(lengths))
                        for a,b in (('identity','native'),('exactsum','full'),('local','independent_prediction')):
                            audits.append({'kind':'logits_'+a+'_'+b,'panel':panel,'direction':direction,**P.agree(z[a],outputs[recv] if b=='native' else z[b])})
                        previous=[]
                        for square in squares:
                            member=square['context'+str(context)];row=rowmap[member['row_id']];side=member['side'+str(cue)];old=parent['reports'][panel]['cells'][side]['arms']['full_mlp4']['margin_effects'][rowindex[member['row_id']]]
                            sign=1 if row['donor_answer_id']==row[member['side1']+'_answer_id'] else -1
                            previous.append(sign*old)
                        audits.append({'kind':'parent_fullsource_margin','panel':panel,'direction':direction,**P.agree(margin(z['full'])-margin(outputs[recv]),torch.tensor(previous,dtype=torch.float64,device='cuda'))})
                        finite=finite and all(bool(v.isfinite().all()) for v in z.values())
                        frames={}
                        for frame,baseline in (('total',outputs[recv]),('interaction',z['transport'])):
                            full=center(z['full'])-center(baseline);mf=margin(z['full'])-margin(baseline);fn=float(full.norm());mn=float(mf.norm());arms={}
                            for arm in ARMS:
                                effect=center(z[arm])-center(baseline);me=margin(z[arm])-margin(baseline)
                                error=float((effect-full).norm())/max(fn,1e-8);merror=float((me-mf).norm())/max(mn,1e-8)
                                arms[arm]={'relative_effect_error':error,'relative_margin_error':merror,'signed_effect_projection':float((effect*full).sum())/max(fn*fn,1e-30),'signed_margin_projection':float((me*mf).sum())/max(mn*mn,1e-30),'effect_norm':float(effect.norm()),'margin_effects':me.cpu().tolist(),'passed':error<=.10 and merror<=.10}
                            frames[frame]={'reference_effect_norm':fn,'reference_margin_norm':mn,'arms':arms}
                        cells[direction]=frames
                        actual=valid(rs[3]['source'].double()-rs[2]['source'].double()-rs[1]['source'].double()+rs[0]['source'].double());norm=float(actual.norm())
                        local_reports[direction]={'native_mixed_write_norm':norm,'input_nonadditivity_norm':float(valid(part['input_nonadditivity']).norm()),'parts':{name:{'norm':float(valid(part[name]).norm()),'relative_error':float((valid(part[name])-actual).norm())/max(norm,1e-8),'signed_projection':float((valid(part[name])*actual).sum())/max(norm*norm,1e-30)} for name in ('local_product','inherited_input')}}
                reports[panel]={'cells':cells,'local_partitions':local_reports,'squares':[[s[k]['row_id'] for k in ('context0','context1')] for s in squares]}
                print(json.dumps({'panel':panel,'local_prediction':{d:{f:v['arms']['local']['relative_effect_error'] for f,v in c.items()} for d,c in cells.items()},'counts':counts}),flush=True)
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    a=finite and source_masks and restored and counts==[128,1120] and all(v['passed'] for v in audits)
    cells=[c for r in reports.values() for c in r['cells'].values()]
    preds={'pred_a_instrument':a,'pred_b_total_cue_prediction':all(c['total']['arms']['local']['passed'] for c in cells),'pred_c_interaction_prediction':all(c['interaction']['arms']['local']['passed'] for c in cells),'pred_d_live_reference':all(v['reference_effect_norm']>1e-8 and v['reference_margin_norm']>1e-8 for c in cells for v in c.values())}
    result={'terminal':'semantic_square_complete' if a else 'invalid','predictions':preds,'reports':reports,'audits':audits,'controls':controls,'local_fourth_independent':fourth_independent,'hooks_restored':restored,'source_masks':source_masks,'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0,'adoption':False},'scope':'Anchored local-vs-inherited cue/context interaction on opened texts, all source tokens, full native suffix. Fourthinput diagnostic not heldout prediction; no rawtext extraction.'}
    atomic_create_json(OUT,result);print(json.dumps({'terminal':result['terminal'],'predictions':preds,'wall_seconds':result['wall_seconds']}));assert a
if __name__=='__main__':main()
