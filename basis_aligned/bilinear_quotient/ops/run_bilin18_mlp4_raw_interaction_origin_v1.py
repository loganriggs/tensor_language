"""Diagnose current-attention versus earlier-residual mixed input at MLP4.
A native1e-3/1e-5,sourcefoldrelative1e-5,112forwards980seq64localMLPs;
Battention-only/Cearlier-only correctionerror<=.10 bothframes/all16cells;
Dcorrection/fullcue norms>=.10 bothframes/allcells. Fourthcornerdiagnostic only.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_attention_origin pred_c_earlier_origin pred_d_material_mixed_input
import json,os,signal,time
from pathlib import Path
import run_bilin18_mlp4_semantic_square_v2 as V
import four_corner_residual_lineage as L
from circuit_fast_screen_managed_runner import atomic_create_json
P=V.P;N=V.N;C=V.C;E=V.E;G=V.G;A=V.A;POLY=V.POLY
RUNNER=Path(__file__).resolve();PRIOR=POLY/'BILIN18_MLP4_RAW_INTERACTION_ORIGIN_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP4_RAW_INTERACTION_ORIGIN_V1_RESULT.json'
FILES={'prior':PRIOR,'lineage':Path(L.__file__),'square_builder':Path(A.__file__),'squares':A.OUT,'capture':Path(G.__file__),'parent_runner':Path(V.__file__),'parent_result':V.OUT}
EXPECTED={'prior': 'f82b075447cd28973bb4e8c61eb1cfe1e3c9e7d8930c502beb9b0937b110f8e6', 'lineage': '157f30560e4076d193dee9cd76b4cb9c6111d506922d8cbf57f74ff20f2b4b14', 'square_builder': '74d127efceaa931eb8dbc7da7756287b1ff51bda77ebea0a30a2eccb29b840d9', 'squares': '37c486e49dac4406ed966e833212f2f1cf111ed805a0538e891a0c3c19469603', 'capture': 'c0fc4e672e167f9d93fe83ed2f6e4fd2aa09719492c47fb439d74f4b9c0f2217', 'parent_runner': '8387c7528a578d8a0c464f8522bc667541d4524f2eeb1850bf56b4c7be12f0df', 'parent_result': 'cdfa7ff018c6799e96df0ebc85159c5cb4033a514389ea1a8d97e643be1ddd43'}
ARMS=('identity','full','no_mixed','attention_only','earlier_only','direct_oracle')

def main():
    observed={k:N.sha(v) for k,v in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in V.FILES.items()}==V.EXPECTED
    assert {k:N.sha(p) for k,p in E.S.FILES.items()}==E.S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    squares_manifest=json.loads(A.OUT.read_text());assert P.value_sha(A.build())==P.value_sha(squares_manifest)
    rows_manifest=json.loads(E.S.ROWS.read_text());splits=rows_manifest['splits']
    assert all(P.value_sha(v)==rows_manifest['row_sha256'][k] for k,v in splits.items())
    controls=L.controls();assert controls['passed']
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':112,'sequences':980,'local_mlp_calls':64,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;model=backend.model;mlp=model.transformer.h[4].mlp;torch.set_num_threads(2)
    import torch.nn.functional as F
    counts=[0,0];local_calls=0;audits=[];reports={};finite=True;masks=True;recurrence=True;embedding_zero=True
    coefficients=L.coefficients(model)
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=model.transformer.h[9].attn.register_forward_hook(count)
    def synthesize(raw):
        nonlocal local_calls
        local_calls+=1;n=F.rms_norm(raw.float(),(model.config.n_embd,))
        return F.linear(F.linear(n,mlp.Left.weight)*F.linear(n,mlp.Right.weight),mlp.Down.weight)+mlp.Down_bias
    parent=json.loads(V.OUT.read_text())
    try:
        with torch.inference_mode():
            for panel,entry in squares_manifest['panels'].items():
                if panel.endswith('_fit'):continue
                squares=entry['squares'];rowmap={r['row_id']:r for r in splits[panel]};lengths=[s['token_length'] for s in squares]
                valid=lambda x:torch.cat([x[j,:n] for j,n in enumerate(lengths)])
                native={};batches={};native_z={};primal={}
                for context in range(2):
                    for cue in range(2):
                        key=(context,cue);members=[s['context'+str(context)] for s in squares];tokens=tuple(tuple(s['tokens'+str(cue)]) for s in members)
                        answers=tuple(rowmap[s['row_id']][s['side'+str(cue)]+'_answer_id'] for s in members);foils=tuple(rowmap[s['row_id']][s['side'+str(cue)]+'_foil_id'] for s in members)
                        positions=tuple(rowmap[s['row_id']][s['side'+str(cue)]+'_semantic_position'] for s in members);assert all(p==n-1 for p,n in zip(positions,lengths))
                        batch=N.producer.ModelBatch(tuple(s['row_id'] for s in members),'base',tokens,answers,foils,positions);batches[key]=batch
                        with L.capture(model) as lineage:
                            with G.capture(model) as record:backend.native(batch,capture=False)
                        record.update(lineage);native[key]=record;native_z[key]=G.endpoint_logits(record['raw_logits'],batch).double()
                        recurrence=recurrence and record['recurrence_bitwise'];assert set(record['sources'])==set(coefficients)
                        folded=sum(coefficients[k]*s.double() for k,s in record['sources'].items());difference=valid(folded-record['raw'].double())
                        rel=float(difference.norm())/max(float(valid(record['raw'].double()).norm()),1e-8)
                        audits.append({'kind':'native_source_fold','panel':panel,'corner':key,'max_abs':float(difference.abs().max()),'relative_frobenius':rel,'passed':rel<=1e-5})
                        primal[key]=synthesize(record['raw']);audits.append({'kind':'native_mlp_primal','panel':panel,'corner':key,**P.agree(primal[key],record['source'])})
                ans=torch.tensor(batches[(0,1)].answer_ids,device='cuda');foil=torch.tensor(batches[(0,0)].answer_ids,device='cuda');ix=torch.arange(len(squares),device='cuda')
                assert tuple(batches[(1,1)].answer_ids)==tuple(batches[(0,1)].answer_ids) and tuple(batches[(1,0)].answer_ids)==tuple(batches[(0,0)].answer_ids)
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[ix,ans]-z[ix,foil]
                cells={};source_reports={}
                for context in range(2):
                    for cue in range(2):
                        direction=f'context{context}_cue{cue}';recv=(context,cue);target=(context,1-cue)
                        rs=[native[k] for k in ((1-context,cue),(1-context,1-cue),recv,target)]
                        mixed=lambda values:(values[3]-values[2])-(values[1]-values[0])
                        raw=[r['raw'].double() for r in rs];D=mixed(raw)
                        terms={k:coefficients[k]*mixed([r['sources'][k].double() for r in rs]) for k in coefficients}
                        embedding_zero=embedding_zero and bool((terms['embedding']==0).all())
                        at=terms['attn4'];earlier=D-at
                        fields={'no_mixed':raw[3]-D,'attention_only':raw[3]-earlier,'earlier_only':raw[3]-at}
                        replacement={'identity':native[recv]['source'],'full':primal[target],**{k:synthesize(x) for k,x in fields.items()},'direct_oracle':native[target]['source']}
                        z={}
                        for arm in ARMS:
                            with C.replace_mlp(mlp,replacement[arm],lengths):
                                with G.capture(model) as record:backend.native(batches[recv],capture=False)
                            z[arm]=G.endpoint_logits(record['raw_logits'],batches[recv]).double()
                            masks=masks and all(torch.equal(record['source'][j,:n],replacement[arm][j,:n]) and torch.equal(record['source'][j,n:],native[recv]['source'][j,n:]) for j,n in enumerate(lengths))
                        audits.append({'kind':'identity_logits','panel':panel,'direction':direction,**P.agree(z['identity'],native_z[recv])})
                        audits.append({'kind':'full_direct_logits','panel':panel,'direction':direction,**P.agree(z['full'],z['direct_oracle'])})
                        old=parent['reports'][panel]['cells'][direction]['total']['arms']['full']
                        mf=margin(z['full'])-margin(native_z[recv]);full=center(z['full'])-center(native_z[recv]);fn=float(full.norm());mn=float(mf.norm())
                        audits.append({'kind':'parent_margin','panel':panel,'direction':direction,**P.agree(mf,torch.tensor(old['margin_effects'],device='cuda',dtype=torch.float64))})
                        nd=abs(fn-old['effect_norm']);audits.append({'kind':'parent_norm','panel':panel,'direction':direction,'max_abs':nd,'passed':nd<=1e-5*max(old['effect_norm'],1e-8)})
                        frames={}
                        for frame,base in (('total',native_z[recv]),('mixed_input',z['no_mixed'])):
                            ref=center(z['full'])-center(base);mr=margin(z['full'])-margin(base);rn=float(ref.norm());rmn=float(mr.norm());arms={}
                            for arm in ARMS:
                                effect=center(z[arm])-center(base);me=margin(z[arm])-margin(base)
                                err=float((effect-ref).norm())/max(rn,1e-8);merr=float((me-mr).norm())/max(rmn,1e-8)
                                arms[arm]={'relative_effect_error':err,'relative_margin_error':merr,'signed_effect_projection':float((effect*ref).sum())/max(rn*rn,1e-30),'signed_margin_projection':float((me*mr).sum())/max(rmn*rmn,1e-30),'effect_norm':float(effect.norm()),'margin_effects':me.cpu().tolist(),'passed':err<=.10 and merr<=.10}
                            frames[frame]={'reference_effect_norm':rn,'reference_margin_norm':rmn,'arms':arms}
                        frames['material']={'full_vector_ratio':frames['mixed_input']['reference_effect_norm']/max(fn,1e-8),'margin_ratio':frames['mixed_input']['reference_margin_norm']/max(mn,1e-8)}
                        cells[direction]=frames;finite=finite and all(bool(x.isfinite().all()) for x in z.values())
                        norm=float(valid(D).norm());residual=valid(earlier-sum(v for k,v in terms.items() if k!='attn4'))
                        source_reports[direction]={'raw_mixed_norm':norm,'earlier_fold_roundoff_norm':float(residual.norm()),'earlier_fold_roundoff_relative_mixed':float(residual.norm())/max(norm,1e-8),'source_terms':{k:{'norm':float(valid(v).norm()),'signed_projection_onto_raw_mixed':float((valid(v)*valid(D)).sum())/max(norm*norm,1e-30)} for k,v in terms.items()}}
                reports[panel]={'cells':cells,'source_diagnostics':source_reports,'squares':[[s[k]['row_id'] for k in ('context0','context1')] for s in squares]}
                print(json.dumps({'panel':panel,'correction':{d:{a:c['mixed_input']['arms'][a]['relative_effect_error'] for a in ('attention_only','earlier_only')} for d,c in cells.items()},'counts':counts,'local_calls':local_calls}),flush=True)
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules());cells=[c for r in reports.values() for c in r['cells'].values()]
    live=all(c[f][k]>1e-8 for c in cells for f in ('total','mixed_input') for k in ('reference_effect_norm','reference_margin_norm'))
    a=finite and masks and recurrence and embedding_zero and restored and live and counts==[112,980] and local_calls==64 and all(v['passed'] for v in audits)
    preds={'pred_a_instrument':a,'pred_b_attention_origin':all(c['mixed_input']['arms']['attention_only']['passed'] for c in cells),'pred_c_earlier_origin':all(c['mixed_input']['arms']['earlier_only']['passed'] for c in cells),'pred_d_material_mixed_input':all(c['material'][k]>=.10 for c in cells for k in ('full_vector_ratio','margin_ratio'))}
    result={'terminal':'raw_interaction_origin_complete' if a else 'invalid','predictions':preds,'reports':reports,'audits':audits,'controls':controls,'coefficients':coefficients,'native_recurrence_bitwise':recurrence,'embedding_mixed_bitwise_zero':embedding_zero,'source_masks':masks,'hooks_restored':restored,'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'local_mlp_syntheses':local_calls,'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0,'adoption':False},'scope':'Fourth-corner raw-input edge interaction localization. Contextual producers and native suffix retained; no independent extraction or new predictor.'}
    atomic_create_json(OUT,result);print(json.dumps({'terminal':result['terminal'],'predictions':preds,'wall_seconds':result['wall_seconds']}));assert a
if __name__=='__main__':main()
