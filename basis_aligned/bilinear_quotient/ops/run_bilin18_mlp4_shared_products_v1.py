"""Fit-only behavioral product proposals; exact native swaps and shared removal.
A instrument1e-3/1e-5,80forwards1392seq4backwards; Bshared<=.10 full/margin
allcells; Crandom advantage>=.05 bothframes/allcells; Dreference live>1e-8.
No score/rank/gain rescue. Original handoff; all native weights charged.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_shared_sufficient pred_c_random_advantage pred_d_live_reference
import json,os,signal,time
from pathlib import Path
import run_bilin18_mlp4_value_lineage_native_v1 as R
import bilinear_product_subsets as S
import source_margin_gradient as G
from circuit_fast_screen_managed_runner import atomic_create_json
P=R.P;N=R.N;C=R.C;E=R.E;POLY=R.POLY
RUNNER=Path(__file__).resolve();PRIOR=POLY/'BILIN18_MLP4_SHARED_PRODUCTS_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP4_SHARED_PRODUCTS_V1_RESULT.json'
FILES={'prior':PRIOR,'subsets':Path(S.__file__),'gradient':Path(G.__file__),'parent_runner':Path(R.__file__),'parent_result':R.OUT}
EXPECTED={'prior': '4fb09951f7613af1f0862cede59ebd329adac6a7d5d82b84c3825faa3d4fd43c', 'subsets': '3e486b961c7bf83a57c3561ad885861ef9e9f46a0b86297bb81cdce7606690b3', 'gradient': 'c0fc4e672e167f9d93fe83ed2f6e4fd2aa09719492c47fb439d74f4b9c0f2217', 'parent_runner': '7919d9bbc758d037cf212027175d88aa4e942fb773f90c8ca1fcfeea03331f24', 'parent_result': '1c8c41842c41f1975386cc41b5327effe0d85d52e7a2b2ed4fb007b4e45eb80d'}
ARMS=('full','identity','shared','own','cross','random','direct_oracle','remove_shared')

def main():
    observed={k:N.sha(v) for k,v in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in R.FILES.items()}==R.EXPECTED
    assert {k:N.sha(p) for k,p in E.S.FILES.items()}==E.S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    manifest=json.loads(E.S.ROWS.read_text());splits=manifest['splits']
    assert all(P.value_sha(v)==manifest['row_sha256'][k] for k,v in splits.items())
    fit_tokens={tuple(r[s+'_ids']) for k,rows in splits.items() if k.endswith('_fit') for r in rows for s in ('base','donor')}
    eval_tokens={tuple(r[s+'_ids']) for k,rows in splits.items() if not k.endswith('_fit') for r in rows for s in ('base','donor')}
    assert not fit_tokens&eval_tokens
    controls={'subsets':S.controls(),'gradient':G.controls()};assert all(v['passed'] for v in controls.values())
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':80,'sequence_evaluations':1392,'backwards':4,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    model=backend.model;mlp=model.transformer.h[4].mlp;down=mlp.Down.weight.detach()
    original_flags=[p.requires_grad for p in model.parameters()];original_grad=torch.is_grad_enabled()
    counts=[0,0];audits=[];reports={};score={};mask={};finite=True;masks_ok=True
    def count(_m,a,_y):counts[0]+=1;counts[1]+=len(a[0])
    handle=model.transformer.h[9].attn.register_forward_hook(count)
    try:
        for task in ('has','is'):
            rows=splits[task+'_fit'];records=[]
            for side in ('base','donor'):
                batch=P.das._batch(backend,rows,side=side)
                with G.capture(model) as native:backend.native(batch,capture=False)
                record=G.gradient(backend,batch)
                audits.append({'kind':'fit_gradient_primal','task':task,'side':side,**P.agree(record['logits'],G.endpoint_logits(native['raw_logits'],batch))})
                audits.append({'kind':'fit_gradient_phi','task':task,'side':side,**P.agree(record['phi'],native['phi'])})
                records.append(record)
            dp=records[1]['phi'].double()-records[0]['phi'].double()
            saliencies=[]
            for record in records:
                per_token=(dp*(record['gradient'].double()@down.double())).abs()
                saliencies.append(torch.stack([per_token[i,:len(r['base_ids'])].sum(0) for i,r in enumerate(rows)]).mean(0))
            score[task]=(saliencies[0]+saliencies[1])/2
            assert bool(score[task].isfinite().all()) and float(score[task].sum())>0
            score[task]=score[task]/score[task].sum()
        score['shared']=torch.minimum(score['has'],score['is'])
        for name,s in score.items():mask[name]=torch.argsort(s,descending=True,stable=True)[:128].tolist()
        mask['random']=torch.randperm(4608,generator=torch.Generator().manual_seed(910320))[:128].tolist()
        print(json.dumps({'frozen_masks':mask,'fit_counts':counts}),flush=True)
        parent=json.loads(R.OUT.read_text())
        with torch.inference_mode():
            for panel,rows in splits.items():
                if panel.endswith('_fit'):continue
                task=panel.split('_')[0];other='is' if task=='has' else 'has'
                lengths=[len(r['base_ids']) for r in rows];assert all(len(r['donor_ids'])==n for r,n in zip(rows,lengths))
                native=[];outputs={name:[] for name in ('native',)+ARMS}
                for side in ('base','donor'):
                    batch=P.das._batch(backend,rows,side=side)
                    with G.capture(model) as record:backend.native(batch,capture=False)
                    native.append(record);outputs['native'].append(G.endpoint_logits(record['raw_logits'],batch).double())
                for arm in ARMS:
                    indices={'shared':mask['shared'],'own':mask[task],'cross':mask[other],'random':mask['random']}.get(arm,list(range(4608)))
                    for i,side in enumerate(('base','donor')):
                        batch=P.das._batch(backend,rows,side=side)
                        source=native[i]['source']
                        if arm=='identity':replacement=source
                        elif arm=='direct_oracle':replacement=native[1-i]['source']
                        elif arm=='remove_shared':replacement=source-S.subset_write(native[i]['phi'],down,mask['shared'])
                        else:replacement=source+S.interchange(native[i]['phi'],native[1-i]['phi'],down,indices)
                        with C.replace_mlp(mlp,replacement,lengths):
                            with G.capture(model) as record:backend.native(batch,capture=False)
                        masks_ok=masks_ok and all(torch.equal(record['source'][j,:n],replacement[j,:n]) and torch.equal(record['source'][j,n:],source[j,n:]) for j,n in enumerate(lengths))
                        outputs[arm].append(G.endpoint_logits(record['raw_logits'],batch).double())
                finite=finite and all(bool(z.isfinite().all()) for zs in outputs.values() for z in zs)
                for i in range(2):
                    audits.append({'kind':'identity','panel':panel,'side':i,**P.agree(outputs['identity'][i],outputs['native'][i])})
                    audits.append({'kind':'all_products_direct_oracle','panel':panel,'side':i,**P.agree(outputs['full'][i],outputs['direct_oracle'][i])})
                ix=torch.arange(len(rows),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rows],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rows],device='cuda')
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[ix,ans]-z[ix,foil]
                cells={}
                for i,side in enumerate(('base','donor')):
                    full=center(outputs['full'][i])-center(outputs['native'][i]);mf=margin(outputs['full'][i])-margin(outputs['native'][i]);fn=float(full.norm());mn=float(mf.norm())
                    previous=parent['reports'][panel]['cells'][side]['arms']['full_mlp4']
                    audits.append({'kind':'parent_fullsource_margin','panel':panel,'side':side,**P.agree(margin(outputs['direct_oracle'][i])-margin(outputs['native'][i]),torch.tensor(previous['margin_effects'],device='cuda',dtype=torch.float64))})
                    norm_error=abs(float((center(outputs['direct_oracle'][i])-center(outputs['native'][i])).norm())-previous['effect_norm'])
                    audits.append({'kind':'parent_fullsource_norm','panel':panel,'side':side,'max_abs':norm_error,'passed':norm_error<=1e-5*max(previous['effect_norm'],1e-8)})
                    arms={}
                    for arm in ARMS:
                        effect=center(outputs[arm][i])-center(outputs['native'][i]);me=margin(outputs[arm][i])-margin(outputs['native'][i])
                        fe=float((effect-full).norm())/max(fn,1e-8);ma=float((me-mf).norm())/max(mn,1e-8)
                        arms[arm]={'relative_effect_error':fe,'relative_margin_error':ma,'signed_effect_projection':float((effect*full).sum())/max(fn*fn,1e-30),'signed_margin_projection':float((me*mf).sum())/max(mn*mn,1e-30),'effect_norm':float(effect.norm()),'margin_effects':me.cpu().tolist(),'passed':fe<=.10 and ma<=.10}
                    cells[side]={'full_effect_norm':fn,'full_margin_norm':mn,'arms':arms}
                contrast=margin(outputs['native'][1])-margin(outputs['native'][0]);removed=margin(outputs['remove_shared'][1])-margin(outputs['remove_shared'][0])
                removal={'native_contrast_norm':float(contrast.norm()),'remaining_contrast_relative_norm':float(removed.norm())/max(float(contrast.norm()),1e-8),'signed_contrast_retention':float((removed*contrast).sum())/max(float(contrast.square().sum()),1e-30)}
                reports[panel]={'cells':cells,'shared_static_removal':removal,'row_ids':[r['row_id'] for r in rows]}
                print(json.dumps({'panel':panel,'shared':{s:c['arms']['shared'] for s,c in cells.items()},'removal':removal,'counts':counts}),flush=True)
    finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()) and original_flags==[p.requires_grad for p in model.parameters()] and original_grad==torch.is_grad_enabled() and all(p.grad is None for p in model.parameters())
    a=finite and masks_ok and restored and counts==[80,1392] and all(v['passed'] for v in audits)
    cells=[c for r in reports.values() for c in r['cells'].values()]
    preds={'pred_a_instrument':a,'pred_b_shared_sufficient':all(c['arms']['shared']['passed'] for c in cells),'pred_c_random_advantage':all(c['arms']['random'][m]-c['arms']['shared'][m]>=.05 for c in cells for m in ('relative_effect_error','relative_margin_error')),'pred_d_live_reference':all(c['full_effect_norm']>1e-8 and c['full_margin_norm']>1e-8 for c in cells)}
    result={'terminal':'shared_products_complete' if a else 'invalid','predictions':preds,'reports':reports,'audits':audits,'controls':controls,'masks':mask,'normalized_fit_scores':{k:v.cpu().tolist() for k,v in score.items()},'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'restored':restored,'source_masks':masks_ok,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'backwards':4,'native_parameters':sum(p.numel() for p in model.parameters()),'shared_source_weight_entries':442368,'shared_indices':128,'actual_weight_saving':0,'adoption':False},'scope':'Fit-only saliency proposal, exact native source-product swaps, full native suffix. Opened evaluation families; no semantic identification or selective-control claim.'}
    atomic_create_json(OUT,result);print(json.dumps({'terminal':result['terminal'],'predictions':preds,'wall_seconds':result['wall_seconds']}));assert a
if __name__=='__main__':main()
