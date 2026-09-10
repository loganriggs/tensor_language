"""Independent token-weight cue-message executor and native carrier screen.
A numerical1e-3/1e-5,FP64support1e-9,40forwards720seq8tokenproducers;
Bdirectwrite signedpairedprojection>=.10 bothframes/all8cells;
Ccompiledfiniteeffecterror<=.01 bothframes/allcells. FirstV staysreceivingnative.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_cue_carrier pred_c_compiled_fidelity
import json,os,signal,time
from pathlib import Path
import run_bilin18_mlp4_shared_products_v1 as Q
import first_attention_token_program as F
import norm_preserving_response_hooks as H
from circuit_fast_screen_managed_runner import atomic_create_json
P=Q.P;N=Q.N;E=Q.E;G=Q.G;POLY=Q.POLY
RUNNER=Path(__file__).resolve();PRIOR=POLY/'BILIN18_FIRST_ATTENTION_CUE_MESSAGE_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_FIRST_ATTENTION_CUE_MESSAGE_V1_RESULT.json'
FILES={'prior':PRIOR,'program':Path(F.__file__),'support':Path(F.T.__file__),'factor_capture':Path(H.__file__),'logit_capture':Path(G.__file__),'row_parent_runner':Path(E.S.__file__),'rows':E.S.ROWS}
EXPECTED={'prior': '3aedf10ae3feda5616ef11805e6c182d1fe6a630bc6e7db7de5d65dfa4a6a10a', 'program': '1309c6fd5d05126c542775647870c08e01482dbd7d91ce82c39034656ea7db86', 'support': 'd41eaccfa49df75829a635a5e45dd64bbc66c855f261dddcdd1e4754e190ca70', 'factor_capture': '697a3f0924326cbd81cda25c11ec94e3674956c908f90ffa7caa6e9b738b75aa', 'logit_capture': 'c0fc4e672e167f9d93fe83ed2f6e4fd2aa09719492c47fb439d74f4b9c0f2217', 'row_parent_runner': 'e6adb596e63e44b7b84f911304e4dcd56cec547d0bc1609da78bdca8d86f2edd', 'rows': '0a9fbb66f055d10e3c14982eb6251e31854a5b3dbfd9dbb86927b89965851a2e'}
ARMS=('direct_swap','compiled_swap','compiled_identity','remove_cue_sources')

def main():
    observed={k:N.sha(v) for k,v in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in E.S.FILES.items()}==E.S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    manifest=json.loads(E.S.ROWS.read_text());splits={k:v for k,v in manifest['splits'].items() if not k.endswith('_fit')}
    assert all(P.value_sha(v)==manifest['row_sha256'][k] for k,v in splits.items())
    controls=F.controls();assert controls['passed']
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':40,'sequences':720,'token_factor_productions':8,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;model=backend.model;attn=model.transformer.h[0].attn;torch.set_num_threads(2)
    counts=[0,0];productions=0;audits=[];reports={};finite=True;padding=True;payload=True
    def count(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    counter=model.transformer.h[9].attn.register_forward_hook(count)
    try:
        with torch.inference_mode():
            for panel,rows in splits.items():
                lengths=[len(r['base_ids']) for r in rows];assert all(len(r['donor_ids'])==n for r,n in zip(rows,lengths))
                valid=lambda x:torch.cat([x[j,:n] for j,n in enumerate(lengths)])
                records=[];factorbanks=[];inputs=[];batches=[];outputs={k:[] for k in ('native',)+ARMS}
                for side in ('base','donor'):
                    batch=P.das._batch(backend,rows,side=side);batches.append(batch)
                    with H.capture(model,layers=(0,)) as native_factors:
                        with F.capture(attn) as record:
                            with G.capture(model) as log:backend.native(batch,capture=False)
                    outputs['native'].append(G.endpoint_logits(log['raw_logits'],batch).double());records.append(record)
                    tokens,_=backend._tensor_batch(batch);inputs.append(tokens);fs=F.factors(model,tokens);factorbanks.append(fs);productions+=1
                    for j,(candidate,native) in enumerate(zip(fs,native_factors[0]['factors'])):audits.append({'kind':'factor','panel':panel,'side':side,'factor':j,**P.agree(candidate,native)})
                    audits.append({'kind':'full_write','panel':panel,'side':side,**P.agree(valid(F.write(model,fs)),valid(record['write']))})
                changed=inputs[0]!=inputs[1];edges=F.T.single_edit_edges(changed)
                delta=F.cue_delta(model,*factorbanks,changed)
                audits.append({'kind':'native_delta','panel':panel,**P.agree(valid(delta),valid(records[1]['write']-records[0]['write']))})
                f64=[tuple(x.double() for x in fs) for fs in factorbanks]
                exact=F.read(f64[1])-F.read(f64[0]);limited=F.read(f64[1],edges)-F.read(f64[0],edges)
                audits.append({'kind':'fp64_support','panel':panel,**P.agree(exact,limited,atol=1e-9,rtol=1e-9)})
                source_edges=changed[:,None,:].expand(-1,changed.shape[1],-1)
                for arm in ARMS:
                    for i,side in enumerate(('base','donor')):
                        if arm=='direct_swap':replacement=records[1-i]['write']
                        elif arm=='compiled_swap':replacement=records[i]['write']+(1-2*i)*delta
                        elif arm=='compiled_identity':replacement=records[i]['write']+F.cue_delta(model,factorbanks[i],factorbanks[i],torch.zeros_like(changed))
                        else:replacement=records[i]['write']-F.write(model,factorbanks[i],source_edges)
                        checks=[]
                        with F.replace_write(attn,replacement,lengths,checks):
                            with F.capture(attn) as record:
                                with G.capture(model) as log:backend.native(batches[i],capture=False)
                        outputs[arm].append(G.endpoint_logits(log['raw_logits'],batches[i]).double())
                        payload=payload and checks==[True] and torch.equal(record['first_value'],records[i]['first_value'])
                        padding=padding and all(torch.equal(record['write'][j,n:],records[i]['write'][j,n:]) and torch.equal(record['write'][j,:n],replacement[j,:n]) for j,n in enumerate(lengths))
                for i in range(2):
                    audits.append({'kind':'compiled_direct_logits','panel':panel,'side':i,**P.agree(outputs['compiled_swap'][i],outputs['direct_swap'][i])})
                    audits.append({'kind':'identity_logits','panel':panel,'side':i,**P.agree(outputs['compiled_identity'][i],outputs['native'][i])})
                finite=finite and all(bool(z.isfinite().all()) for zs in outputs.values() for z in zs)
                ix=torch.arange(len(rows),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rows],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rows],device='cuda')
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[ix,ans]-z[ix,foil]
                target=center(outputs['native'][1])-center(outputs['native'][0]);mt=margin(outputs['native'][1])-margin(outputs['native'][0]);tn=float(target.norm());mn=float(mt.norm());cells={}
                for i,side in enumerate(('base','donor')):
                    direct=center(outputs['direct_swap'][i])-center(outputs['native'][i]);md=margin(outputs['direct_swap'][i])-margin(outputs['native'][i])
                    compiled=center(outputs['compiled_swap'][i])-center(outputs['native'][i]);mc=margin(outputs['compiled_swap'][i])-margin(outputs['native'][i]);dn=float(direct.norm());dmn=float(md.norm());sign=1-2*i
                    ep=sign*float((direct*target).sum())/max(tn*tn,1e-30);mp=sign*float((md*mt).sum())/max(mn*mn,1e-30)
                    error=float((compiled-direct).norm())/max(dn,1e-8);merror=float((mc-md).norm())/max(dmn,1e-8)
                    cells[side]={'natural_effect_norm':tn,'natural_margin_norm':mn,'direct_effect_norm':dn,'direct_margin_norm':dmn,'signed_paired_effect_projection':ep,'signed_paired_margin_projection':mp,'compiled_relative_effect_error':error,'compiled_relative_margin_error':merror,'direct_margin_effects':md.cpu().tolist(),'compiled_margin_effects':mc.cpu().tolist(),'carrier_passed':ep>=.10 and mp>=.10,'compiled_passed':error<=.01 and merror<=.01}
                removed=margin(outputs['remove_cue_sources'][1])-margin(outputs['remove_cue_sources'][0])
                removal={'signed_contrast_retention':float((removed*mt).sum())/max(mn*mn,1e-30),'relative_contrast_norm':float(removed.norm())/max(mn,1e-8),'margin_contrasts':removed.cpu().tolist()}
                reports[panel]={'cells':cells,'cue_source_removal':removal,'row_ids':[r['row_id'] for r in rows],'single_cue_edges_per_head':int((edges & (torch.arange(changed.shape[1],device='cuda')[None,:,None]<torch.tensor(lengths,device='cuda')[:,None,None])).sum())}
                print(json.dumps({'panel':panel,'cells':cells,'removal':removal,'counts':counts}),flush=True)
    finally:counter.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()) and 'squared_attention' not in attn.__dict__
    cells=[c for r in reports.values() for c in r['cells'].values()];live=all(c[k]>1e-8 for c in cells for k in ('natural_effect_norm','natural_margin_norm','direct_effect_norm','direct_margin_norm'))
    a=finite and payload and padding and restored and live and counts==[40,720] and productions==8 and all(v['passed'] for v in audits)
    preds={'pred_a_instrument':a,'pred_b_cue_carrier':all(c['carrier_passed'] for c in cells),'pred_c_compiled_fidelity':all(c['compiled_passed'] for c in cells)}
    result={'terminal':'first_attention_cue_message_complete' if a else 'invalid','predictions':preds,'reports':reports,'audits':audits,'controls':controls,'receiving_first_value_preserved':payload,'padding_preserved':padding,'hooks_restored':restored,'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'token_factor_productions':productions,'native_parameters':sum(p.numel() for p in model.parameters()),'actual_weight_saving':0,'adoption':False},'scope':'Exact token-weight attention0 write-change program, receivingfirstV/background retained. Carrier and fidelity screen on opened rows, no semanticidentification or smallerwholemodel.'}
    atomic_create_json(OUT,result);print(json.dumps({'terminal':result['terminal'],'predictions':preds,'wall_seconds':result['wall_seconds']}));assert a
if __name__=='__main__':main()
