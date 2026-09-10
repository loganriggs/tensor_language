"""Shared fixed queries in established has/had and is/was contextual read heads.

A:36/624 counts, hashes, unchanged input, hooks/masks, native folded pattern
1e-3 AND1e-5. B: every panel KLmean<=.001,p99<=.01,no top1 change.
C: full centered paired effect and margin contrast relative error<=.10.
D: both-head removal effect>=.01 native paired effect and nonzero.
512 shared prototype scalars, no gradients/weight updates, all native weights priced.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_shared_distributions pred_c_shared_paired_effects pred_d_live_readers
import json
import os
from pathlib import Path
import signal
import time
import run_bilin18_mlp1_dual_reader_native_v1 as P
import shared_value_read_capture as R
import fixed_query_router as M
from circuit_fast_screen_managed_runner import atomic_create_json

RUNNER=Path(__file__).resolve();N=P.N;POLY=P.POLY
PRIOR=POLY/'BILIN18_L9_SHARED_QUERY_ROUTER_V1_PREREGISTRATION.md'
ROWS=POLY/'BILIN18_L9_SHARED_QUERY_ROUTER_V1_ROWS.json'
OUT=POLY/'BILIN18_L9_SHARED_QUERY_ROUTER_V1_RESULT.json'
FILES={'prior':PRIOR,'rows':ROWS,'primitive':Path(M.__file__),'read_capture':Path(R.__file__),'backend_parent':Path(N.__file__),'backend_binding':N.BINDING,'capture_parent':Path(P.__file__)}
EXPECTED={'prior': 'a3e90fb4ece317cb5bcb76d269d6057e209f622d8f2abaa85c5d1a7b7f981190', 'rows': '0a9fbb66f055d10e3c14982eb6251e31854a5b3dbfd9dbb86927b89965851a2e', 'primitive': '3c4d91c8b0bcd6477e607f70ed152c6b7de702462ddb83e67052629f92e9f29f', 'read_capture': '9f7056b572a4feae9974fc5330a6a3cbedff8c223971a56e8b0a4bc30d10f919', 'backend_parent': 'cd9b4efd8ffe9fa8d4b76d2c1113e5de496c695dc20bb9313ac58dc3680abdf2', 'backend_binding': 'd5ee3731c07fe85525c4b69eaa8a7d88cf1d29b8e889e79be30602af96609460', 'capture_parent': '12f00093193f40dbc5c141f5208fe71189e0e911ccb6c397c241a39e7038001f'}
HEADS=(1,4);ARMS=('native','shared','task','remove')


def main():
    observed={k:N.sha(v) for k,v in FILES.items()};assert observed==EXPECTED
    authorities=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in authorities}==authorities
    assert {k:N.sha(v) for k,v in P.FILES.items()}==P.EXPECTED
    manifest=json.loads(ROWS.read_text());splits=manifest['splits']
    assert {k:P.value_sha(v) for k,v in splits.items()}==manifest['row_sha256']
    assert {p:N.sha(p) for p in manifest['source_sha256']}==manifest['source_sha256']
    assert N.sha(manifest['builder'])==manifest['builder_sha256']
    assert manifest['split_counts']=={'has_fit':16,'has_heldout':16,'has_a2':32,'is_fit':8,'is_heldout':8,'is_a2':16}
    controls=M.controls();assert controls['passed']
    dry={'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':36,'sequence_evaluations':624,'fit_rows':24,'evaluation_rows':72,'arms':ARMS,'controls':controls}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    attn=backend.model.transformer.h[9].attn;assert attn.head_dim==128 and attn.n_head==9
    count=[0,0];audits=[];finite=True;inputs_same=True;masks_valid=True;prototypes={};fit_geometry={};reports={}
    def counted(_m,args,_out):count[0]+=1;count[1]+=int(args[0].shape[0])
    counter=attn.register_forward_hook(counted)
    def query_capture(store):
        handles=[]
        for kind,target in enumerate((attn.c_q,attn.c_q2)):
            def save(_m,_args,out,kind=kind):store[kind]=out.detach().clone()
            handles.append(target.register_forward_hook(save))
        return handles
    def endpoint_queries(store,positions):
        return torch.stack([torch.stack([value.view(value.shape[0],value.shape[1],9,128)[i,int(pos),list(HEADS)] for i,pos in enumerate(positions)]) for _,value in sorted(store.items())])
    with torch.inference_mode():
        try:
            for task in ('has','is'):
                samples=[];rs=splits[task+'_fit']
                for side in ('base','donor'):
                    batch=P.das._batch(backend,rs,side=side);raw={};handles=query_capture(raw)
                    try:backend.native(batch,capture=False)
                    finally:
                        for h in handles:h.remove()
                    assert set(raw)=={0,1};samples.append(endpoint_queries(raw,batch.semantic_positions).double())
                samples=torch.cat(samples,dim=1);mean=samples.mean(1)
                prototypes[task]=mean.float()
                fit_geometry[task]={'sample_count':samples.shape[1],'raw_relative_residual':float((samples-mean[:,None]).norm()/samples.norm())}
            # Equal weight per task, then freeze the actual deployed FP32 constants.
            prototypes['shared']=((prototypes['has'].double()+prototypes['is'].double())/2).float()
            assert all(tuple(v.shape)==(2,2,128) and bool(v.isfinite().all()) for v in prototypes.values())
            print(json.dumps({'stage':'prototypes_frozen','forwards':count,'geometry':fit_geometry}),flush=True)
            for name,rs in splits.items():
                if name.endswith('_fit'):continue
                task=name.split('_')[0];outputs={};native_inputs={};native_raw={}
                for arm in ARMS:
                    outputs[arm]=[]
                    prototype=prototypes[arm if arm=='shared' else task] if arm in ('shared','task') else None
                    for side in ('base','donor'):
                        batch=P.das._batch(backend,rs,side=side);positions=batch.semantic_positions;raw={};input_capture=[]
                        with M.install(attn,positions,HEADS,prototype,arm=='remove'):
                            handles=query_capture(raw)
                            handles.append(attn.register_forward_pre_hook(lambda _m,args:input_capture.append(args[0].detach().clone())))
                            try:
                                with R.capture(backend.model,{9:HEADS}) as reads:out=backend.native(batch,capture=True)
                            finally:
                                for h in handles:h.remove()
                        assert set(raw)=={0,1} and len(input_capture)==1 and set(reads)=={9}
                        x=input_capture[0]
                        if arm=='native':native_inputs[side]=x;native_raw[side]=raw
                        else:inputs_same=inputs_same and torch.equal(x,native_inputs[side])
                        if prototype is not None:
                            for kind,value in raw.items():
                                expected=M.override(native_raw[side][kind],prototype[kind],positions,HEADS,128)
                                masks_valid=masks_valid and torch.equal(value,expected)
                            mask=torch.arange(x.shape[1],device='cuda')[None]<=torch.tensor(positions,device='cuda')[:,None]
                            for j,head in enumerate(HEADS):
                                keys=torch.stack([layer.weight[head*128:(head+1)*128] for layer in (attn.c_k,attn.c_k2)])
                                folded=M.folded_pattern(x,prototype[:,j],keys,attn.rotary.cos_cached,attn.rotary.sin_cached,positions,torch.finfo(torch.float32).eps)
                                actual=torch.stack([reads[9][head]['pattern'][i,int(pos)] for i,pos in enumerate(positions)])
                                audits.append({'panel':name,'arm':arm,'side':side,'head':head,**P.agree(folded[mask],actual[mask])})
                        z=P.das.head_logits(backend,P.atlas.states(torch,backend,out,rs)).double()
                        finite=finite and bool(z.isfinite().all());outputs[arm].append(z)
                        assert len(attn._forward_hooks)==1 and not attn._forward_pre_hooks
                        assert not attn.c_q._forward_hooks and not attn.c_q2._forward_hooks and not attn.c_proj._forward_pre_hooks
                        assert 'squared_attention' not in attn.__dict__
                native=outputs['native'];ix=torch.arange(len(rs),device='cuda');ans=torch.tensor([r['donor_answer_id'] for r in rs],device='cuda');foil=torch.tensor([r['donor_foil_id'] for r in rs],device='cuda')
                center=lambda z:z-z.mean(-1,keepdim=True)
                effect=center(native[1])-center(native[0]);effect_norm=float(effect.norm())
                contrasts=lambda zs:(zs[1][ix,ans]-zs[1][ix,foil])-(zs[0][ix,ans]-zs[0][ix,foil])
                contrast=contrasts(native);margin_norm=float(contrast.norm())
                cell={}
                for arm in ('shared','task','remove'):
                    zs=outputs[arm];kl=torch.stack([(b.log_softmax(-1).exp()*(b.log_softmax(-1)-v.log_softmax(-1))).sum(-1) for b,v in zip(native,zs)]).flatten()
                    flips=sum(int((b.argmax(-1)!=v.argmax(-1)).sum()) for b,v in zip(native,zs))
                    error=float((center(zs[1])-center(zs[0])-effect).norm());margin_error=float((contrasts(zs)-contrast).norm())
                    removal_norm=float(torch.stack([center(v)-center(b) for b,v in zip(native,zs)]).norm())
                    cell[arm]={'mean_kl':float(kl.mean()),'p99_kl':float(torch.quantile(kl,.99)),'max_kl':float(kl.max()),'top1_changes':flips,
                        'paired_effect_relative_error':error/effect_norm if effect_norm>1e-8 else None,
                        'margin_contrast_relative_error':margin_error/margin_norm if margin_norm>1e-8 else None,
                        'paired_effect_error_norm':error,'native_paired_effect_norm':effect_norm,'margin_error_norm':margin_error,'native_margin_contrast_norm':margin_norm,
                        'endpoint_change_norm':removal_norm,'endpoint_change_to_native_pair_norm':removal_norm/max(effect_norm,1e-8),
                        'distribution_pass':float(kl.mean())<=.001 and float(torch.quantile(kl,.99))<=.01 and flips==0,
                        'paired_effect_pass':(error<=.1*effect_norm if effect_norm>1e-8 else error<=1e-8) and (margin_error<=.1*margin_norm if margin_norm>1e-8 else margin_error<=1e-8)}
                native_correct=[]
                for s,side in enumerate(('base','donor')):
                    aa=torch.tensor([r[side+'_answer_id'] for r in rs],device='cuda');ff=torch.tensor([r[side+'_foil_id'] for r in rs],device='cuda')
                    native_correct.append({'answer_over_foil':int((native[s][ix,aa]>native[s][ix,ff]).sum()),'full_vocab_correct':int((native[s].argmax(-1)==aa).sum())})
                reports[name]={'pair_count':len(rs),'arms':cell,'native_correct':native_correct,
                    'rows':[{'row_id':r['row_id'],'native_contrast':float(contrast[i]),'arm_contrasts':{a:float(contrasts(outputs[a])[i]) for a in ('shared','task','remove')}} for i,r in enumerate(rs)]}
                print(json.dumps({'panel':name,'arms':cell,'forwards':count}),flush=True)
        finally:counter.remove()
    a=bool(finite and inputs_same and masks_valid and count==[36,624] and all(v['passed'] for v in audits))
    b=all(r['arms']['shared']['distribution_pass'] for r in reports.values())
    c=all(r['arms']['shared']['paired_effect_pass'] for r in reports.values())
    d=all(r['arms']['remove']['endpoint_change_norm']>1e-8 and r['arms']['remove']['endpoint_change_to_native_pair_norm']>=.01 for r in reports.values())
    result={'terminal':'invalid' if not a else 'shared_fixed_query_router_screen_pass' if b and c and d else 'shared_fixed_query_router_null',
        'predictions':{'pred_a_instrument':a,'pred_b_shared_distributions':b,'pred_c_shared_paired_effects':c,'pred_d_live_readers':d},
        'controls':controls,'fold_audits':audits,'inputs_unchanged':inputs_same,'query_masks_valid':masks_valid,
        'prototypes':{k:v.cpu().tolist() for k,v in prototypes.items()},'fit_geometry':fit_geometry,'reports':reports,
        'authority_sha256':observed,'runner_sha256':N.sha(RUNNER),
        'price':{'model_forwards':count[0],'sequence_evaluations':count[1],'shared_prototype_scalars':512,'separate_task_prototype_scalars':1024,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'model_updates':0,'gradient_steps':0,'adoption':False,'actual_projection_or_storage_saving':0},
        'scope':'Opened unfiltered has/had and is/was task rows; fit-disjoint A1/A2 evaluation. Fixed contextual-key reader hypothesis, not a shared-value or whole-model simplification claim.',
        'wall_seconds':time.perf_counter()-tic}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','price','wall_seconds')},indent=2))


if __name__=='__main__':main()
