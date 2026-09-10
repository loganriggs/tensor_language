"""Direct embedding source at L9H1/H4 query ports; no fit or contextual query input.

A:32/576 counts/hashes/restoration, unchanged attention input/unselectedqueries,
identity and FP64 projection1e-3/1e-5, native/removal parent replay.
B:every panel nativeKLmean<=.001,p99<=.01,zero top1 changes.
C:paired centeredlogit andmargin relativeerror<=.10; D:removal live>=.01pairnorm.
All native weights retained, no gradients, modelupdates or adoption.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_embedding_distributions pred_c_embedding_paired_effects pred_d_live_readers
import json
import os
from pathlib import Path
import signal
import time
import run_bilin18_l9_shared_query_router_v1 as S
import embedding_query_router as M
import query_router_metrics as metrics
from circuit_fast_screen_managed_runner import atomic_create_json

P=S.P;N=S.N;POLY=S.POLY;RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_L9_EMBEDDING_QUERY_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_L9_EMBEDDING_QUERY_V1_RESULT.json'
FILES={'prior':PRIOR,'primitive':Path(M.__file__),'metrics':Path(metrics.__file__),'parent_runner':Path(S.__file__),'parent_result':S.OUT}
EXPECTED={'prior': '9be62ea4399fbb6151823facd7bc85b09f146726e8300c80bbc236af657aaadd', 'primitive': 'c3d842eeeef8eb37daffaa0e38a3b4c71cc03ab2af03679458a8faa7d331242d', 'metrics': '0b46f810ce4b1151c85a9a7217def08b49706d73e6762ff70383375fecb7eae5', 'parent_runner': 'e6adb596e63e44b7b84f911304e4dcd56cec547d0bc1609da78bdca8d86f2edd', 'parent_result': '231584f01c683880daec5f38e3ddee52c454c8672afb24cb4ca0d602449c6dc1'}
ARMS=('native','embedding','identity','remove')


def main():
    observed={k:N.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in S.FILES.items()}==S.EXPECTED
    authorities=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in authorities}==authorities
    manifest=json.loads(S.ROWS.read_text());splits={k:v for k,v in manifest['splits'].items() if not k.endswith('_fit')}
    assert sum(map(len,splits.values()))==72
    assert all(P.value_sha(v)==manifest['row_sha256'][k] for k,v in splits.items())
    parent=json.loads(S.OUT.read_text());assert parent['predictions']['pred_a_instrument']
    controls=M.controls();assert controls['passed']
    dry={'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':32,'sequence_evaluations':576,'fit_rows':0,'arms':ARMS,'controls':controls}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    attn=backend.model.transformer.h[9].attn
    coefficients=[b.lambdas.detach().double().cpu().tolist() for b in backend.model.transformer.h[:10]]
    alpha=M.direct_coefficient(coefficients);assert __import__('math').isfinite(alpha) and abs(alpha)>1e-8
    counts=[0,0];audits=[];reports={};unchanged=True;finite=True
    def count(_m,args,_out):counts[0]+=1;counts[1]+=int(args[0].shape[0])
    counter=attn.register_forward_hook(count)
    with torch.inference_mode():
        try:
            for name,rs in splits.items():
                outputs={};native_input={}
                for arm in ARMS:
                    outputs[arm]=[]
                    for side in ('base','donor'):
                        batch=P.das._batch(backend,rs,side=side);positions=batch.semantic_positions
                        tokens=torch.tensor([r[side+'_ids'][int(pos)] for r,pos in zip(rs,positions)],device='cuda')
                        original_embedding=torch.nn.functional.rms_norm(backend.model.transformer.wte(tokens),(1152,))
                        source=torch.nn.functional.rms_norm(alpha*original_embedding,(1152,))
                        context=(M.install(attn,source,positions,S.HEADS,identity=arm=='identity',audit=(local:=[])) if arm in ('embedding','identity')
                                 else S.M.install(attn,positions,S.HEADS,remove=arm=='remove'))
                        inputs=[]
                        with context:
                            h=attn.register_forward_pre_hook(lambda _m,args:inputs.append(args[0].detach().clone()))
                            try:out=backend.native(batch,capture=True)
                            finally:h.remove()
                        assert len(inputs)==1
                        if arm=='native':native_input[side]=inputs[0]
                        else:unchanged=unchanged and torch.equal(inputs[0],native_input[side])
                        if arm in ('embedding','identity'):
                            assert len(local)==2
                            for record in local:
                                unchanged=unchanged and record['unselected_unchanged']
                                audits.append({'panel':name,'arm':arm,'side':side,'kind':record['kind'],**P.agree(record['actual'],record['expected'])})
                        z=P.das.head_logits(backend,P.atlas.states(torch,backend,out,rs)).double()
                        finite=finite and bool(z.isfinite().all());outputs[arm].append(z)
                        assert not attn.c_q._forward_hooks and not attn.c_q2._forward_hooks and not attn.c_proj._forward_pre_hooks and not attn._forward_pre_hooks
                audits.extend([{'panel':name,'kind':'identity_query_full_logit',**P.agree(a,b)} for a,b in zip(outputs['identity'],outputs['native'])])
                report=metrics.score(rs,outputs);reports[name]=report
                old=parent['reports'][name]
                audits.append({'panel':name,'kind':'parent_native_contrast',**P.agree(torch.tensor([r['native_contrast'] for r in report['rows']],dtype=torch.float64),torch.tensor([r['native_contrast'] for r in old['rows']],dtype=torch.float64))})
                # Reusing the scorer is checked against the independently completed parent.
                keys=('mean_kl','p99_kl','max_kl','paired_effect_relative_error','margin_contrast_relative_error','endpoint_change_to_native_pair_norm')
                error=max(abs(report['arms']['remove'][k]-old['arms']['remove'][k]) for k in keys)
                audits.append({'panel':name,'kind':'parent_removal_metrics','max_abs':error,'passed':error<=1e-6 and report['arms']['remove']['top1_changes']==old['arms']['remove']['top1_changes']})
                print(json.dumps({'panel':name,'arms':report['arms'],'forwards':counts}),flush=True)
        finally:counter.remove()
    a=bool(finite and unchanged and counts==[32,576] and not attn._forward_hooks and all(x['passed'] for x in audits))
    b=all(r['arms']['embedding']['distribution_pass'] for r in reports.values())
    c=all(r['arms']['embedding']['paired_effect_pass'] for r in reports.values())
    d=all(r['arms']['remove']['endpoint_change_norm']>1e-8 and r['arms']['remove']['endpoint_change_to_native_pair_norm']>=.01 for r in reports.values())
    result={'terminal':'invalid' if not a else 'direct_embedding_query_screen_pass' if b and c and d else 'contextual_query_input_required_at_registered_fidelity',
        'predictions':{'pred_a_instrument':a,'pred_b_embedding_distributions':b,'pred_c_embedding_paired_effects':c,'pred_d_live_readers':d},
        'alpha_direct_embedding':alpha,'lambda_coefficients':coefficients,'controls':controls,'audits':audits,'inputs_and_unselected_queries_unchanged':unchanged,'reports':reports,
        'authority_sha256':observed,'runner_sha256':N.sha(RUNNER),'wall_seconds':time.perf_counter()-tic,
        'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'fit_updates':0,'model_updates':0,'new_learned_parameters':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_projection_or_storage_saving':0,'adoption':False},
        'scope':'Direct skip/reentry embedding at native query ports with recomputed source norm; opened has/had and is/was rows, no complete token-field or whole-model simplification claim.'}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','alpha_direct_embedding','price','wall_seconds')},indent=2))


if __name__=='__main__':main()
