"""Normalized source-gain query atlas, two shared heads, both tasks.
A: exact controls, native/direct source oracle1e-3/1e-5, 336/6048 counts.
B: same singleton all panels KL.001/.01 zero flips and paired errors<=.10.
C: robust omission dependency sets nonempty Jaccard>=.75. D:zero live>=.01.
Nulls remain exploratory; full native weights charged, no fitting/adoption.
"""
# BQGATE: EXPERIMENT pred_a_instrument pred_b_common_single_source pred_c_shared_dependencies pred_d_live_consumers
import json
import os
from pathlib import Path
import signal
import time
import run_bilin18_l9_embedding_query_v1 as E
import source_gain_attention as M
from circuit_fast_screen_managed_runner import atomic_create_json

S=E.S;P=E.P;N=E.N;POLY=E.POLY;RUNNER=Path(__file__).resolve()
PRIOR=POLY/'BILIN18_L9_QUERY_SOURCE_ATLAS_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_L9_QUERY_SOURCE_ATLAS_V1_RESULT.json'
FILES={'prior':PRIOR,'primitive':Path(M.__file__),'source_algebra':POLY/'projected_query_source_edits.py','parent_runner':Path(E.__file__),'parent_result':E.OUT}
EXPECTED={'prior': '4f3c60a199da299b2027e59b395c0dfc19a34452eb154c6c4d493cb8bb05f2f3', 'primitive': '9cdc51733b3fc9c5d9d586bffab6d8db3c573c3826f4009f12ac31336969704e', 'source_algebra': '8f2854a2fd13614c3fe559cea198d5b68bc43c974eb059a50214c4a9ee9e48f5', 'parent_runner': 'b9f36bf756b0f50bbcf68c08efe6e1640e78b92a29b5df06ae54798d3cabc78e', 'parent_result': '2d7dabb074e1ad2994490eae54c869da0ff4700c71161bd8417474c925027893'}
LABELS=['embedding']+[f'{kind}{j}' for j in range(9) for kind in ('attention','mlp')]


def main():
    observed={k:N.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    assert {k:N.sha(p) for k,p in E.FILES.items()}==E.EXPECTED
    assert {k:N.sha(p) for k,p in S.FILES.items()}==S.EXPECTED
    binding=json.loads(N.BINDING.read_text());assert {p:N.sha(p) for p in binding}==binding
    manifest=json.loads(S.ROWS.read_text());splits={k:v for k,v in manifest['splits'].items() if not k.endswith('_fit')}
    assert all(P.value_sha(v)==manifest['row_sha256'][k] for k,v in splits.items())
    controls=M.controls();assert controls['passed']
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':336,'sequence_evaluations':6048,'controls':controls}));return
    assert not OUT.exists();signal.alarm(600);tic=time.perf_counter()
    backend=N.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    attn=backend.model.transformer.h[9].attn;eps=torch.finfo(torch.float32).eps
    masks={'unity':torch.ones(19,device='cuda',dtype=torch.float64),'zero':torch.zeros(19,device='cuda',dtype=torch.float64)}
    for i,label in enumerate(LABELS):
        masks['only_'+label]=torch.eye(19,device='cuda',dtype=torch.float64)[i]
        masks['omit_'+label]=1-masks['only_'+label]
    counts=[0,0];audits=[];reports={};finite=True
    def counter(_m,args,_out):counts[0]+=1;counts[1]+=len(args[0])
    handle=attn.register_forward_hook(counter)
    parent=json.loads(E.OUT.read_text())
    with torch.inference_mode():
        try:
            for panel,rows in splits.items():
                outputs={k:[] for k in ('native',*masks,'direct_omit_mlp8')}
                for side in ('base','donor'):
                    batch=P.das._batch(backend,rows,side=side);positions=batch.semantic_positions
                    with M.capture(backend.model,positions) as r:out=backend.native(batch,capture=True)
                    outputs['native'].append(P.das.head_logits(backend,P.atlas.states(torch,backend,out,rows)).double())
                    args=tuple(r[k] for k in ('sources','readers','keys','values','cos','sin'))+(positions,)
                    bank=M.prepare(*args,eps)
                    reconstructed=torch.nn.functional.rms_norm(r['sources'].sum(1),(1152,),eps=eps)
                    audits.append({'panel':panel,'side':side,'kind':'source_lineage',**P.agree(reconstructed,r['normalized_input'].double())})
                    for name,mask in masks.items():
                        z=mask.expand(len(rows),-1);read=M.evaluate(bank,z)
                        direct=M.direct(*args,z,eps)
                        audits.append({'panel':panel,'side':side,'kind':name+'_local',**P.agree(read,direct,atol=1e-8,rtol=1e-9)})
                        if name=='unity':audits.append({'panel':panel,'side':side,'kind':'native_read',**P.agree(read,r['native_read'])})
                        with M.install_reads(attn,read,positions):out=backend.native(batch,capture=True)
                        outputs[name].append(P.das.head_logits(backend,P.atlas.states(torch,backend,out,rows)).double())
                    z=masks['omit_mlp8'].expand(len(rows),-1)
                    source=torch.nn.functional.rms_norm(torch.einsum('bs,bsd->bd',z,r['sources']).float(),(1152,),eps=eps)
                    with E.M.install(attn,source,positions):out=backend.native(batch,capture=True)
                    outputs['direct_omit_mlp8'].append(P.das.head_logits(backend,P.atlas.states(torch,backend,out,rows)).double())
                for left,right in (('unity','native'),('direct_omit_mlp8','omit_mlp8')):
                    audits.extend({'panel':panel,'kind':left+'_full_logits',**P.agree(a,b)} for a,b in zip(outputs[left],outputs[right]))
                finite=finite and all(bool(z.isfinite().all()) for values in outputs.values() for z in values)
                report=E.metrics.score(rows,outputs);reports[panel]=report
                audits.append({'panel':panel,'kind':'parent_native_contrast',**P.agree(torch.tensor([r['native_contrast'] for r in report['rows']],dtype=torch.float64),torch.tensor([r['native_contrast'] for r in parent['reports'][panel]['rows']],dtype=torch.float64))})
                print(json.dumps({'panel':panel,'counts':counts,'singleton_pass':[label for label in LABELS if all(report['arms']['only_'+label][k] for k in ('distribution_pass','paired_effect_pass'))]}),flush=True)
        finally:handle.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and 'squared_attention' not in attn.__dict__
    a=finite and restored and counts==[336,6048] and all(x['passed'] for x in audits)
    common=[label for label in LABELS if all(all(r['arms']['only_'+label][k] for k in ('distribution_pass','paired_effect_pass')) for r in reports.values())]
    dependencies={task:[label for label in LABELS if all(r['arms']['omit_'+label]['endpoint_change_to_native_pair_norm']>=.10 for name,r in reports.items() if name.startswith(task+'_'))] for task in ('has','is')}
    left,right=map(set,dependencies.values());jaccard=len(left&right)/len(left|right) if left|right else 0.
    d=all(r['arms']['zero']['endpoint_change_norm']>1e-8 and r['arms']['zero']['endpoint_change_to_native_pair_norm']>=.01 for r in reports.values())
    predictions={'pred_a_instrument':a,'pred_b_common_single_source':bool(common),'pred_c_shared_dependencies':bool(left and right and jaccard>=.75),'pred_d_live_consumers':d}
    result={'terminal':'invalid' if not a else 'source_atlas_complete','predictions':predictions,'common_singleton_sources':common,'robust_dependency_sets':dependencies,'dependency_jaccard':jaccard,'source_labels':LABELS,'reports':reports,'audits':audits,'controls':controls,'hooks_restored':restored,'runner_sha256':N.sha(RUNNER),'authority_sha256':observed,'wall_seconds':time.perf_counter()-tic,'price':{'model_forwards':counts[0],'sequence_evaluations':counts[1],'native_parameters':sum(p.numel() for p in backend.model.parameters()),'new_learned_parameters':0,'actual_weight_saving':0,'source_monomials':190,'adoption':False},'scope':'Opened rows, fixed native sources/keys/values; query-consumer edge edits, not global upstream module removals. All source choices exploratory.'}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ('terminal','predictions','common_singleton_sources','robust_dependency_sets','dependency_jaccard','price','wall_seconds')},indent=2))


if __name__=='__main__':main()
