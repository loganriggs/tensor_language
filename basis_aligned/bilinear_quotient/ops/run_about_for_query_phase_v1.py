#!/usr/bin/env python3
# BQGATE: frozen phase map, native fit recipe, controls, five-arm semantics and four predictions.
"""Managed ABOUT_FOR_QUERY_PHASE_V1; see hash-bound preregistration for every bar.

A valid instrument/replay; B fw phase portability >=.80 with gain>=.10;
C full-vector AND margin causal error<=.10; D short identities/cross control.
Null: query phase alone does not explain contextual transfer. No phase/head/rank rescue.
"""
import hashlib, json, os, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RUNNER=Path(__file__).resolve()
POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import rotary_phase_capture as phase
from circuit_fast_screen_managed_runner import atomic_create_json

PRIOR=POLY/'ABOUT_FOR_QUERY_PHASE_V1_PREREGISTRATION.md'
ROWS=POLY/'ABOUT_FOR_QUERY_PHASE_V1_ROWS.json'
OUT=POLY/'ABOUT_FOR_QUERY_PHASE_V1_RESULT.json'
PARENT=Path(__file__).resolve().parent.parent/'circuits/followups/unit_broad_circuit_v473_result.json'
SHA_FILE=POLY/'ABOUT_FOR_QUERY_PHASE_V1_BINDING.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(SHA_FILE.read_text())
    assert all(sha(p)==v for p,v in binding.items())
    rows=json.loads(ROWS.read_text())
    assert all(sha(p)==v for p,v in rows['source_hashes'].items())
    controls=json.loads((POLY/'ROTARY_PHASE_TRANSPORT_V1_CONTROLS.json').read_text())
    assert controls['transport']['passed'] and controls['capture']['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'model_loaded':False,'gpu_accessed':False,
                          'model_forwards_max':2600,'sequence_evaluations_max':300000,
                          'backward_steps':120,'fit_pairs':80,'fit_controls':80,'held_pairs':128,
                          'gap_histograms':rows['gap_histograms']}));return
    assert not OUT.exists();signal.alarm(600)
    backend=producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    started=time.perf_counter();counts=[0,0];audits=[]
    def count(_m,args,_out):
        counts[0]+=1;counts[1]+=len(args[0])
        assert counts[0]<=2600 and counts[1]<=300000
    counter=backend.model.transformer.h[0].attn.register_forward_hook(count)
    flags=[p.requires_grad for p in backend.model.parameters()]
    def agree(a,b,name):
        a,b=a.double(),b.double();delta=a-b
        maximum=float(delta.abs().max());relative=float(delta.norm()/b.norm().clamp_min(1e-30))
        passed=bool(a.isfinite().all() and b.isfinite().all()) and maximum<=1e-3 and relative<=1e-5
        audits.append(dict(kind=name,max_abs=maximum,relative=relative,passed=passed))
    reports={};parent=json.loads(PARENT.read_text())['groups']['fit_AE']['per_shape']
    try:
        fitrows=sum(rows['fit'].values(),[]);crows=sum(rows['fit_controls'].values(),[])
        fit=g.prepare(backend,fitrows,valid_only=True);assert fit.dropped==0
        ctrl=g.prepare(backend,crows)
        _,_,greedy=g.greedy_heads(backend,fit,pool=40,target=.97,min_gain=.001,max_units=30)
        units=list(greedy['chosen'])
        mu={u:torch.stack([torch.as_tensor(cache[(rid,u)]).float()
                           for cache in (fit.base_cache,fit.donor_cache) for rid in fit.base_batch.row_ids]).mean(0)
            for u in units}
        q,history=g.fit_block_subspace_constrained(backend,fit,units,rank=1,steps=120,lr=.05,
                  seed=0,complement_weight=1.,controls=(ctrl,),control_weight=30.,mu=mu)
        print(json.dumps({'fit_complete':True,'n_units':len(units),'counts':counts}),flush=True)
        # Freeze before any new phase response or held-frame outcome is read.
        with torch.no_grad():
            for cell,held in rows['held'].items():
                p=g.prepare(backend,held,valid_only=True);assert p.dropped==0
                virtual_positions=[]
                for row in held:
                    changed=[i for i,(a,b) in enumerate(zip(row['base_ids'],row['donor_ids'])) if a!=b]
                    assert len(changed)==1 and len(row['base_ids'])==len(row['donor_ids'])
                    virtual_positions.append(min(row['base_semantic_position'],changed[0]+4))
                banks=[]
                for batch,cache in [(p.base_batch,p.base_cache),(p.donor_batch,p.donor_cache)]:
                    with phase.capture(backend.model,batch,virtual_positions) as (native,virtual):
                        backend.native(batch,capture=False)
                    agree(torch.stack([native[k] for k in native]),
                          torch.stack([cache[k] for k in native]),cell+'_native_cache')
                    banks.append(virtual)
                short=virtual_positions==list(p.base_batch.semantic_positions)
                alternative=dict(p.donor_cache)
                if not short:
                    for rid in p.base_batch.row_ids:
                        for u in units:alternative[(rid,u)]=p.base_cache[(rid,u)]+banks[1][(rid,u)]-banks[0][(rid,u)]
                _,baseline=g.forward_units(backend,p.base_batch,return_logits=True)
                outputs={};recoveries={}
                for arm,donor,axes in [('complete',p.donor_cache,None),('axis',p.donor_cache,q),
                                      ('phase_axis',alternative,q),('phase_complete',alternative,None)]:
                    answer,logits=g.forward_units(backend,p.base_batch,units=units,donor_cache=donor,
                                  base_cache=p.base_cache,q=axes,return_logits=True)
                    outputs[arm]=logits.double()
                    patched=[-(float(a)-float(f)) for a,f in answer.tolist()]
                    recoveries[arm]=g.recovery(p,patched)
                idx=torch.arange(len(held),device='cuda')
                ans=torch.tensor(p.donor_batch.answer_ids,device='cuda');foil=torch.tensor(p.donor_batch.foil_ids,device='cuda')
                center=lambda z:z-z.mean(-1,keepdim=True)
                margin=lambda z:z[idx,ans]-z[idx,foil]
                ref=center(outputs['complete']-baseline);candidate=center(outputs['phase_axis']-baseline)
                mr=margin(outputs['complete']-baseline);mc=margin(outputs['phase_axis']-baseline)
                ferr=float((candidate-ref).norm()/ref.norm().clamp_min(1e-8))
                merr=float((mc-mr).norm()/mr.norm().clamp_min(1e-8))
                if short:agree(outputs['phase_axis'],outputs['axis'],cell+'_phase_identity')
                old=parent['verb_preposition_'+cell]
                replay=max(abs(recoveries['complete']-old['units_recovery']),
                           abs(recoveries['axis']-old['cross_abs_recovery']))
                reports[cell]={'recovery':recoveries,'axis_fraction':recoveries['axis']/recoveries['complete'],
                       'phase_fraction':recoveries['phase_axis']/recoveries['complete'],
                       'full_vector_error':ferr,'margin_error':merr,'parent_replay_error':replay,
                       'short_phase_identity':short,'virtual_positions':virtual_positions,
                       'n_held':len(held),'reference_full_norm':float(ref.norm()),'reference_margin_norm':float(mr.norm()),
                       'finite':all(bool(z.isfinite().all()) for z in outputs.values())}
                print(json.dumps({'cell':cell,**reports[cell]}),flush=True)
    finally:
        counter.remove()
        for p,flag in zip(backend.model.parameters(),flags):p.requires_grad_(flag)
    restored=all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    restored=restored and not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules())
    a=restored and len(units)==24 and all(r['parent_replay_error']<=.015 and r['finite'] and r['n_held']==16 for r in reports.values()) and all(x['passed'] for x in audits)
    fw=reports['fw_about_for']
    b=fw['phase_fraction']>=.8 and fw['phase_fraction']-fw['axis_fraction']>=.1
    c=all(r['full_vector_error']<=.1 and r['margin_error']<=.1 for name,r in reports.items() if name!='at_to')
    d=abs(reports['at_to']['recovery']['phase_axis'])<=.3 and all(x['passed'] for x in audits if x['kind'].endswith('_phase_identity'))
    preds={'pred_a_instrument':a,'pred_b_phase_portability':b,'pred_c_causal_fidelity':c,'pred_d_control_identity':d}
    result=dict(terminal='query_phase_screen_complete' if a else 'invalid',predictions=preds,reports=reports,
                audits=audits,units=units,fit_history=history,methods_hooks_restored=restored,
                runner_sha256=sha(RUNNER),binding_sha256=sha(SHA_FILE),wall_seconds=time.perf_counter()-started,
                price=dict(forwards=counts[0],sequence_evaluations=counts[1],backward_steps=120,
                           native_weights=sum(p.numel() for p in backend.model.parameters()),actual_weight_saving=0),
                scope='Opened v473 rows; native-prefix-conditioned phase transport; no independent extraction or new OOD.')
    atomic_create_json(OUT,result);print(json.dumps({'predictions':preds,'price':result['price']}));assert a


if __name__=='__main__':main()
