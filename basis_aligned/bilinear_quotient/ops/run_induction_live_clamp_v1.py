#!/usr/bin/env python3
# BQGATE: live equality-contribution semantics;81forwards2592seq,0fits.
"""A instrument; B necessary joint invariance; C necessary filler controls."""
import hashlib,json,os,signal,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).resolve();OPS=RUNNER.parent;ROOT=RUNNER.parents[3]
POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(OPS));sys.path.insert(0,str(ROOT))
from circuit_fast_screen_managed_runner import atomic_create_json
from induction_context_transport_v2 import digest,lse
import induction_live_clamp_v1 as C
import attention_write_factorial_executor_v1 as S
BINDING=POLY/'INDUCTION_LIVE_CLAMP_V1_BINDING.json'
OUT=POLY/'INDUCTION_LIVE_CLAMP_V1_RESULT.json'
RAW=Path('/dev/shm/bilin18_induction_live_clamp_v1')


def row_bank():
    envelope=json.loads((POLY/'INDUCTION_R594_MANAGED_RESULT.json').read_text())
    root=Path(envelope['raw_root'])/'induction_centered_fixed_geometry_rung594_evidence/FIT'
    receipt=json.loads((POLY/'induction_centered_fixed_geometry_rung594_receipt.json').read_text())
    names=('directed_records.jsonl','endpoint_records.jsonl','endpoint_tokens.npy',
           'factorized_equality_term.npy','directed_replay_logits.npy')
    for name in names:
        assert digest(root/name)==receipt['evidence_files']['FIT/'+name]['sha256'],name
    endpoints={r['endpoint_id']:r for r in map(json.loads,(root/'endpoint_records.jsonl').open())}
    all_rows=list(map(json.loads,(root/'directed_records.jsonl').open()))
    chosen={r['directed_id'] for r in map(json.loads,(POLY/'INDUCTION_CONTEXT_TRANSPORT_V2_ROWS.jsonl').open())}
    indices=[i for i,r in enumerate(all_rows) if r['directed_id'] in chosen]
    bank=[all_rows[i] for i in indices]
    assert len(bank)==len(chosen)==864 and all(not r['answer_changes'] for r in bank)
    assert len({r['group_id'] for r in bank})==72
    tokens=np.load(root/'endpoint_tokens.npy',mmap_mode='r')
    terms=np.load(root/'factorized_equality_term.npy',mmap_mode='r')
    recipient=np.array([endpoints[r['recipient_endpoint_id']]['array_index'] for r in bank])
    donor=np.array([endpoints[r['donor_endpoint_id']]['array_index'] for r in bank])
    specs=[endpoints[r['recipient_endpoint_id']] for r in bank]
    return bank,np.asarray(tokens[recipient]),specs,np.asarray(terms[recipient]),np.asarray(terms[donor]),np.asarray(np.load(root/'directed_replay_logits.npy',mmap_mode='r')[indices])


def main():
    requested_dry=bool(os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'))
    binding=json.loads(BINDING.read_text());assert all(digest(p)==v for p,v in binding.items())
    bank,tokens,specs,recipient,donor,old_replay=row_bank()
    assert tokens.shape==(864,30) and recipient.shape==donor.shape==(864,4,1152)
    if requested_dry:
        print(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,rows=864,groups=72,forwards=81,sequences=2592,controls=C.controls())));return
    assert not OUT.exists() and not RAW.exists()
    stat=os.statvfs('/dev/shm');assert stat.f_bavail*stat.f_frsize>=2**30
    RAW.mkdir();signal.alarm(600)
    import induction_centered_fixed_geometry_rung594 as p
    from induction_centered_fixed_geometry_rung594_runtime import R594ModelExecutor
    authority,_=p.load_authority();runtime=R594ModelExecutor(p,authority)
    torch=runtime.torch;torch.set_num_threads(2)
    counts=[0,0]
    def count(_m,a,_o):counts[0]+=1;counts[1]+=len(a[0])
    handle=runtime.model.transformer.h[0].attn.register_forward_hook(count)
    saved={};terms_by_arm={};state_by_arm={};observed=0;maximum_rounding=0.;tic=time.perf_counter()
    try:
        for arm,target in (('native',None),('self',recipient),('live',donor)):
            outs=[];terms_list=[];states_list=[]
            for start in range(0,len(bank),32):
                stop=start+32
                z,live,states,observation=C.execute(runtime,tokens[start:stop],specs[start:stop],None if target is None else target[start:stop])
                outs.append(z);terms_list.append(live);states_list.append(states)
                if observation is not None:
                    observed+=32*3
                    maximum_rounding=max(maximum_rounding,observation['maximum_rounding_residual'])
            saved[arm]=np.concatenate(outs)
            np.save(RAW/(arm+'_logits.npy'),saved[arm])
            if target is not None:
                terms_by_arm[arm]=np.concatenate(terms_list)
                np.save(RAW/(arm+'_current_terms.npy'),terms_by_arm[arm])
                state_by_arm[arm]={k:np.concatenate([s[k] for s in states_list]) for k in ('before','total','after')}
                for k,a in state_by_arm[arm].items():np.save(RAW/(arm+'_layer_'+k+'.npy'),a)
    finally:handle.remove()
    bridges=dict(native_saved=S.bridge(saved['native'],old_replay),self_native=S.bridge(saved['self'],saved['native']))
    first_error=S.bridge(terms_by_arm['live'][:,0],recipient[:,0])
    correction=terms_by_arm['live']-recipient
    relative=np.linalg.norm(correction[:,1:].astype(np.float64),axis=-1)/np.maximum(np.linalg.norm(recipient[:,1:].astype(np.float64),axis=-1),1e-30)
    joint=np.array([r['family']=='selector_payload_joint_answer_preserved' for r in bank])
    active=float(np.mean(np.max(relative[joint],axis=-1)>1e-5))
    original=json.loads((POLY/'induction_centered_fixed_geometry_rung594_results.json').read_text())['split_scores']['FIT']
    rows=[];grouped=defaultdict(list)
    for i,r in enumerate(bank):
        z=saved['native'][i].astype(np.float64);a=saved['live'][i].astype(np.float64)
        answer,other=r['recipient_answer_id'],r['other_answer_id']
        cell='|'.join(str(r[k]) for k in ('family','variant','recipient_condition','direction'))
        m=dict(directed_id=r['directed_id'],group_id=r['group_id'],cell=cell,
               ce_damage=lse(a)-a[answer]-lse(z)+z[answer],
               margin_change=float((a[answer]-a[other])-(z[answer]-z[other])),
               correct=bool(a[answer]>a[other]),vocab_rms=float(np.sqrt(np.mean((a-z)**2))),
               maximum_later_correction_relative=float(np.max(relative[i])),
               frozen_ce_damage=r['arms']['joint']['correct_ce']-r['replay']['correct_ce'])
        rows.append(m);grouped[cell].append(m)
    reports={};joint_pass=[];filler_pass=[]
    for key,items in sorted(grouped.items()):
        assert len(items)==len({r['group_id'] for r in items})==72
        is_joint=key.startswith('selector_payload')
        frozen=original['joint_diagonal']['FIT|'+key] if is_joint else original['controls']['FIT|'+key+'|joint']
        vocabulary_scale=frozen['vocabulary_scale'] if is_joint else frozen['scales']['vocabulary']
        report=dict(groups=72,mean_ce_damage=float(np.mean([r['ce_damage'] for r in items])),
                    frozen_mean_ce_damage=float(np.mean([r['frozen_ce_damage'] for r in items])),
                    median_vocab_rms=float(np.median([r['vocab_rms'] for r in items])),
                    vocabulary_bar=.25*vocabulary_scale,
                    median_absolute_margin_change=float(np.median([abs(r['margin_change']) for r in items])),
                    correct_fraction=float(np.mean([r['correct'] for r in items])))
        passed=report['mean_ce_damage']<=.10 and report['median_vocab_rms']<=report['vocabulary_bar']
        if not is_joint:
            report['margin_bar']=.25*frozen['scales']['margin']
            passed &= report['median_absolute_margin_change']<=report['margin_bar'] and report['correct_fraction']>=.75
        report['passed']=bool(passed);reports[key]=report
        (joint_pass if is_joint else filler_pass).append(bool(passed))
    instrument=counts==[81,2592] and observed==5184 and active>=.75
    instrument &= all(b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())
    instrument &= first_error['relative']<=1e-5 and all(np.isfinite(a).all() for a in saved.values())
    assert len(joint_pass)==4 and len(filler_pass)==8
    files={path.name:dict(sha256=digest(path),bytes=path.stat().st_size) for path in sorted(RAW.iterdir())}
    rowpath=POLY/'INDUCTION_LIVE_CLAMP_V1_ROWS.jsonl'
    with rowpath.open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
    predictions={'pred_a_instrument':bool(instrument),'pred_b_necessary_joint_invariance':all(joint_pass),'pred_c_necessary_filler_control':all(filler_pass)}
    out=dict(terminal='complete' if instrument else 'invalid',predictions=predictions,reports=reports,
             bridges=bridges,first_site_term_bridge=first_error,active_joint_correction_fraction=active,
             observed_layer_transactions=observed,maximum_rounding_residual=maximum_rounding,
             price=dict(forwards=counts[0],sequences=counts[1],fits=0,native_parameters=sum(x.numel() for x in runtime.model.parameters()),weight_saving=0),
             scope='FIT necessary invariance screen of live equality replacement; full R585 identification and active-control coverage untested, no SELECT/FINAL/OOD',
             raw_root=str(RAW),raw_storage_volatile=True,raw_files=files,row_sha256=digest(rowpath),
             runner_sha256=digest(RUNNER),binding_sha256=digest(BINDING),wall_seconds=time.perf_counter()-tic)
    atomic_create_json(OUT,out)
    print(json.dumps({k:out[k] for k in ('predictions','reports','bridges','active_joint_correction_fraction','price','wall_seconds')},indent=2))
    assert instrument


if __name__=='__main__':main()
