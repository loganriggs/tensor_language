"""CPU audit of saved physical additions and unchanged R594 failure cells.

This is an independent arithmetic/scoring implementation, not a separate-agent review.
"""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from fp32_add_observation_v1 import observe


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def main():
    poly=Path(__file__).parent
    envelope=json.loads((poly/'INDUCTION_R594_MANAGED_RESULT.json').read_text())
    result=envelope['result'];root=Path(envelope['raw_root'])
    receipt=json.loads(Path(envelope['raw_receipt']).read_text())
    assert digest(envelope['raw_result'])==receipt['result_sha256']==envelope['raw_result_sha256']
    assert digest(envelope['raw_receipt'])==envelope['raw_receipt_sha256']
    evidence=root/'induction_centered_fixed_geometry_rung594_evidence';total_bytes=0
    for name,meta in receipt['evidence_files'].items():
        p=evidence/name
        assert p.stat().st_size==meta['byte_length'] and digest(p)==meta['sha256'],name
        total_bytes+=meta['byte_length']
    phase=evidence/'FIT'
    names=('layer_before.npy','layer_total.npy','layer_after.npy','layer_delta.npy','planned_component_deltas.npy')
    arrays={name:np.load(phase/name,mmap_mode='r',allow_pickle=False) for name in names}
    assert arrays['layer_before.npy'].shape==(3744,4,3,1152)
    maximum_residual=0.;old_failures=0;transactions=0;nonzero=0
    for start in range(0,3744,16):
        a={k:np.asarray(v[start:start+16]) for k,v in arrays.items()}
        r=observe(a['layer_before.npy'],a['layer_total.npy'],a['layer_after.npy'])
        assert r['passed'] and np.array_equal(r['actual_delta'],a['layer_delta.npy'])
        plan=a['planned_component_deltas.npy']
        expected=np.stack((plan[:,:,0],plan[:,:,1],np.add(plan[:,:,2],plan[:,:,3],dtype=np.float32)),axis=2)
        assert np.array_equal(expected.view(np.uint32),a['layer_total.npy'].view(np.uint32))
        legacy=(a['layer_after.npy']-a['layer_before.npy']).astype(np.float64)-a['layer_total.npy'].astype(np.float64)
        old_failures+=int((np.max(np.abs(legacy),axis=-1)>1e-5).sum())
        nonzero+=int(np.any(a['layer_total.npy']!=0,axis=-1).sum())
        transactions+=int(np.prod(a['layer_total.npy'].shape[:-1]))
        maximum_residual=max(maximum_residual,r['maximum_rounding_residual'])
    rows=[json.loads(line) for line in (phase/'directed_records.jsonl').read_text().splitlines()]
    score=result['split_scores']['FIT'];grouped=defaultdict(list)
    for row in rows:
        key='|'.join(str(row[k]) for k in ('split','family','variant','recipient_condition','direction'))
        grouped[key].append(row)
    diagonals={};controls={}
    for key,registered in score['joint_diagonal'].items():
        selected=grouped[key];assert len(selected)==72
        ce=float(np.mean([r['arms']['joint']['correct_ce']-r['replay']['correct_ce'] for r in selected]))
        vr=float(np.median([r['arms']['joint']['vocab_rms'] for r in selected]))
        assert abs(ce-registered['mean_joint_minus_replay_ce'])<1e-12
        assert abs(vr-registered['median_joint_vocab_rms'])<1e-12
        diagonals[key]=dict(mean_ce_damage=ce,ce_bar=.10,ce_passed=ce<=.10,
            vocabulary_rms=vr,vocabulary_bar=.25*registered['vocabulary_scale'],
            vocabulary_passed=vr<=.25*registered['vocabulary_scale'],groups=len(selected))
    for key in result['failure_classes']['broad_contextual_equality_write']:
        stem,arm=key.rsplit('|',1);selected=grouped[stem];r=score['controls'][key]
        ce=float(np.mean([x['arms'][arm]['correct_ce']-x['replay']['correct_ce'] for x in selected]))
        assert abs(ce-r['mean_ce_change'])<1e-12
        controls[key]=dict(mean_ce_damage=ce,active_fraction=r['active_fraction'],
            margin_passed=r['median_absolute_margin_change']<=.25*r['scales']['margin'],
            vocabulary_passed=r['median_vocab_rms']<=.25*r['scales']['vocabulary'],ce_passed=ce<=.10)
    target_summary={}
    for family in ('two_valid_sources_selector_swap','payload_swap_match_preserved','match_break_payload_preserved'):
        target_summary[family]={}
        for arm in ('score','payload','joint'):
            selected=[v for k,v in score['targets'].items() if k.split('|')[1]==family and k.endswith('|'+arm)]
            if selected:
                target_summary[family][arm]=dict(cells=len(selected),all_passed=all(v['passes'] for v in selected),
                    mean_recovery_range=[min(v['recovery']['mean_recovery'] for v in selected),max(v['recovery']['mean_recovery'] for v in selected)])
    out=dict(experiment='audit_induction_r594_saved_v1',all_raw_hashes_passed=True,
        raw_files=len(receipt['evidence_files']),raw_bytes=total_bytes,
        physical_transactions=transactions,nonzero_planned_transactions=nonzero,
        all_rounded_poststates_bitwise=True,all_measured_deltas_exact=True,all_registered_totals_bitwise=True,
        old_absolute_check_failed_transactions=old_failures,maximum_representability_residual=maximum_residual,
        target_summary=target_summary,joint_diagonal=diagonals,failing_controls=controls,
        active_families_by_coverage_key={k:v['active_families'] for k,v in score['coverage'].items()},
        all_original_gates_preserved=True,model_forwards=0,raw_storage_volatile=True,
        source_sha256=digest(__file__),result_sha256=envelope['raw_result_sha256'],receipt_sha256=envelope['raw_receipt_sha256'])
    with (poly/'INDUCTION_R594_SAVED_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('active_families_by_coverage_key','joint_diagonal','failing_controls')},indent=2))


if __name__=='__main__':main()
