"""Exploratory second-reader test, fixed source-token identities, no vocabulary fit."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from induction_context_transport_v2 import digest,loss_terms


def main():
    p=Path(__file__).parent
    result=json.loads((p/'INDUCTION_LIVE_CLAMP_V1_RESULT.json').read_text())
    root=Path(result['raw_root'])
    for name in ('native_logits.npy','live_logits.npy'):
        assert digest(root/name)==result['raw_files'][name]['sha256']
    base=np.load(root/'native_logits.npy',mmap_mode='r');live=np.load(root/'live_logits.npy',mmap_mode='r')
    env=json.loads((p/'INDUCTION_R594_MANAGED_RESULT.json').read_text())
    old=Path(env['raw_root'])/'induction_centered_fixed_geometry_rung594_evidence/FIT'
    receipt=json.loads((p/'induction_centered_fixed_geometry_rung594_receipt.json').read_text())
    for name in ('directed_records.jsonl','endpoint_records.jsonl','endpoint_logits.npy','logit_differences.npy'):
        assert digest(old/name)==receipt['evidence_files']['FIT/'+name]['sha256']
    endpoints={r['endpoint_id']:r for r in map(json.loads,(old/'endpoint_records.jsonl').open())}
    allrows=list(map(json.loads,(old/'directed_records.jsonl').open()))
    oldloc={r['directed_id']:i for i,r in enumerate(allrows)}
    newrows=list(map(json.loads,(p/'INDUCTION_LIVE_CLAMP_V1_ROWS.jsonl').open()))
    native=np.load(old/'endpoint_logits.npy',mmap_mode='r');diffs=np.load(old/'logit_differences.npy',mmap_mode='r')
    grouped=defaultdict(list);n=0
    for i,row in enumerate(newrows):
        if not row['cell'].startswith('selector_payload_joint_answer_preserved'):continue
        oi=oldloc[row['directed_id']];r=allrows[oi]
        b=endpoints[r['recipient_endpoint_id']];d=endpoints[r['donor_endpoint_id']]
        qb=b['token_ids'][b['final_position']];qd=d['token_ids'][d['final_position']]
        assert qb!=qd and qb not in (r['recipient_answer_id'],r['other_answer_id'])
        assert qd not in (r['recipient_answer_id'],r['other_answer_id'])
        z=base[i].astype(np.float64);zl=live[i].astype(np.float64)
        zb=native[b['array_index']].astype(np.float64);zd=native[d['array_index']].astype(np.float64)
        assert np.array_equal(z,zb)
        denominator=float((zd[qd]-zd[qb])-(zb[qd]-zb[qb]))
        frozen=np.asarray(diffs[oi,3],dtype=np.float64)
        item=dict(native_source_margin_change=denominator,
                  live_source_margin_change=float((zl[qd]-zl[qb])-(z[qd]-z[qb])),
                  frozen_source_margin_change=float(frozen[qd]-frozen[qb]),
                  live_loss=loss_terms(z,zl-z,r['recipient_answer_id']),
                  frozen_loss=loss_terms(z,frozen,r['recipient_answer_id']))
        grouped[row['cell']].append(item);n+=1
    reports={}
    for key,items in sorted(grouped.items()):
        assert len(items)==72
        denominator=float(np.mean([r['native_source_margin_change'] for r in items]))
        report=dict(groups=72,mean_native_source_margin_change=denominator,
                    positive_native_fraction=float(np.mean([r['native_source_margin_change']>0 for r in items])))
        for mode in ('live','frozen'):
            effect=float(np.mean([r[mode+'_source_margin_change'] for r in items]))
            positive=float(np.mean([r[mode+'_source_margin_change']>0 for r in items]))
            recovery=effect/denominator if abs(denominator)>1e-12 else None
            report[mode]=dict(mean_source_margin_change=effect,positive_effect_fraction=positive,
                              mean_recovery=recovery,
                              carry_viable=bool(denominator>=.10 and recovery>=.30 and positive>=.75),
                              **{'mean_'+k:float(np.mean([r[mode+'_loss'][k] for r in items]))
                                 for k in ('ce_damage','linear_loss_change','kl')})
        reports[key]=report
    assert n==288 and len(reports)==4
    out=dict(experiment='induction_source_identity_reader_v1',post_result_exploratory=True,
             rows=n,model_forwards=0,reports=reports,
             source_identity_carry_viable=all(r['live']['carry_viable'] for r in reports.values()),
             no_downstream_module_identified=True,original_verdicts_preserved=True,
             source_sha256=digest(__file__),live_result_sha256=digest(p/'INDUCTION_LIVE_CLAMP_V1_RESULT.json'))
    with (p/'INDUCTION_SOURCE_IDENTITY_READER_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
