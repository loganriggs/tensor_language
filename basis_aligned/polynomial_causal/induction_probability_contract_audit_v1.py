"""Saved-output feasibility audit; no threshold changes and no new model calls."""
import hashlib,json,math
from pathlib import Path
import numpy as np


def probability_loss_feasibility(native_probabilities,required_mean_loss):
    """Since edited probabilities are nonnegative, mean loss <= native mean.

    Passing this necessary condition is not a prediction that an edit succeeds.
    """
    p=np.asarray(native_probabilities,dtype=float)
    if not p.size or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Expected nonempty probabilities in [0,1]')
    if not math.isfinite(required_mean_loss) or required_mean_loss<0:
        raise ValueError('Expected finite nonnegative required loss')
    upper=float(p.mean())
    return dict(upper_bound=upper,required_mean_loss=required_mean_loss,
                feasible=upper>=required_mean_loss)


def main():
    root=Path(__file__).resolve().parents[2];poly=Path(__file__).parent
    old_path=root/'basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_results.json'
    now_path=poly/'INDUCTION_VALUE_PRODUCER_SPLIT_V1_RESULT.json'
    old=json.loads(old_path.read_text());now=json.loads(now_path.read_text())
    previous={r['group_id']:r for r in old['raw_evidence']['group_factorial_measurements'] if r['split']=='FIT'}
    logits=np.asarray(now['reader_logits']['native']);panels={};max_margin=0.
    for ci,name in enumerate(('s0p0','s0p1','s1p0','s1p1')):
        take=[i for i,r in enumerate(now['reader_rows']) if r['condition']==name]
        assert len(take)==72
        prior=[previous[now['reader_rows'][i]['group_id']]['cells'][name] for i in take]
        assert all(p['answer_id']==now['reader_rows'][i]['answer'] for i,p in zip(take,prior))
        margins=np.array([p['correct_margin'] for p in prior]);p=np.exp(-np.array([p['correct_ce'] for p in prior]))
        discrepancy=float(np.max(np.abs(margins-(logits[take,0]-logits[take,1]))));max_margin=max(max_margin,discrepancy)
        report=next(r for r in now['reports'] if r['condition']==name)
        panels[name]=dict(prior_probability_loss_ceiling=probability_loss_feasibility(p,.10),
            prior_correct_fraction=float((margins>0).mean()),new_correct_fraction=report['native_accuracy'],
            prior_capability_accuracy_bar=.75,new_capability_accuracy_bar=.85,
            probability_mean_replay_abs=abs(float(p.mean())-report['native_gold_probability']),
            margin_replay_max_abs=discrepancy,
            full_removal_relative_probability_loss=report['gold_probability_loss']['full']/report['native_gold_probability'])
    checks=dict(impossible_loss_rejected=not probability_loss_feasibility([.03,.04],.1)['feasible'],
        feasible_not_guaranteed=probability_loss_feasibility([.2,.4],.1)['feasible'],
        boundary=probability_loss_feasibility([.1],.1)['feasible'],
        prior_all_panels_already_below_loss_bar=all(not p['prior_probability_loss_ceiling']['feasible'] for p in panels.values()),
        accuracy_replayed=all(p['prior_correct_fraction']==p['new_correct_fraction'] for p in panels.values()))
    assert all(checks.values())
    out=dict(experiment='induction_probability_contract_audit_v1',checks=checks,panels=panels,
        maximum_native_margin_replay_abs=max_margin,model_forwards=0,
        decision='Original .10 absolute probability-loss bar was infeasible from already available native evidence. Preserve failed experiment; no rescoring/promotion. Both branch-fidelity errors independently exceed .10.',
        bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path,now_path,Path(__file__))})
    with (poly/'INDUCTION_PROBABILITY_CONTRACT_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()
