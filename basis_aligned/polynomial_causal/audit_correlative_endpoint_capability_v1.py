"""Actual native endpoint audit plus exact counterexample to aggregated capability."""
import hashlib,json
from pathlib import Path
from circuit_endpoint_capability_v1 import summarize


def main():
    p=Path(__file__).resolve().parent;source=p/'CORRELATIVE_MATCHED_CONTEXT_V1_RESULT.json'
    r=json.loads(source.read_text())
    # Two reciprocal cue directions: model always prefers and, by7.5 or2.2.
    # Integer-scaled values give an exact arithmetic witness to positive
    # effect denominators and opposite average axes despite zero correct pairs.
    base=[75,-22];donor=[-22,75]
    toy=summarize(base,donor)
    assert toy['positive_effect_denominators']==2 and toy['both_endpoints_correct']==0
    toy.update({'base_own_answer_margins':base,'donor_own_answer_margins':donor,
                'base_donor_oriented_axis_mean':-sum(base)/2,
                'donor_own_answer_axis_mean':sum(donor)/2,
                'model_prediction':'and for BOTH both and neither inputs'})
    assert summarize([2,3],[4,5])['both_endpoints_correct']==2
    assert summarize([-2],[1])['positive_effect_denominators']==0
    panels={n:summarize(q['base_margin'],q['donor_margin']) for n,q in r['reports'].items()}
    prior=p.parent/'bilinear_quotient/circuits/followups/unit_correlative_recency_v493_result.json'
    result={'schema':'correlative.endpoint_capability.audit.v1','panels':panels,'exact_toy':toy,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'prior_art_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),
        'prior_art_correction':'v493 reports n/dropped and averaged donor-oriented axes, not each endpoint correctness. Those aggregates cannot establish its still-open-correlative tracking claim. New same-template native counterexamples refute universal correct completion; original32row endpoint accuracy cannot be reconstructed from the stored aggregate alone.',
        'scope':'CPU receipt audit, no model forwards. Positive denominator remains useful for effect normalization, not a substitute for correct target behavior.'}
    out=p/'CORRELATIVE_ENDPOINT_CAPABILITY_AUDIT_V1_RESULT.json'
    with out.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps({'panels':panels,'exact_toy':toy},indent=2))


if __name__=='__main__':main()
