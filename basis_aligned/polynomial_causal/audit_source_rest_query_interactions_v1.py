"""Exact finite-cohort source-versus-rest interaction identity, no new model run."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'BILIN18_L9_QUERY_SOURCE_ATLAS_V1_RESULT.json'
OUT=ROOT/'SOURCE_REST_QUERY_INTERACTIONS_V1_AUDIT.json'


def norm(xs):return math.sqrt(sum(x*x for x in xs))


def audit(result):
    assert result['predictions']['pred_a_instrument']
    panels={};closure=0.
    for name,report in result['reports'].items():
        rows=report['rows'];total=[r['native_contrast']-r['arm_contrasts']['zero'] for r in rows]
        denominator=norm(total);sources={}
        for label in result['source_labels']:
            standalone=[];rest=[];interaction=[];conditional=[]
            for r in rows:
                f=r['native_contrast'];a=r['arm_contrasts'];zero=a['zero']
                s=a['only_'+label]-zero;t=a['omit_'+label]-zero;i=f-a['only_'+label]-a['omit_'+label]+zero
                standalone.append(s);rest.append(t);interaction.append(i);conditional.append(f-a['omit_'+label])
                closure=max(closure,abs(f-zero-s-t-i),abs((f-a['omit_'+label])-s-i))
            sources[label]={'standalone_norm':norm(standalone),'conditional_with_rest_norm':norm(conditional),'interaction_norm':norm(interaction),'interaction_relative_to_all_query_margin_effect':norm(interaction)/denominator if denominator>1e-8 else None,'interaction_by_row':interaction,'standalone_by_row':standalone,'conditional_by_row':conditional}
        # Separately test a naive sum of all 19 leave-one-out effects. It is not
        # mathematically expected to equal the joint cut in a nonlinear network.
        naive=[sum(r['native_contrast']-r['arm_contrasts']['omit_'+label] for label in result['source_labels']) for r in rows]
        panels[name]={'pair_count':len(rows),'all_query_margin_effect_norm':denominator,'sources':sources,'naive_sum_omissions_relative_error':norm([a-b for a,b in zip(naive,total)])/denominator if denominator>1e-8 else None}
    return {'passed':closure<=1e-12,'identity_max_abs_error':closure,'panels':panels,'model_forwards':0,'fits':0,'scope':'Post-atlas descriptive exact four-cell algebra on opened paired answer-margin contrasts. Each source/rest partition differs; not a19-player Shapley value, population/full-logit bound, source ranking certificate or additive circuit decomposition.'}


if __name__=='__main__':
    result=audit(json.loads(SOURCE.read_text()));assert result['passed']
    result['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':result['passed'],'identity_max_abs_error':result['identity_max_abs_error'],'naive_sum_errors':{k:v['naive_sum_omissions_relative_error'] for k,v in result['panels'].items()}},indent=2))
