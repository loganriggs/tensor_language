"""Post-result grouped effects and endpoint-space geometry; no native gain fit."""
import json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    source=ROOT/'CORRELATIVE_VALUE_SOURCE_V1_RESULT.json';out=ROOT/'CORRELATIVE_VALUE_SOURCE_AUDIT_V1_RESULT.json'
    assert not out.exists();r=json.loads(source.read_text());rng=np.random.default_rng(9111284);panels={}
    for name in ('A1','A2'):
        p=r['reports'][name];rows=p['rows'];groups=sorted({row['group_id'] for row in rows})
        idx=[np.array([i for i,row in enumerate(rows) if row['group_id']==group]) for group in groups]
        counts=np.array([len(i) for i in idx]);draws=rng.integers(0,len(groups),size=(4000,len(groups)))
        def sum_draw(values):return np.array([np.asarray(values)[i].sum() for i in idx])[draws].sum(1)
        def mean_ci(values):return np.quantile(sum_draw(values)/counts[draws].sum(1),[.025,.975]).tolist()
        base=np.array(p['native_base_margin']);den=base+np.array(p['native_donor_margin']);ef=p['effects']
        gold=np.array(ef['full']['centered_effect_squared_per_row']);G=gold.sum()
        arms={};dots={};norms={}
        for arm in ('full','first','local'):
            recovery=(base-np.array(ef[arm]['base_oriented_margin']))/den
            err=np.array(ef[arm]['full_effect_error_squared_per_row'])
            arms[arm]={'raw_recovery':float(recovery.mean()),'recovery_ci95':mean_ci(recovery),
                       'effect_error_ci95':np.quantile(np.sqrt(sum_draw(err)/sum_draw(gold)),[.025,.975]).tolist()}
            if arm!='full':
                A=np.array(ef[arm]['centered_effect_squared_per_row']).sum();dot=(G+A-err.sum())/2
                norms[arm]=A;dots[arm]=dot;cosine=dot/np.sqrt(G*A)
                assert abs(cosine)<=1+1e-10
                arms[arm]['endpoint_rescaling']={'optimal_scalar':float(dot/A),'cosine_to_full':float(cosine),
                       'minimum_relative_error':float(np.sqrt(max(0,1-cosine*cosine)))}
        I=np.array(p['interaction_squared_per_row']).sum()
        cross=(I-G-norms['first']-norms['local']+2*dots['first']+2*dots['local'])/2
        gram=np.array([[norms['first'],cross],[cross,norms['local']]])
        eig=np.linalg.eigvalsh(gram);assert eig.min()>0
        rhs=np.array([dots['first'],dots['local']]);coef=np.linalg.solve(gram,rhs)
        residual=G-rhs@coef;assert residual>=-1e-9*G
        mfull=np.array(ef['full']['base_oriented_margin']);mf=np.array(ef['first']['base_oriented_margin']);ml=np.array(ef['local']['base_oriented_margin'])
        margin_interaction=mfull-mf-ml+base
        panels[name]={'rows':len(rows),'groups':len(groups),'arms':arms,
                      'task_margin_interaction_relative':float(np.linalg.norm(margin_interaction)/np.linalg.norm(mfull-base)),
                      'full_vocabulary_interaction_relative':float(np.sqrt(I/G)),
                      'full_vocabulary_interaction_ci95':np.quantile(np.sqrt(sum_draw(p['interaction_squared_per_row'])/sum_draw(gold)),[.025,.975]).tolist(),
                      'raw_recovery_additivity_difference':arms['full']['raw_recovery']-arms['first']['raw_recovery']-arms['local']['raw_recovery'],
                      'two_endpoint_vectors':{'coefficients':coef.tolist(),'minimum_relative_error':float(np.sqrt(max(0,residual/G))),
                                              'gram_condition_number':float(eig.max()/eig.min())}}
    # Independent planted vector oracle for recovering dot products and the
    # two-vector projection residual from the saved sufficient statistics.
    a=np.array([1.,2.,-1.,.5]);b=np.array([-.5,1.,2.,1.]);f=np.array([2.,1.,1.,3.])
    norm=lambda x:float(x@x)
    ga,gb,gf=norm(a),norm(b),norm(f);fa=(gf+ga-norm(f-a))/2;fb=(gf+gb-norm(f-b))/2
    ab=(norm(f-a-b)-gf-ga-gb+2*fa+2*fb)/2
    q=np.linalg.solve(np.array([[ga,ab],[ab,gb]]),[fa,fb]);pred=gf-np.array([fa,fb])@q
    oracle=norm(f-q[0]*a-q[1]*b);assert abs(pred-oracle)<1e-12
    result={'schema':'correlative.value_source_audit.v1','panels':panels,'bootstrap_draws':4000,'bootstrap_seed':9111284,
            'vector_oracle_abs_error':abs(pred-oracle),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'Post-result statistics on16authored groups per panel. Endpoint-vector rescaling bounds are geometry of already observed effects; they do not bound nonlinear native input-dose changes, arbitrary context-dependent adapters or undiscovered programs. No new acceptance gates or gain rescue.'}
    with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
