"""Effect accounting for a producer-context by recipient-context interchange."""
import numpy as np


def analyse(effects):
    """effects[producer O/F, reader O/F, ...]; values are causal output effects."""
    e=np.asarray(effects,dtype=float)
    if e.shape[:2]!=(2,2) or not np.isfinite(e).all():
        raise ValueError('Finite two-by-two effects required')
    e00,e01,e10,e11=e[0,0],e[0,1],e[1,0],e[1,1]
    producer=((e10-e00)+(e11-e01))/2
    reader=((e01-e00)+(e11-e10))/2
    interaction=e11-e10-e01+e00
    change=e11-e00
    relative=lambda a,b:float(np.linalg.norm(a)/np.linalg.norm(b)) if np.linalg.norm(b)>0 else None
    producer_errors=[relative(e10-e11,e11),relative(e01-e00,e00)]
    reader_errors=[relative(e10-e00,e00),relative(e01-e11,e11)]
    closure=float(np.max(abs(producer+reader-change)))
    return {'producer_following_errors':producer_errors,'reader_following_errors':reader_errors,
            'producer_following_passed':all(v is not None and v<=.10 for v in producer_errors),
            'reader_following_passed':all(v is not None and v<=.10 for v in reader_errors),
            'symmetric_producer_change':producer.tolist(),'symmetric_reader_change':reader.tolist(),
            'interaction':interaction.tolist(),'structural_effect_change':change.tolist(),
            'decomposition_absolute_error':closure}


def controls():
    a=np.array([1.,2.,3.]);b=np.array([-2.,1.,4.])
    producer_only=analyse(np.array([[a,a],[b,b]]))
    reader_only=analyse(np.array([[a,b],[a,b]]))
    interacting=analyse(np.array([[a,2*a],[3*a,6*a]]))
    assert producer_only['producer_following_passed'] and not producer_only['reader_following_passed']
    assert reader_only['reader_following_passed'] and not reader_only['producer_following_passed']
    assert np.array_equal(interacting['interaction'],2*a)
    assert not interacting['producer_following_passed'] and not interacting['reader_following_passed']
    assert max(x['decomposition_absolute_error'] for x in (producer_only,reader_only,interacting))==0
    return {'passed':True,'producer_only_identified':True,'reader_only_identified':True,
            'multiplicative_case_rejects_both_single_factor_rules':True,
            'symmetric_change_decomposition_error':0.,
            'scope':'Exact synthetic accounting only; no native interchange has been observed.'}
