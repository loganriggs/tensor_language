"""A declared noun-slot selector and its complete Boolean factorial domain.

This specifies the hypothesis. It is not an extracted native computation.
"""
from collections import Counter
from fractions import Fraction
from itertools import product
import hashlib,json
import numpy as np

CORNERS=tuple(product((-1,1),repeat=3))  # control verb, subject number, object number
TERMS=('1','c','s','o','cs','co','so','cso')
NOUNS=(('king','kings'),('man','men'),('boy','boys'),('father','fathers'),
       ('brother','brothers'),('son','sons'),('prince','princes'),('actor','actors'))
ACTIONS=('defend','introduce')


def features(c,s,o):return (1,c,s,o,c*s,c*o,s*o,c*s*o)


def selector(c,s,o):return Fraction(s+o+c*(s-o),2)


def design():return np.array([features(*corner) for corner in CORNERS],dtype=float)


def controls():
    coefficients={term:sum(selector(*corner)*features(*corner)[j] for corner in CORNERS)/8
                  for j,term in enumerate(TERMS)}
    expected={'1':0,'c':0,'s':Fraction(1,2),'o':Fraction(1,2),'cs':Fraction(1,2),
              'co':Fraction(-1,2),'so':0,'cso':0}
    assert coefficients==expected
    h=design();assert np.array_equal(h.T@h,8*np.eye(8))
    assert all(selector(c,s,o)==(s if c==1 else o) for c,s,o in CORNERS)
    # Bilinear coefficient matrix on (1,c) x (s,o) has determinant -1/2.
    determinant=Fraction(1,2)*Fraction(-1,2)-Fraction(1,2)*Fraction(1,2)
    assert determinant==Fraction(-1,2)
    return {'passed':True,'coefficients':{k:str(v) for k,v in coefficients.items()},
            'bilinear_matrix_determinant':str(determinant),'bilinear_matrix_rank':2,
            'first_noun_rule_errors':sum(selector(c,s,o)!=s for c,s,o in CORNERS),
            'nearest_noun_rule_errors':sum(selector(c,s,o)!=o for c,s,o in CORNERS),
            'model_forwards':0,'scope':'Exact task specification and finite-grid algebra, not native discovery.'}


def build():
    import tiktoken
    enc=tiktoken.get_encoding('gpt2');worlds=[]
    for ni,nouns in enumerate(NOUNS):
        for action in ACTIONS:
            world={'world_id':f'{nouns[0]}_{action}','split':'fit' if ni%2==0 else 'held','rows':[]}
            for c,s,o in CORNERS:
                verb='promised' if c==1 else 'persuaded'
                subject=nouns[1 if s==1 else 0];obj=nouns[1 if o==1 else 0]
                text=f'The {subject} {verb} the {obj} to {action}'
                tokens=enc.encode(text)
                assert all(len(enc.encode(' '+v))==1 for v in (subject,obj,verb,action))
                world['rows'].append({'row_id':f'controller:{world["world_id"]}:{c}:{s}:{o}',
                    'factors':[c,s,o],'text':text,'ids':tokens,'semantic_position':len(tokens)-1,
                    'expected_plural':int(selector(c,s,o)==1)})
            assert len({len(r['ids']) for r in world['rows']})==1
            index={tuple(r['factors']):r for r in world['rows']}
            for c in (-1,1):
                assert Counter(index[c,1,-1]['ids'])==Counter(index[c,-1,1]['ids'])
            worlds.append(world)
    return {'schema':1,'corners':CORNERS,'terms':TERMS,'worlds':worlds,
            'positive_answer_id':enc.encode(' themselves')[0],'negative_answer_id':enc.encode(' himself')[0],
            'controls':controls(),'audit':{'worlds':16,'rows':128,'fit_worlds':8,'held_worlds':8,
                'equal_length_worlds':16,'opposite_number_token_multiset_matches':32,
                'lexical_preflight_note':'uncle/uncles rejected before native access because uncles is two tokens; son/sons used.'}}


if __name__=='__main__':
    from pathlib import Path
    root=Path(__file__).resolve().parent
    result=build();result['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (root/'SUBJECT_OBJECT_CONTROLLER_V1_ROWS.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'audit':result['audit'],'controls':result['controls']},indent=2))
