"""Fresh third-noun factorial to distinguish the predeclared two-slot alternatives."""
from collections import Counter
from itertools import product
from pathlib import Path
import hashlib,json
import numpy as np

FACTORS=('c','s','o','a','h')
CORNERS=tuple(product((-1,1),repeat=5))
LEXICON=(
    (('king','kings'),('brother','brothers'),('crate','crates')),
    (('man','men'),('boy','boys'),('box','boxes')),
    (('father','fathers'),('son','sons'),('cart','carts')),
    (('actor','actors'),('prince','princes'),('book','books')))


def features(corner):
    return tuple(np.prod([corner[i] for i in range(5) if mask&(1<<i)],dtype=int) for mask in range(32))


def labels(c,s,o,a,h):
    return {'controller':s if c==1 else o,'object':o,'nearest_noun':a,
            'nearest_human':a if h==1 else o}


def build():
    import tiktoken
    enc=tiktoken.get_encoding('gpt2');worlds=[]
    for i,(core,human,thing) in enumerate(LEXICON):
        for action in ('defend','introduce'):
            world={'world_id':f'{core[0]}_{action}_third','rows':[]}
            for corner in CORNERS:
                c,s,o,a,h=corner;verb='promised' if c==1 else 'persuaded'
                subject=core[s==1];obj=core[o==1];attractor=(human if h==1 else thing)[a==1]
                assert all(len(enc.encode(' '+v))==1 for v in (subject,obj,attractor,verb,action))
                text=f'The {subject} {verb} the {obj} near the {attractor} to {action}'
                tokens=enc.encode(text)
                world['rows'].append({'row_id':f'third:{world["world_id"]}:{corner}',
                    'factors':corner,'labels':labels(*corner),'text':text,'ids':tokens,'semantic_position':len(tokens)-1})
            assert len({len(r['ids']) for r in world['rows']})==1
            index={tuple(r['factors']):r for r in world['rows']}
            for c,a,h in product((-1,1),repeat=3):
                assert Counter(index[c,1,-1,a,h]['ids'])==Counter(index[c,-1,1,a,h]['ids'])
            worlds.append(world)
    basis=np.array([features(c) for c in CORNERS],dtype=int)
    assert np.array_equal(basis.T@basis,32*np.eye(32,dtype=int))
    values=np.array([labels(*c)['nearest_human'] for c in CORNERS]);coef=basis.T@values/32
    expected=np.zeros(32);expected[1<<2]=.5;expected[1<<3]=.5
    expected[(1<<4)|(1<<3)]=.5;expected[(1<<4)|(1<<2)]=-.5
    assert np.array_equal(coef,expected)
    old=json.loads((Path(__file__).resolve().parent/'SUBJECT_OBJECT_CONTROLLER_V1_ROWS.json').read_text())
    oldtext={r['text'] for w in old['worlds'] for r in w['rows']}
    assert not oldtext.intersection(r['text'] for w in worlds for r in w['rows'])
    return {'schema':1,'worlds':worlds,'corners':CORNERS,'factors':FACTORS,
            'terms':[''.join(name for i,name in enumerate(FACTORS) if mask&(1<<i)) or '1' for mask in range(32)],
            'answer_id':enc.encode(' themselves')[0],'foil_id':enc.encode(' himself')[0],
            'audit':{'rows':256,'worlds':8,'equal_length_worlds':8,'opposite_subject_object_bag_matches':64,
                     'prior_text_overlap':0,'basis_orthogonal':True,'nearest_human_polynomial_verified':True}}


if __name__=='__main__':
    root=Path(__file__).resolve().parent;result=build()
    result['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (root/'THIRD_NOUN_ANIMACY_V1_ROWS.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['audit'],indent=2))
