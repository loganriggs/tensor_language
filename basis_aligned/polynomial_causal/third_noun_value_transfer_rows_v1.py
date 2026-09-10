"""Fresh words and a second reflexive reader; no native output access."""
from collections import Counter
from itertools import product
from pathlib import Path
import hashlib,json


def build():
    import tiktoken
    enc=tiktoken.get_encoding('gpt2');corners=list(product((-1,1),repeat=5))
    banks={'male':(('husband','husbands'),('gentleman','gentlemen'),('monk','monks'),('dad','dads')),
           'female':(('queen','queens'),('woman','women'),('girl','girls'),('mother','mothers'))}
    things=(('rock','rocks'),('lamp','lamps'),('chair','chairs'),('desk','desks'))
    token=lambda word:enc.encode(' '+word)
    assert all(len(token(x))==1 for pairs in list(banks.values())+[things] for pair in pairs for x in pair)
    worlds=[]
    for panel,bank in banks.items():
        for i,core in enumerate(bank):
            human=bank[(i+1)%4];thing=things[i]
            for action in ('defend','introduce'):
                wid=f'{panel}:{core[0]}:{action}';rows=[]
                for corner in corners:
                    c,s,o,a,h=corner
                    text=f'The {core[s==1]} {"promised" if c==1 else "persuaded"} the {core[o==1]} near the {(human if h==1 else thing)[a==1]} to {action}'
                    ids=enc.encode(text);assert len(ids)==10
                    rows.append({'row_id':f'transfer:{wid}:{corner}','factors':corner,'text':text,'ids':ids,'semantic_position':9})
                index={tuple(r['factors']):r for r in rows}
                for c,a,h in product((-1,1),repeat=3):
                    assert Counter(index[c,1,-1,a,h]['ids'])==Counter(index[c,-1,1,a,h]['ids'])
                worlds.append({'world_id':wid,'panel':panel,'foil':'himself' if panel=='male' else 'herself','rows':rows})
    p=Path(__file__).resolve().parent;old=json.loads((p/'THIRD_NOUN_ANIMACY_V1_ROWS.json').read_text())
    oldtext={r['text'] for w in old['worlds'] for r in w['rows']}
    newtext={r['text'] for w in worlds for r in w['rows']};assert len(newtext)==512 and not(oldtext&newtext)
    reference=next(w for w in old['worlds'] if w['world_id']=='king_defend_third')
    return {'worlds':worlds,'reference':dict(reference,panel='reference',foil='himself'),'corners':corners,
        'reader_names':['themselves','himself','herself'],'reader_ids':[token(x)[0] for x in ('themselves','himself','herself')],
        'audit':{'new_worlds':16,'new_rows':512,'parent_full_text_overlap':0,'equal_length_tokens':10,
                 'bag_equal_opposite_subject_object_pairs':128,'reference_rows':32,
                 'preoutcome_token_rejections':['grooms','uncles','nephews']}}


if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=build();r['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (p/'THIRD_NOUN_VALUE_TRANSFER_V1_ROWS.json').open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r['audit'],indent=2))
