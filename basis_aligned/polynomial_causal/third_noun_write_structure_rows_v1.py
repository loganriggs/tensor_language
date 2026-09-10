"""Frozen structural changes to an existing lexical cube; no native model access."""
from collections import Counter
from itertools import product
from pathlib import Path
import hashlib
import json
import tiktoken


def positions(world):
    rows=world['rows'];index={tuple(r['factors']):r for r in rows}
    length=len(rows[0]['ids']);supports={}
    assert all(len(r['ids'])==length and r['semantic_position']==length-1 for r in rows)
    for factor in (2,4):
        changed=set()
        for corner,row in index.items():
            flipped=list(corner);flipped[factor]*=-1
            changed.update(i for i,(a,b) in enumerate(zip(row['ids'],index[tuple(flipped)]['ids'])) if a!=b)
        assert len(changed)==1
        supports[factor]=next(iter(changed))
    assert supports[2]!=supports[4]
    for c,a,h in product((-1,1),repeat=3):
        assert Counter(index[c,1,-1,a,h]['ids'])==Counter(index[c,-1,1,a,h]['ids'])
    world.update(length=length,interaction_start=max(supports.values()),
                 object_position=supports[2],category_position=supports[4])
    return world


def build():
    directory=Path(__file__).resolve().parent
    source=json.loads((directory/'THIRD_NOUN_VALUE_TRANSFER_V1_ROWS.json').read_text())
    enc=tiktoken.get_encoding('gpt2');worlds=[]
    for layout in ('intervening_pp','fronted_pp'):
        for old in source['worlds']:
            wid=f'{layout}:{old["world_id"]}';rows=[]
            for oldrow in old['rows']:
                parts=oldrow['text'].split()
                _,subject,verb,_,obj,_,_,attractor,_,action=parts
                text=(f'The {subject} {verb} the {obj} in the room near the {attractor} to {action}'
                      if layout=='intervening_pp' else
                      f'Near the {attractor}, the {subject} {verb} the {obj} to {action}')
                ids=enc.encode(text)
                rows.append(dict(oldrow,row_id=f'structure:{wid}:{oldrow["factors"]}',text=text,ids=ids,semantic_position=len(ids)-1))
            worlds.append(positions(dict(old,world_id=wid,layout=layout,rows=rows)))
    reference=positions(dict(source['worlds'][0],layout='reference'))
    texts={r['text'] for w in worlds for r in w['rows']}
    parenttexts={r['text'] for w in source['worlds'] for r in w['rows']}
    assert len(texts)==1024 and not texts&parenttexts
    audit={'new_worlds':32,'new_rows':1024,'parent_full_text_overlap':0,
           'layouts':{layout:{key:next(w[key] for w in worlds if w['layout']==layout)
                    for key in ('length','object_position','category_position','interaction_start')}
                     for layout in ('intervening_pp','fronted_pp')},
           'equal_bag_opposite_subject_object_pairs':256}
    return {**{key:source[key] for key in ('corners','reader_names','reader_ids')},
            'worlds':worlds,'reference':reference,'audit':audit}


if __name__=='__main__':
    data=build();data['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (Path(__file__).resolve().parent/'THIRD_NOUN_WRITE_STRUCTURE_V1_ROWS.json').open('x') as handle:
        json.dump(data,handle,indent=2);handle.write('\n')
    print(json.dumps(data['audit'],indent=2))
