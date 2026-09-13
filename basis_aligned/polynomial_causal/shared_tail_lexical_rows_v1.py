"""Frozen lexical-transfer panel; no model scores used to select examples."""
import json
from pathlib import Path
from tokenizers import Tokenizer
from regional_cue_row_check_v1 import validate

def main():
    p=Path(__file__).resolve().parent
    t=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    pairs=[('civilisation','civilization','The historian studied the rise of ancient'),
           ('modelling','modeling','The engineer specialises in mathematical'),
           ('criticised','criticized','The newspaper strongly'),
           ('organisation','organization','She founded a charitable'),
           ('metres','meters','The tower is fifty'),
           ('colours','colors','The artist mixed several bright')]
    old=json.loads((p/'THREE_TERM_FRESH_TEMPLATE_V1_ROWS.json').read_text())['rows']
    prior_pairs=set();prior_sequences=set();inventories=[]
    for f in p.glob('*ROWS*.json'):
        if f.name.startswith('SHARED_TAIL_LEXICAL'):continue
        try:data=json.loads(f.read_text())
        except ValueError:continue
        rs=data.get('rows',[]) if isinstance(data,dict) else data
        if not isinstance(rs,list):continue
        inventories.append(f.name)
        for row in rs:
            if not isinstance(row,dict):continue
            if 'uk_id' in row and 'us_id' in row:prior_pairs.add((row['uk_id'],row['us_id']))
            ids=row.get('ids')
            if isinstance(ids,list) and all(isinstance(x,int) for x in ids):prior_sequences.add(tuple(ids))
    rows=[]
    for family in range(4):
        for concept,(uk,us,ending) in enumerate(pairs):
            a=t.encode(' '+uk).ids;b=t.encode(' '+us).ids
            assert len(a)==len(b)==1 and (a[0],b[0]) not in prior_pairs
            for side in range(2):
                template=old[family*12+side]['text']
                assert template.endswith('The evening television')
                text=template[:-len('The evening television')]+ending
                ids=t.encode(text).ids;assert tuple(ids) not in prior_sequences
                assert t.encode(text+' '+uk).ids==ids+a and t.encode(text+' '+us).ids==ids+b
                row_id=len(rows)
                rows.append(dict(row_id=row_id,family=family,concept=concept,cue=('British','American')[side],
                    text=text,ids=ids,uk_id=a[0],us_id=b[0],control_ids=old[0]['control_ids'],donor_id=row_id^1))
    validate(rows)
    result=dict(rows=rows,pairs=pairs,checked_inventories=inventories,
                scope='New exact lexical contrasts and token sequences in checked local row inventories; instruction styles reused, some word roots related to prior tests. Not corpus-wide OOD or new semantic tasks.')
    (p/'SHARED_TAIL_LEXICAL_V1_ROWS.json').write_text(json.dumps(result,indent=2)+'\n')
    print({'rows':len(rows),'inventories':len(inventories),'prior_pairs':len(prior_pairs),'prior_sequences':len(prior_sequences)})
if __name__=='__main__':main()
