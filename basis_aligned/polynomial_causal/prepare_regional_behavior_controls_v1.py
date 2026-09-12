"""Freeze new task endpoints before model scoring; no label fitting."""
from pathlib import Path
import json,math,tiktoken
P=Path(__file__).resolve().parent;enc=tiktoken.get_encoding('gpt2')
def token(word):
    ids=enc.encode(' '+word);assert len(ids)==1,(word,ids);return ids[0]
families={
'regional':[("My preferred ice cream",'flavour','flavor'),("The car's bright",'colour','color'),('The building is in the town','centre','center'),('They treated the visitor with','honour','honor')],
 'tense':[('Yesterday the clerk','was','is'),('Last year the school','was','is'),('Two days ago the shop','was','is'),('In 1990 the museum','was','is')],
 'number':[('The two cats','are','is'),('Both dogs','are','is'),('These horses','are','is'),('Several birds','are','is')],
 'meaning':[('A triangle has three','sides','wheels'),('Fresh snow is usually','white','black'),('A week has seven','days','months'),('A bicycle has two','wheels','wings')]}
rows=[]
for family,(name,cases) in enumerate(families.items()):
    for case,(tail,a,b) in enumerate(cases):
        for cue in ('British','American'):
            text=f'A {cue} journalist notes: {tail}'
            rows.append(dict(row_id=len(rows),family=family,family_name=name,concept=case,cue=cue,text=text,ids=enc.encode(text),uk_id=token(a),us_id=token(b),control_ids=[token('good'),token('bad')]))
for i in range(0,len(rows),2):
    a,b=rows[i:i+2];assert len(a['ids'])==len(b['ids']);assert sum(x!=y for x,y in zip(a['ids'],b['ids']))==1
old=set()
for name in ('REGIONAL_SOURCE_BLOCK_V2_ROWS.json','REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json'):
    old.update(x['text'] for x in json.loads((P/name).read_text())['rows'])
assert not old.intersection(x['text'] for x in rows)
lengths={n:sum(len(r['ids'])==n for r in rows) for n in sorted(set(len(r['ids']) for r in rows))}
out=P/'REGIONAL_BEHAVIOR_CONTROLS_V1_ROWS.json';assert not out.exists();result=dict(rows=rows,length_counts=lengths,body_forwards=sum(math.ceil(v/8) for v in lengths.values()),scope='Fresh task prefixes. Regional target words overlap earlier panels; controls test new behaviors. Correct/foil IDs reuse uk_id/us_id schema only; controls have same correct answer under both cues. No model scores used.')
out.write_text(json.dumps(result,indent=2)+'\n');print(lengths,result['body_forwards'])
