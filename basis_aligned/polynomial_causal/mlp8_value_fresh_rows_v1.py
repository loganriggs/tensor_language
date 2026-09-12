"""Score-free full-prefix confirmation; endpoints reused, cities not globally new."""
from pathlib import Path
import json,tiktoken
from regional_cue_row_check_v1 import validate
from scalar_new_endpoints_rows_v1 import PAIRS
P=Path(__file__).resolve().parent
enc=tiktoken.get_encoding('gpt2')
old=set()
for f in P.glob('*ROWS.json'):
 if f.name=='MLP8_VALUE_FRESH_V1_ROWS.json':continue
 data=json.loads(f.read_text())
 for rr in ([data] if isinstance(data,list) else [v for v in data.values() if isinstance(v,list)] if isinstance(data,dict) else []):
  for r in rr:
   if isinstance(r,dict) and 'ids' in r and all(isinstance(x,int) for x in r['ids']):old.add(tuple(r['ids']))
rows=[]
for family in range(3):
 for variant,cities in enumerate([('Nottingham','Detroit'),('Sheffield','Baltimore')]):
  for concept,(uk,us,stem) in enumerate(PAIRS):
   for side,cue in enumerate(['British','American']):
    city=cities[side]
    text=[f'Writing from {city}, a resident sends this note: "{stem}',f'Here is a note from a lifelong resident of {city}: "{stem}',f'The writer grew up in {city}. After discussing the weather and the journey, the writer adds: "{stem}'][family]
    ids=enc.encode(text);assert tuple(ids) not in old
    ui,si=enc.encode(uk),enc.encode(us);assert len(ui)==len(si)==1
    rows.append(dict(row_id=len(rows),family=family,family_name=['fronted_city','near_quote_city','distant_city'][family],variant=variant,concept=concept,uk_token=uk,us_token=us,cue=cue,cue_word=city,city=city,text=text,ids=ids,uk_id=ui[0],us_id=si[0],control_ids=[670,3946],donor_id=len(rows)^1))
checks=validate(rows)
out=P/'MLP8_VALUE_FRESH_V1_ROWS.json';assert not out.exists()
out.write_text(json.dumps(dict(rows=rows,checks=checks,scope='New full prefixes and cue constructions, frozen phi4; endpoints reused; cities may appear in prior broader component panels. No corpus/pretraining OOD claim.'),indent=2)+'\n')
print(dict(rows=len(rows),max_tokens=max(len(r['ids']) for r in rows),checks=checks))
