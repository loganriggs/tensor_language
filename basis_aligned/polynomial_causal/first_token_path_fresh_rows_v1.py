"""Score-free fresh syntax test, with existing endpoints and city pairs."""
from pathlib import Path
import json,tiktoken
from regional_cue_row_check_v1 import validate
from scalar_new_endpoints_rows_v1 import PAIRS
P=Path(__file__).resolve().parent;enc=tiktoken.get_encoding('gpt2');outfile=P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json';assert not outfile.exists()
seen=set()
for f in P.glob('*ROWS.json'):
 data=json.loads(f.read_text());lists=[data] if isinstance(data,list) else [v for v in data.values() if isinstance(v,list)] if isinstance(data,dict) else []
 for ls in lists:
  for r in ls:
   if isinstance(r,dict) and isinstance(r.get('ids'),list) and all(isinstance(t,int) for t in r['ids']):seen.add(tuple(r['ids']))
old=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'][24:48];rows=[dict(r,row_id=i,donor_id=i^1,family=0,family_name='old_near_anchor') for i,r in enumerate(old)]
for family in range(1,4):
 for variant,cities in enumerate([('Nottingham','Detroit'),('Sheffield','Baltimore')]):
  for concept,(uk,us,stem) in enumerate(PAIRS):
   for side,cue in enumerate(['British','American']):
    city=cities[side];text=[f'Please read this message by a lifelong resident of {city}: "{stem}',f'A person who has always lived in {city} writes the following: "{stem}',f'The author is a lifelong resident of {city}. The note discusses ordinary events and a recent journey. It ends: "{stem}'][family-1];ids=enc.encode(text);assert tuple(ids) not in seen;ui,si=enc.encode(uk),enc.encode(us);assert len(ui)==len(si)==1
    rows.append(dict(row_id=len(rows),donor_id=len(rows)^1,family=family,family_name=['','near_message','near_person','distant_note'][family],variant=variant,concept=concept,uk_token=uk,us_token=us,cue=cue,cue_word=city,city=city,text=text,ids=ids,uk_id=ui[0],us_id=si[0],control_ids=[670,3946]))
checks=validate(rows);outfile.write_text(json.dumps(dict(rows=rows,checks=checks,scope='24reusednear anchors and72fresh fullprefixes; endpoints/cities reused. No modelscorefilter, corpusOOD or newlexicalendpointclaim.'),indent=2)+'\n');print(dict(rows=len(rows),max_tokens=max(len(r['ids']) for r in rows)))
