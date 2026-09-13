from pathlib import Path
import json,tiktoken
from regional_cue_row_check_v1 import validate
from scalar_new_endpoints_rows_v1 import PAIRS
P=Path(__file__).resolve().parent;out=P/'CROSSFIRST_THREE_GROUP_FRESH_V1_ROWS.json';assert not out.exists();enc=tiktoken.get_encoding('gpt2');seen=set()
for f in P.glob('*ROWS.json'):
 try:d=json.loads(f.read_text())
 except (ValueError,OSError):continue
 lists=[d] if isinstance(d,list) else [v for v in d.values() if isinstance(v,list)] if isinstance(d,dict) else []
 for values in lists:
  for r in values:
   if isinstance(r,dict) and isinstance(r.get('ids'),list) and all(isinstance(x,int) for x in r['ids']):seen.add(tuple(r['ids']))
templates=['A reader from {city} sent this reply after reviewing the draft: "{stem}', 'The following words were written by someone who grew up in {city}: "{stem}', 'After returning home to {city}, the writer opened a notebook and wrote: "{stem}', 'The contributor lives in {city}. We have reproduced the wording exactly, including the spelling: "{stem}'];rows=[]
for family,template in enumerate(templates):
 for variant,cities in enumerate([('London','Boston'),('Manchester','Chicago')]):
  for concept,(uk,us,stem) in enumerate(PAIRS):
   for side,cue in enumerate(['British','American']):
    city=cities[side];text=template.format(city=city,stem=stem);ids=enc.encode(text);assert tuple(ids) not in seen;seen.add(tuple(ids));ui,si=enc.encode(uk),enc.encode(us);assert len(ui)==len(si)==1
    rows.append(dict(row_id=len(rows),donor_id=len(rows)^1,family=family,family_name=['reader_reply','grew_up','return_home','exact_spelling'][family],variant=variant,concept=concept,cue=cue,city=city,cue_word=city,uk_token=uk,us_token=us,uk_id=ui[0],us_id=si[0],control_ids=[670,3946],text=text,ids=ids))
checks=validate(rows);result=dict(rows=rows,checks=checks,scope='96newfullprefixes relative to scanned repository rowfiles,4constructions and2citypairs new to this local three-group path screen. Existing six endpoints. No scorefilter/corpusOOD or global unseen-city claim.');out.write_text(json.dumps(result,indent=2)+'\n');print(dict(rows=len(rows),pairs=len(rows)//2,max_tokens=max(len(r['ids']) for r in rows),checks=checks))
