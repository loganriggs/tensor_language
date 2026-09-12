"""Freeze unseen spelling pairs and geographic cue templates before scoring."""
from pathlib import Path
import json,tiktoken,math
P=Path(__file__).resolve().parent;enc=tiktoken.get_encoding('gpt2')
concepts=[('labour','labor','wrote about manual','work','jobs'),('favour','favor','asked for a small','gift','help'),('organised','organized','said the meeting was well','planned','managed'),('analyse','analyze','wanted to','read','write'),('neighbour','neighbor','spoke to a','friend','stranger'),('theatre','theater','went to the','cinema','museum')]
def token(x):
 z=enc.encode(' '+x);assert len(z)==1;return z[0]
rows=[]
for template_id,template in enumerate(('The {city} newspaper reporter {tail}','According to a newspaper from {city}, the reporter {tail}')):
 for city_id,(uk_city,us_city) in enumerate((('London','Boston'),('Edinburgh','Chicago'))):
  family=2*template_id+city_id
  for concept,(uk,us,tail,c1,c2) in enumerate(concepts):
   for cue,city in (('British',uk_city),('American',us_city)):
    text=template.format(city=city,tail=tail);rows.append(dict(row_id=len(rows),family=family,concept=concept,cue=cue,city=city,text=text,ids=enc.encode(text),uk_id=token(uk),us_id=token(us),control_ids=[token(c1),token(c2)]))
assert len(rows)==48
old=json.loads((P/'REGIONAL_SOURCE_BLOCK_V2_ROWS.json').read_text())['rows'];assert not {r['uk_id'] for r in rows}&{r['uk_id'] for r in old}
lengths={n:sum(len(r['ids'])==n for r in rows) for n in sorted(set(len(r['ids']) for r in rows))}
out=P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(rows=rows,length_counts=lengths,body_forwards=sum(math.ceil(n/8) for n in lengths.values()),scope='Unseen output spelling pairs, geographic cues and prompt templates relative to V1/V2. Not a new natural corpus; no model scores used in construction.'),indent=2)+'\n');print(lengths)
