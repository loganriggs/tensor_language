"""Frozen novel cue-channel panel; tokenize/check only, no model scores."""
from pathlib import Path
import json,tiktoken
from scalar_producers_context_transfer_rows_v1 import PAIRS
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent

def main():
 enc=tiktoken.get_encoding('gpt2');oldids=set();oldcities=set();sources=[]
 for f in sorted(set(P.glob('REGIONAL*ROWS.json'))|set(P.glob('STRUCTURED_PRODUCER*ROWS.json'))|set(P.glob('SCALAR*ROWS.json'))):
  data=json.loads(f.read_text())
  if not isinstance(data,dict):continue
  sources.append(f.name)
  for r in data.get('rows',data.get('regional',[])):
   if not isinstance(r,dict):continue
   if 'ids' in r:oldids.add(tuple(r['ids']))
   for key in ['city','writer_city','reader_city']:
    if key in r:oldcities.add(r[key])
 citypairs=[('Leeds','Dallas'),('Birmingham','Houston')];assert not ({c for pair in citypairs for c in pair}&oldcities)
 rows=[]
 for family in range(3):
  for variant in range(2):
   for concept,(uk,us,stem) in enumerate(PAIRS):
    ui,si=enc.encode(uk),enc.encode(us);assert len(ui)==len(si)==1
    for side,cue in enumerate(['British','American']):
     if family==0:word=citypairs[variant][side];text=f'The columnist has always lived in {word}. A personal message begins: "{stem}'
     elif family==1:
      word=cue;text=(f'The author is {word}. A personal note begins: "{stem}' if variant==0 else f'The {word} editor follows local spelling. The text reads: "{stem}')
     else:
      word=['UK','US'][side];text=(f'The publisher requires {word} spelling. The next sentence is: "{stem}' if variant==0 else f'According to the {word} style guide, the manuscript reads: "{stem}')
     ids=enc.encode(text);assert tuple(ids) not in oldids
     row=dict(row_id=len(rows),family=family,family_name=['new_cities','nationality','style_rule'][family],variant=variant,concept=concept,cue=cue,cue_word=word,text=text,ids=ids,uk_id=ui[0],us_id=si[0],control_ids=[670,3946],donor_id=len(rows)^1)
     if family==0:row['city']=word
     rows.append(row)
 checks=validate(rows);out=P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(rows=rows,checks=checks,prior_files=sources,scope='72newcontrolled prompts: newcities, explicitnationality, explicitstyle-rule cues; old sixspelling contrasts. One-token paired cuechanges, no modelscorefiltering. Distribution shift relative to circuitdiscovery, not corpus/pretraining-disjoint OOD. Frozen rank64 component remains primary; no newfit or rank selection.'),indent=2)+'\n');print(dict(rows=len(rows),pairs=len(rows)//2,checks=checks,max_tokens=max(len(r['ids']) for r in rows)))
if __name__=='__main__':main()
