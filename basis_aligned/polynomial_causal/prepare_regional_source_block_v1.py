"""Freeze spelling cue pairs without querying the model."""
from pathlib import Path
import json,tiktoken
P=Path(__file__).resolve().parent
enc=tiktoken.get_encoding('gpt2')
concepts=[('colour','color',"described the paint's",'blue','red'),('favourite','favorite','said this was her','best','worst'),('centre','center','met them in the town','north','south'),('recognise','recognize','could immediately','remember','forget'),('realised','realized','suddenly','knew','thought'),('behaviour','behavior','disliked his','actions','words'),('honour','honor','said it was an','pride','shame'),('travelling','traveling','enjoyed','walking','running')]
def token(word):
    ids=enc.encode(' '+word);assert len(ids)==1;return ids[0]
rows=[]
for family,template in enumerate(('The {cue} journalist {tail}','In a {cue} newspaper, the writer {tail}')):
    for concept,(uk,us,tail,control1,control2) in enumerate(concepts):
        for cue in ('British','American'):
            text=template.format(cue=cue,tail=tail);ids=enc.encode(text)
            rows.append(dict(row_id=len(rows),family=family,concept=concept,cue=cue,text=text,ids=ids,uk_id=token(uk),us_id=token(us),control_ids=[token(control1),token(control2)]))
assert len(rows)==32 and all(len(r['ids'])>=4 for r in rows)
out=P/'REGIONAL_SOURCE_BLOCK_V1_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(rows=rows,scope='Prospective templates, fixed tokens; no native scores inspected during construction.'),indent=2)+'\n')
print({n:sum(len(r['ids'])==n for r in rows) for n in sorted(set(len(r['ids']) for r in rows))})
