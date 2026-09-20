"""Frozen new constructions for semantic pair support transfer; no model access."""
import json,hashlib
from pathlib import Path
import tiktoken

def main():
    p=Path(__file__).parent;enc=tiktoken.get_encoding('gpt2')
    pairs=[('guard','guards'),('judge','judges'),('clerk','clerks'),('coach','coaches'),('chef','chefs'),('cook','cooks')]
    templates=[('near_greeted','The {s} whom the {a} near the station greeted yesterday'),('outside_called','The {s} that the {a} outside the office called earlier'),('yesterday_helped','Yesterday the {s} whom the {a} near the office helped'),('earlier_noticed','Earlier the {s} that the {a} beside the station noticed')]
    old=set()
    for name in ['SUBJECT_ATTRACTOR_CONTROL_V696_ROWS.json','SUBJECT_CONGRUENT_ATTRACTOR_V697_ROWS.json','SUBJECT_HYBRID_FRESH_V687_ROWS.json']:
        old.update(tuple(r['token_ids']) for r in json.loads((p/name).read_text()))
    result={}
    for panel in ['opposite','congruent']:
        rows=[]
        for template,pattern in templates:
            for i,pair in enumerate(pairs):
                for number in [0,1]:
                    subject=pair[number];attractor=pairs[(i+2)%len(pairs)][number if panel=='congruent' else 1-number]
                    text=pattern.format(s=subject,a=attractor);ids=enc.encode(text)
                    st=enc.encode(' '+subject);at=enc.encode(' '+attractor);assert len(st)==len(at)==1
                    assert ids.count(st[0])==ids.count(at[0])==1
                    spos=ids.index(st[0]);apos=ids.index(at[0]);assert spos<apos<len(ids)-1 and tuple(ids) not in old
                    rows.append(dict(template=template,text=text,token_ids=ids,subject_position=spos,control_position=apos,readout_position=len(ids)-1,answer_ids=[389,318] if number else [318,389],family=template+'|'+('plural' if number else 'singular'),control_number=panel))
        assert len(rows)==48 and len({tuple(r['token_ids']) for r in rows})==48
        out=p/f'SEMANTIC_PORT_FRESH_{panel.upper()}_ROWS.json';out.write_text(json.dumps(rows,indent=2)+'\n');result[panel]=dict(file=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),rows=48)
    (p/'SEMANTIC_PORT_FRESH_ROW_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
