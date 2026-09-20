"""Freeze 48 new combinations before hybrid-output validation."""
import json,hashlib
from pathlib import Path
import tiktoken

def main():
    enc=tiktoken.get_encoding('gpt2');rows=[]
    nouns=[('guard','guards'),('judge','judges'),('clerk','clerks'),('coach','coaches'),('chef','chefs'),('cook','cooks')]
    templates=[('relative_called_station','The {s} whom the {a} called near the station'),
               ('relative_noticed_office','The {s} that the {a} noticed outside the office'),
               ('front_station_greeted_yesterday','Beside the station the {s} who greeted the {a} yesterday'),
               ('front_office_helped_today','Outside the office the {s} whom the {a} helped today')]
    for name,template in templates:
        group=[]
        for i,pair in enumerate(nouns):
            for plural in [False,True]:
                subject=pair[int(plural)];attractor=nouns[(i+2)%len(nouns)][int(not plural)]
                text=template.format(s=subject,a=attractor);ids=enc.encode(text);sid=enc.encode(' '+subject)
                assert len(sid)==1 and ids.count(sid[0])==1
                group.append(dict(template=name,text=text,token_ids=ids,subject_position=ids.index(sid[0]),readout_position=len(ids)-1,
                                  answer_ids=[389,318] if plural else [318,389],family=name+('|plural' if plural else '|singular')))
        assert len({len(r['token_ids']) for r in group})==1
        rows+=group
    parent=Path(__file__).parent
    old=json.loads((parent/'SUBJECT_HYBRID_FRESH_V687_ROWS.json').read_text())
    assert not {r['text'] for r in old}&{r['text'] for r in rows}
    path=parent/'SUBJECT_MODAL_PATH_FRESH_V691_ROWS.json';path.write_text(json.dumps(rows,indent=2)+'\n')
    print(len(rows),hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
