"""Build an outcome-blind fresh template/city panel for O framing transfer."""
import hashlib
import json
from pathlib import Path

import tiktoken
from regional_cue_row_check_v1 import validate
from regional_city_article_check_v1 import validate_city_articles

TEMPLATES=[
    'A letter from {city} was preserved by the local history society. Reproduce its closing line exactly: "We invited our new',
    'The radio archive interviewed a lifelong {city} resident. Transcribe the speaker\'s original sentence: "We invited our new',
]
CITY_PAIRS=[('Cambridge','Phoenix'),('Leeds','Chicago')]
ENDPOINTS=[('neighbours','neighbors'),('organise','organize'),('realise','realize'),('labelled','labeled'),('defence','defense'),('metre','meter')]


def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_FRAMING_FRESH_V1_ROWS.json';assert not out.exists()
    tokenizer=tiktoken.get_encoding('gpt2');rows=[]
    for family,template in enumerate(TEMPLATES):
        for pair,(uk_city,us_city) in enumerate(CITY_PAIRS):
            for endpoint,(uk_word,us_word) in enumerate(ENDPOINTS):
                target_ids=[tokenizer.encode(' '+word) for word in [uk_word,us_word]]
                if any(len(x)!=1 for x in target_ids):raise ValueError('Endpoint must be one token')
                for cue,city in [('British',uk_city),('American',us_city)]:
                    text=template.format(city=city);ids=tokenizer.encode(text)
                    row={'family':family,'template':family,'pair':pair,'endpoint':endpoint,'cue':cue,'city':city,'text':text,'ids':ids,'uk_id':target_ids[0][0],'us_id':target_ids[1][0],'control_ids':[670,3946]}
                    row['row_id']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest();rows.append(row)
    assert len(rows)==48;validate(rows);validate_city_articles(rows)
    prior=set()
    for path in root.glob('*ROWS.json'):
        if path==out:continue
        try:data=json.loads(path.read_text()).get('rows',[])
        except Exception:continue
        prior.update(tuple(x.get('ids',[])) for x in data)
    assert not any(tuple(row['ids']) in prior for row in rows)
    assert all(row['ids'].count(366)==1 for row in rows)
    result={'schema':'odd_framing_fresh.rows.v1','selection':'Authored templates, city pairs and endpoints frozen without model execution or scores. City and endpoint strings verified as single GPT2 tokens.','templates':TEMPLATES,'city_pairs':CITY_PAIRS,'endpoints':ENDPOINTS,'rows':rows}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'rows':len(rows),'pairs':len(rows)//2,'families':len(TEMPLATES),'lengths':sorted(set(len(x['ids']) for x in rows)),'quote_positions':sorted(set(x['ids'].index(366) for x in rows))}))


if __name__=='__main__':main()
