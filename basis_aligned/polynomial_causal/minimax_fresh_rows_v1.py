"""Score-free96freshprefixes plus24fixed anchors, frozen endpoint semantics."""
import json
from pathlib import Path
import tiktoken
from regional_cue_row_check_v1 import validate
from scalar_new_endpoints_rows_v1 import PAIRS

P=Path(__file__).resolve().parent


def main():
    out=P/'MINIMAX_FRESH_CACHE_V1_ROWS.json';assert not out.exists()
    seen=set()
    for file in P.glob('*ROWS.json'):
        data=json.loads(file.read_text())
        lists=[data] if isinstance(data,list) else [v for v in data.values() if isinstance(v,list)] if isinstance(data,dict) else []
        for values in lists:
            for row in values:
                if isinstance(row,dict) and isinstance(row.get('ids'),list) and all(isinstance(t,int) for t in row['ids']):
                    seen.add(tuple(row['ids']))
    enc=tiktoken.get_encoding('gpt2')
    anchors=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'][:24]
    rows=[dict(r,row_id=i,donor_id=i^1,family=0,family_name='historical_anchor') for i,r in enumerate(anchors)]
    templates=[
        'An editor saved this note from a writer based in {city}. The note begins: "{stem}',
        'This passage comes from the personal journal of a resident of {city}: "{stem}',
        'Someone living in {city} supplied the following sentence for the newsletter: "{stem}',
        'The archive lists {city} as the home of the author. Here is the opening of the letter: "{stem}',
    ]
    for family,template in enumerate(templates,1):
        for variant,cities in enumerate([('London','Boston'),('Manchester','Chicago')]):
            for concept,(uk,us,stem) in enumerate(PAIRS):
                for side,cue in enumerate(('British','American')):
                    text=template.format(city=cities[side],stem=stem);ids=enc.encode(text)
                    assert tuple(ids) not in seen;seen.add(tuple(ids))
                    ui,si=enc.encode(uk),enc.encode(us);assert len(ui)==len(si)==1
                    rows.append(dict(row_id=len(rows),donor_id=len(rows)^1,family=family,
                                     family_name=['anchor','editor_note','personal_journal','newsletter','archive_letter'][family],
                                     variant=variant,concept=concept,cue=cue,city=cities[side],cue_word=cities[side],
                                     uk_token=uk,us_token=us,uk_id=ui[0],us_id=si[0],control_ids=[670,3946],text=text,ids=ids))
    checks=validate(rows);assert len(rows)==120
    with out.open('x') as f:
        json.dump(dict(rows=rows,checks=checks,scope='24historicalanchors+96unseenfullprefixes; existing endpoints/cities, no scorefilter or corpusOOD.'),f,indent=2)
        f.write('\n')
    print(dict(rows=len(rows),max_tokens=max(len(r['ids']) for r in rows),checks=checks))


if __name__=='__main__':main()
