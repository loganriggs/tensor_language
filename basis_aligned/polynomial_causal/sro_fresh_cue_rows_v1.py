"""Token-only prospective template shift; no model scores used."""
from pathlib import Path
import json
import tiktoken
from regional_cue_row_check_v1 import validate
from scalar_producers_context_transfer_rows_v1 import PAIRS


def main():
    p=Path(__file__).resolve().parent
    out=p/'SRO_FRESH_CUE_V1_ROWS.json'
    assert not out.exists()
    enc=tiktoken.get_encoding('gpt2'); old=set(); sources=[]
    for path in sorted(p.glob('*ROWS.json')):
        data=json.loads(path.read_text())
        if not isinstance(data,dict):continue
        for key in ['rows','regional']:
            for row in data.get(key,[]):
                if isinstance(row,dict) and 'ids' in row:old.add(tuple(row['ids']))
        sources.append(path.name)
    rows=[]
    for family in range(3):
        for city_pair,cities in enumerate([('Liverpool','Boston'),('Bristol','Austin')]):
            for concept,(uk,us,stem) in enumerate(PAIRS):
                ends=[enc.encode(x) for x in [uk,us]]
                assert all(len(x)==1 for x in ends)
                for side,(cue,city) in enumerate(zip(['British','American'],cities)):
                    other=cities[1-side]
                    if family==0:
                        text=f'The archive contains a note dated 1987. Its author grew up in {city}. In the author\'s original spelling: "{stem}'
                    elif family==1:
                        text=f'A reviewer in {other} quotes a childhood diary written in {city}. Preserve the diary\'s spelling: "{stem}'
                    else:
                        text=f'For a {city} community newsletter, copy this sentence using local spelling: "{stem}'
                    ids=enc.encode(text);assert tuple(ids) not in old
                    rows.append(dict(row_id=len(rows),family=family,city_pair=city_pair,concept=concept,
                                     cue=cue,city=city,text=text,ids=ids,uk_id=ends[0][0],
                                     us_id=ends[1][0],control_ids=[670,3946]))
    checks=[validate(rows[:24]),validate(rows[24:48],expected_token_differences=2),validate(rows[48:])]
    assert len({tuple(r['ids']) for r in rows})==72
    result=dict(rows=rows,checks=checks,novelty_source_files=sources,
                scope='New token sequences/templates relative to listed panels; endpoints reused. No pretraining/corpus independence or unseen cities globally claimed. No model-score filtering.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(dict(rows=len(rows),maximum_tokens=max(len(r['ids']) for r in rows),checks=checks))


if __name__=='__main__':main()
