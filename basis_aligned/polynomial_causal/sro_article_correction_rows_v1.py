"""Consistent article correction for both affected 24-row families."""
from pathlib import Path
from datetime import datetime,timezone
import json,tiktoken
from regional_cue_row_check_v1 import validate
from regional_city_article_check_v1 import validate_city_articles


def main():
    p=Path(__file__).resolve().parent;out=p/'SRO_ARTICLE_CORRECTION_V1_ROWS.json'
    assert not out.exists();enc=tiktoken.get_encoding('gpt2');rows=[];rejected=[]
    for family,stem in enumerate(['SRO_FRESH_CUE_V1','SRO_FOUR_TERM_CONFIRMATION_V1']):
        old=json.loads((p/(stem+'_ROWS.json')).read_text())['rows'][48:]
        for row in old:
            try:validate_city_articles([row])
            except ValueError:rejected.append([stem,row['row_id']])
            r=dict(row);r.update(family=family,row_id=len(rows),source_stem=stem,source_row=row['row_id'])
            r['text']=row['text'].replace('For a ','For the ',1) if family==0 else row['text'].replace('A ','The ',1)
            assert r['text']!=row['text'];r['ids']=enc.encode(r['text']);rows.append(r)
    assert len(rows)==48 and len(rejected)==48
    validate(rows);checks=validate_city_articles(rows)
    # A/a and an variants both rejected; neutral forms accepted. No scores used.
    for text in ['A Oxford museum','an Denver museum','a Liverpool newsletter']:
        city=text.split()[1]
        try:validate_city_articles([dict(city=city,text=text)])
        except ValueError:pass
        else:raise AssertionError('Guard failed: '+text)
    result=dict(rows=rows,checks=checks,rejected_prior_rows=rejected,
                scope='All rows in each affected family corrected uniformly. Other families/results remain unchanged. Same endpoints and cities, no score filtering.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    (p/'SRO_ARTICLE_CORRECTION_V1_CPU_CONTROL.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=True,prior_rows_rejected=len(rejected),corrected_rows_accepted=len(rows),max_tokens=max(len(r['ids']) for r in rows)),indent=2)+'\n')
    print(dict(rows=len(rows),prior_rows_rejected=len(rejected),examples=[rows[0]['text'],rows[24]['text']]))


if __name__=='__main__':main()
