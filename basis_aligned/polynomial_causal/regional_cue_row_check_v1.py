"""Checks for the repeated article confound in controlled regional cue panels."""
import re

def validate(rows, *, expected_token_differences=1):
    if expected_token_differences not in (1, 2):
        raise ValueError('Specify one cue change or an explicit two-position role swap')
    for row in rows:
        if re.search(r'\ba\s+American\b|\ban\s+British\b',row['text'],flags=re.I):
            raise ValueError(f"Article/cue confound in row {row.get('row_id')}: {row['text']}")
    for i in range(0,len(rows),2):
        left,right=rows[i:i+2]
        assert left['cue']=='British' and right['cue']=='American'
        assert len(left['ids'])==len(right['ids'])
        assert sum(a!=b for a,b in zip(left['ids'],right['ids']))==expected_token_differences
        if expected_token_differences == 2:
            changed=[(a,b) for a,b in zip(left['ids'],right['ids']) if a!=b]
            assert changed[0] == changed[1][::-1], 'Two changes must exchange the same cue tokens'
    return dict(rows=len(rows),pairs=len(rows)//2,expected_token_differences=expected_token_differences,scope='Controlled British/American article and explicit cue-change checks; not a general grammaticality validator.')
