"""Checks for the repeated article confound in controlled regional cue panels."""
import re

def validate(rows):
    for row in rows:
        if re.search(r'\ba\s+American\b|\ban\s+British\b',row['text'],flags=re.I):
            raise ValueError(f"Article/cue confound in row {row.get('row_id')}: {row['text']}")
    for i in range(0,len(rows),2):
        left,right=rows[i:i+2]
        assert left['cue']=='British' and right['cue']=='American'
        assert len(left['ids'])==len(right['ids'])
        assert sum(a!=b for a,b in zip(left['ids'],right['ids']))==1
    return dict(rows=len(rows),pairs=len(rows)//2,scope='Controlled British/American article and one-token cue checks; not a general grammaticality validator.')
