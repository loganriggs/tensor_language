"""Preserve all V1 cases and thresholds; correct the shared article only."""
from pathlib import Path
import json,tiktoken
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent;enc=tiktoken.get_encoding('gpt2')
a=json.loads((P/'REGIONAL_BEHAVIOR_CONTROLS_V1_ROWS.json').read_text());old=a['rows']
try:validate(old)
except ValueError:old_rejected=True
else:raise AssertionError('Known malformed original rows were not rejected')
rows=[]
for row in old:
    assert row['text'].startswith('A ')
    new=dict(row,text='The '+row['text'][2:]);new['ids']=enc.encode(new['text']);assert new['ids'][1:]==row['ids'][1:];rows.append(new)
control=validate(rows);assert control['rows']==32
out=P/'REGIONAL_BEHAVIOR_CONTROLS_V2_ROWS.json';assert not out.exists();a.update(rows=rows,scope='Article-only correction of V1: The British/The American. Same cases, target labels and token lengths; previous V1 semantic results confounded. No model outcomes used to choose replacement cases.');out.write_text(json.dumps(a,indent=2)+'\n')
(P/'REGIONAL_CUE_ROW_CHECK_V1_RESULT.json').write_text(json.dumps(dict(original_rejected=old_rejected,corrected=control),indent=2)+'\n');print(control)
