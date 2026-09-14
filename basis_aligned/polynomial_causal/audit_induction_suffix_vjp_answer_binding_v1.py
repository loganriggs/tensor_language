#!/usr/bin/env python3
import json
from pathlib import Path
P=Path(__file__).resolve().parent
frozen=json.loads((P/'INDUCTION_CONTEXTUAL_CONSUMER_RESPONSE_V1_ROWS.json').read_text())
rows=frozen['rows']
required=('recipient_answer_id','recipient_other_answer_id','recipient_endpoint_id')
assert len(rows)==96 and all(all(k in row for k in required) for row in rows)
assert len({(row['recipient_answer_id'],row['recipient_other_answer_id']) for row in rows})==48
out={'schema':'induction_suffix_vjp_answer_binding_audit_v1','row_count':len(rows),'answer_pair_count':48,'answer_source':'directed rows, not endpoint specs','passed':True,'gpu_used':False,'model_loaded':False}
(P/'INDUCTION_SUFFIX_VJP_ANSWER_BINDING_AUDIT_V1.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,sort_keys=True))
