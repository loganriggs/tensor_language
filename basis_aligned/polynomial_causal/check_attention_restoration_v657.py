"""CPU-only independent scoring of the opened v657 causal screen."""
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'basis_aligned/polynomial_causal'
source = ROOT / 'basis_aligned/bilinear_quotient/circuits/followups/subject_attention_freeze_v657_result.json'
data = json.loads(source.read_text())
def relative(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)) / sum(y*y for y in b))
rows = {}
for key, cell in data['cells'].items():
    target = cell['target']
    frozen = relative(cell['frozen_prediction'], target)
    native = relative(cell['restored_prediction'], target)
    folded = relative(cell['prediction'], target)
    assert abs(frozen-cell['frozen_relative_error']) < 1e-12
    assert abs(native-cell['restored_relative_error']) < 1e-12
    assert abs(folded-cell['relative_l2']) < 1e-12
    rows[key] = dict(frozen_error=frozen, native12_error=native, folded12_error=folded,
        error_reduction_fraction=1-native/frozen,
        folded_native12_discrepancy_over_target=math.sqrt(sum((x-y)**2 for x,y in zip(cell['prediction'],cell['restored_prediction']))/sum(y*y for y in target)))
negative_gate = all(v['error_reduction_fraction'] >= .5 for k,v in rows.items() if k.startswith('post'))
positive_gate = all(v['folded12_error'] <= .1 for v in rows.values())
assert negative_gate == data['predictions']['pred_b_effect_prediction']
assert positive_gate == data['predictions']['pred_c_capable_position_transfer']
result = dict(cells=rows, independent_scoring_pass=True,
    native12_halves_error=negative_gate, folded12_dense_suffix_pass=positive_gate,
    evidence_scope='48 opened rows, dense MLP suffix; not combined with reduced response chain',
    remaining='Fresh generalization, selective removal, reuse and total port-closed simplicity unestablished')
(P/'ATTENTION_RESTORATION_CPU_AUDIT_2026-09-20.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
