"""RUNG 294: signed a16 adoption gate on the true corrected mixed104 program.

REGISTERED PREDICTIONS: (a) live baseline census <=0.0065 and max fresh <=0.020; (b) direct signed
compiled/native effect cosine >=0.90 and normalized error <=0.60; (c) non-own circuit Spearman >=0.90
and a16-own median magnitude ratio in [0.60,1.40].  NULL: cosine <0.70 or rho <0.75.  The underlying
runner also requires physical QK indices {0..95,120..127}, factor width 104, at layers 2..17.  Diagnostic
only; storage price unchanged and still pending an exact bill.  Self-reviewed; bqrunner only.
"""

import os
import runpy

_PREDICATE_SCHEMA = {
    'pred_a_live_config': None,
    'pred_b_signed_effect': None,
    'pred_c_circuits': None,
}
assert len(_PREDICATE_SCHEMA) == 3
os.environ['BILIN18_A16_TRUE_MIXED'] = '1'
runpy.run_path('/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/'
               'a16_transfer_mixed_native_a1v.py', run_name='__main__')
