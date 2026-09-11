"""CPU fixed-support refit and compiled oblique reader execution control."""
import hashlib
import json
from pathlib import Path
import torch
from orthogonal_reader_msp_v1 import polar
from oblique_sparse_reader_v1 import canonical_maps,encode
from sparse_reader_program_v1 import SparseReaderProgram

torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(735)
d,m,k=12,17,3
matrix=polar(torch.randn(d,d))@torch.diag(torch.logspace(0,1,d))@polar(torch.randn(d,d))
shape=matrix@matrix.T
maps=canonical_maps(polar(torch.randn(d,d)),shape)
readers=torch.randn(2*m,d)
indices,values,encoding=encode(maps['dictionary'],maps['weight_coder'],readers,k,chunk=7)
codes=torch.zeros(2*m,d).scatter_(1,indices,values)
reconstructed=codes@maps['input_features']
down,bias=torch.randn(9,m),torch.randn(9)
program=SparseReaderProgram(maps['input_features'],indices,values,down,bias)
x=torch.randn(13,d)
left,right=reconstructed.split(m)
expected=((x@left.T)*(x@right.T))@down.T+bias
actual=program(x)
error=float((actual-expected).norm()/expected.norm())
full_error=float((maps['dictionary']@maps['weight_coder']-torch.eye(d)).norm())
result=dict(predictions={
    'pred_a_exact_conditional_solve':encoding['normal_equation_relative_residual']<=1e-10 and full_error<=1e-10,
    'pred_b_refit_improves':encoding['refit_squared_error']<=encoding['truncation_squared_error'] and encoding['aggregate_relative_gain']>=.01,
    'pred_c_executable_replay':error<=1e-10},encoding=encoding,
    full_coordinate_identity_error=full_error,execution_relative_error=error,
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('oblique_sparse_reader_v1.py'))},
    gpu_access=False,corpus_access=False,
    scope='Synthetic oblique basis encoding and exact program replay, not native factorization or behavioral evidence.')
with Path(__file__).with_name('OBLIQUE_SPARSE_READER_V1_CONTROL.json').open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
