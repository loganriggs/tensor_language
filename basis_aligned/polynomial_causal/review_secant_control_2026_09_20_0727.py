"""Bounded CPU falsifier: endpoint closure does not identify reusable readers."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
P=Path(__file__).resolve().parent
OUT=P/'REVIEW_SECANT_CONTROL_2026-09-20_0727.json'
# Predeclared controls: exact closure <=1e-12, same endpoint derivative,
# distinct cubic readers; quadratic full-rank endpoint reader rescues new edit.
x=np.array([1.,2.,3.]); y=np.array([2.,4.,5.]); delta=y-x
mid=(x+y)/2
# f=(a*b)*c versus a*(b*c); exact midpoint product rule at each node.
gleft=np.array([mid[1]*mid[2],mid[0]*mid[2],(x[0]*x[1]+y[0]*y[1])/2])
gright=np.array([(x[1]*x[2]+y[1]*y[2])/2,mid[0]*mid[2],mid[0]*mid[1]])
change=float(np.prod(y)-np.prod(x))
closure=max(abs(gleft@delta-change),abs(gright@delta-change))
assert closure<1e-12 and np.linalg.norm(gleft-gright)>0
# Both rules have the correct equal-endpoint derivative, hence that check
# does not remove computational-graph dependence either.
grad=np.array([x[1]*x[2],x[0]*x[2],x[0]*x[1]])
equal_left=np.array([x[1]*x[2],x[0]*x[2],(x[0]*x[1]+x[0]*x[1])/2])
equal_right=np.array([(x[1]*x[2]+x[1]*x[2])/2,x[0]*x[2],x[0]*x[1]])
assert np.array_equal(grad,equal_left) and np.array_equal(grad,equal_right)
Q=np.array([[1.,2.],[2.,3.]])
xq=np.zeros(2); train=np.array([1.,0.]); test=np.array([0.,1.])
reader=Q@(xq+train); target=float(test@Q@test-xq@Q@xq)
frozen=float(reader@(test-xq)); rescued=float((Q@(xq+test))@(test-xq))
assert abs(rescued-target)<1e-12 and abs(frozen-target)>.1
# Explicit non-identifiability even for a fixed endpoint: add an orthogonal
# vector times ||delta||^2, which vanishes in the equal-endpoint limit.
orth=np.array([delta[1],-delta[0],0.]); alternative=gleft+(delta@delta)*orth
assert abs(alternative@delta-change)<1e-12
files=['CITY_VALUE_PATH_COMPOSITION_V1_RESULT.json','CIRCUIT_GRAPH_REGISTRY_V1.md',
       '../bilinear_quotient/circuits/followups/full_suffix_reader_v676_result.json',
       '../bilinear_quotient/circuits/followups/subject_attention_freeze_v674_result.json']
sources={}
for name in files:
 path=(P/name).resolve();sources[str(path.relative_to(ROOT))]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mtime_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat())
d=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='float64 planted CPU falsifier, not native-model evidence',
 cubic=dict(x=x.tolist(),y=y.tolist(),left_reader=gleft.tolist(),right_reader=gright.tolist(),true_change=change,max_closure_error=float(closure),reader_difference=float(np.linalg.norm(gleft-gright))),
 quadratic=dict(matrix=Q.tolist(),training_endpoint=train.tolist(),test_endpoint=test.tolist(),frozen_prediction=frozen,true_change=target,endpoint_specific_rescue=rescued),
 gates=dict(exact_closure=True,distinct_readers=True,frozen_reader_transfer_falsified=True,full_rank_rescue=True),sources=sources)
with OUT.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(d,indent=2))
