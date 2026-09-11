"""Repeat the tiny deterministic dense1902 control with a larger step budget.
Original step-limit miss preserved. No native, GPU or text data.
"""
import json
from pathlib import Path
import torch
from orthogonal_reader_msp_v1 import fit, polar, topk_energy

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
torch.manual_seed(700)
d = 24
truth = polar(torch.randn(d,d))
for probability in (.15,1.):
    def draw(n):
        coefficients = torch.randn(d,n)*(torch.rand(d,n)<probability)
        y = truth@coefficients
        return y/y.norm(dim=0,keepdim=True).clamp_min(1e-30)
    train,test = draw(8*d),draw(4096)
a,result = fit(train,1902,max_steps=20000,max_seconds=30)
previous = json.loads(Path(__file__).with_name('READER_MSP_FINITE_SAMPLE_V1_AUDIT.json').read_text())['starts'][-1]
old_point_error = abs(result['history'][2000]['objective']-previous['final']['objective'])
test_capture = topk_energy(a,test,3)
baseline = topk_energy(torch.eye(d),test,3)
gap = topk_energy(a,train,3)-test_capture
output=dict(predictions={
    'pred_a_original_point_replayed':old_point_error<=1e-12,
    'pred_b_converged':result['converged'],
    'pred_c_dense_null_unchanged':test_capture-baseline<=.02 and gap>=.05},
    fit=result,old_point_objective_error=old_point_error,
    train_top3_energy=topk_energy(a,train,3),test_top3_energy=test_capture,
    baseline_test_top3_energy=baseline,train_test_gap=gap,
    scope='Full deterministic replay with higher cap, including original2000steps in time; old failed prediction unchanged.')
with Path(__file__).with_name('READER_MSP_DENSE_POLISH_V1_AUDIT.json').open('x') as f:
    json.dump(output,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in output.items() if k!='fit'},indent=2))
print(json.dumps(result['final']))
