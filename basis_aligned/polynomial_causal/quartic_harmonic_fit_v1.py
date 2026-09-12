"""Frozen-bank harmonic fit plus exact target lower components.

A control/replay/solve; B both native write errors halve; C both <=.1.
No text fitting; dense target traces charged in addition to quadratic bank.
"""
from pathlib import Path
import hashlib,json
import torch
from coupled_quartic_writer_v1 import gram,solve,features
from quartic_harmonic_metric_v1 import metric
from quartic_harmonic_v1 import evaluate_lower

P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
assert json.loads((P/'QUARTIC_HARMONIC_METRIC_V1_CONTROL.json').read_text())['pred_a']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
tp=P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_PROGRAMS.pt'
assert digest(tp)==json.loads((P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_RESULT.json').read_text())['artifact_sha256']
target=torch.load(tp,weights_only=True)['programs'][0];reports=[];saved=[]
for stem in ('COUPLED_QUARTIC_NONLINEAR_V2','COUPLED_QUARTIC_LBFGS_V1'):
    source=P/(stem+'_PROGRAM.pt');previous=json.loads((P/(stem+'_RESULT.json')).read_text());assert digest(source)==previous['artifact_sha256']
    program=torch.load(source,weights_only=True);assert torch.equal(program['output_writers'],target['output_writers'])
    b,n,a=program['input_readers'],program['inner_weights'],program['mixing']
    k=gram(b,n);c=k@a;hk,hc=metric(k,c,b,n,target['native_target_traces']);mix,diag=solve(hk,hc)
    beta=(n*n.sum(-1,keepdim=True)+2*n.square())/3
    flat=b.permute(1,0,2).reshape(1152,-1)
    fitted_traces=torch.stack([(b*(beta*mix[:,m,None])[:,None,:]).permute(1,0,2).reshape(1152,-1)@flat.T for m in range(2)])
    remaining_trace=target['native_target_traces']-fitted_traces
    # All parameters are fixed before loading validation inputs.
    x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double()
    pre=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double()
    den=pre.square().mean(-1)+torch.finfo(torch.float32).eps
    ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0]
    feat=features(b,n,x);writer=program['output_writers']
    old=feat@a@writer.T/den[:,None]
    radial,quadratic=evaluate_lower(remaining_trace,x)
    new=(feat@mix+radial+quadratic)@writer.T/den[:,None]
    before=float((old-ref).norm()/ref.norm());after=float((new-ref).norm()/ref.norm())
    objective=lambda m:float((m*(hk@m)).sum()-2*(m*hc).sum())
    r=dict(stem=stem,program_sha256=digest(source),baseline_error=before,harmonic_combined_error=after,normal_residual=diag['normal_residual'],native_replay_error=abs(before-previous['final_native_error']),harmonic_objective_before=objective(a),harmonic_objective_after=objective(mix),correction_write_norm_relative=float(((radial+quadratic)@writer.T/den[:,None]).norm()/ref.norm()))
    reports.append(r);saved.append(dict(stem=stem,source_sha256=digest(source),mixing=mix));print(json.dumps(r),flush=True)
artifact=P/'QUARTIC_HARMONIC_FIT_V1_MIXING.pt';assert not artifact.exists();torch.save(saved,artifact)
r=dict(pred_a=all(max(z['normal_residual'],z['native_replay_error'])<=1e-8 and z['harmonic_objective_after']<=z['harmonic_objective_before'] for z in reports),pred_b=all(z['harmonic_combined_error']<=.5*z['baseline_error'] for z in reports),pred_c=all(z['harmonic_combined_error']<=.1 for z in reports),reports=reports,artifact_sha256=digest(artifact),target_trace_sha256=digest(tp),diagnostic_floats=592704+2*1152*1152,scope='Harmonic degree4 fit with exact dense lower correction; fixed learned banks, native radius and denominator retained. No data fit, no compact or causal claim.')
out=P/'QUARTIC_HARMONIC_FIT_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');assert r['pred_a']
