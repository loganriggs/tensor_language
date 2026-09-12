"""Weight-only repeated-input mixing on frozen learned banks; existing cache only validates.

A dense/lowrank identity, baseline replay, normal residual <=1e-8;
B >=20% native error reduction on both banks; C both native errors <=.1.
"""
from pathlib import Path
import json,hashlib
import torch
from coupled_quartic_writer_v1 import gram,solve,features
from quartic_repeated_input_v1 import square_traces,repeated_inner
from quartic_repeated_lowrank_v1 import metric

P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(91851)
b=torch.linalg.qr(torch.randn(3,7,2),mode='reduced')[0];n=torch.randn(3,2)
t=torch.randn(2,7,7);t=(t+t.transpose(-1,-2))/2;c=torch.randn(3,2);k=gram(b,n)
newk,newc=metric(k,c,b,n,t);traces=square_traces(b,n)
refk= repeated_inner(k,traces,traces);refc=repeated_inner(c,traces,t)
control=max(float((newk-refk).norm()/refk.norm()),float((newc-refc).norm()/refc.norm()));assert control<=1e-10
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
tp=P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_PROGRAMS.pt'
tr=json.loads((P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_RESULT.json').read_text());assert digest(tp)==tr['artifact_sha256']
target=torch.load(tp,weights_only=True)['programs'][0]
reports=[];saved=[]
for stem in ('COUPLED_QUARTIC_NONLINEAR_V2','COUPLED_QUARTIC_LBFGS_V1'):
    source=P/(stem+'_PROGRAM.pt');receipt=json.loads((P/(stem+'_RESULT.json')).read_text());assert digest(source)==receipt['artifact_sha256']
    program=torch.load(source,weights_only=True);assert torch.equal(program['output_writers'],target['output_writers'])
    b,n,a=program['input_readers'],program['inner_weights'],program['mixing'];k=gram(b,n);c=k@a
    capture=float((a*c).sum());objective_replay=abs(capture/program['divisor']+receipt['history'][-1]['objective'])
    wk,wc=metric(k,c,b,n,target['native_target_traces']);mix,diag=solve(wk,wc)
    def loss(m):return float((m*(wk@m)).sum()-2*(m*wc).sum())
    oldloss,newloss=loss(a),loss(mix)
    # Fitting is complete before reading the developmental validation ports.
    x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double()
    pre=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double()
    denominator=pre.square().mean(-1)+torch.finfo(torch.float32).eps
    reference=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0]
    f=features(b,n,x);w=program['output_writers']
    def error(m):return float((f@m@w.T/denominator[:,None]-reference).norm()/reference.norm())
    before,after=error(a),error(mix)
    r=dict(stem=stem,program_sha256=digest(source),baseline_error=before,repeated_input_error=after,normal_residual=diag['normal_residual'],objective_replay_error=objective_replay,native_replay_error=abs(before-receipt['final_native_error']),objective_before=oldloss,objective_after=newloss)
    reports.append(r);saved.append(dict(stem=stem,source_sha256=digest(source),mixing=mix));print(json.dumps(r),flush=True)
artifact=P/'QUARTIC_LEARNED_REPEATED_V1_MIXING.pt';assert not artifact.exists();torch.save(saved,artifact)
r=dict(pred_a=control<=1e-10 and all(max(z['normal_residual'],z['objective_replay_error'],z['native_replay_error'])<=1e-8 and z['objective_after']<=z['objective_before'] for z in reports),pred_b=all(z['repeated_input_error']<=.8*z['baseline_error'] for z in reports),pred_c=all(z['repeated_input_error']<=.1 for z in reports),dense_control_error=control,reports=reports,artifact_sha256=digest(artifact),target_trace_sha256=digest(tp),scope='Exact fixed learned-bank isotropic mixing using previously checked native traces; C=KA uses saved coefficient-optimal mixing. No nonlinear reader fitting or text fitting.')
out=P/'QUARTIC_LEARNED_REPEATED_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');assert r['pred_a']
