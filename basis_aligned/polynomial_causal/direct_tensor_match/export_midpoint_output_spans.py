"""Exact output-space oracle controls; no fitted output activations."""
from pathlib import Path
import json,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
programs=torch.load(p/'MIDPOINT_CENTERED_CORRECTION_GRAPHS_V1.pt',weights_only=True);checks=[]
for name,e in programs.items():
 spans=[e['output_basis'].double()]
 if 'centered_correction_writers' in e:spans.append(e['centered_correction_writers'].double())
 C=torch.cat(spans,1);U,s,_=torch.linalg.svd(S@C,full_matrices=False);rank=int((s>s[0]*1e-10).sum());Q=U[:,:rank]
 readers=S@Q;writers=torch.linalg.solve(S,Q)
 error=float((C-writers@(readers.T@C)).norm()/C.norm());assert error<1e-9
 e['span_readers']=readers;e['span_writers']=writers
 checks.append(dict(program=name,span_rank=rank,replay=error,orthogonality=float((Q.T@Q-torch.eye(rank,dtype=Q.dtype)).abs().max())))
output=p/'MIDPOINT_OUTPUT_SPAN_CONTROLS_V1.pt';assert not output.exists();torch.save(programs,output)
(p/'MIDPOINT_OUTPUT_SPAN_CONTROLS_V1.json').write_text(json.dumps(dict(checks=checks,scope='Span of exported output basis and added correction writers in vocabulary-centered metric. This can be a superset of actual effective readout range, giving a valid but potentially loose coefficient-error lower bound. Oracle uses exact native output values; not an executable compressed replacement.'),indent=2)+'\n');print(checks)
