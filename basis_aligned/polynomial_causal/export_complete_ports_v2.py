"""Export the frozen conditional response and execute its registered isolation check.

A: 432 scalar/norm/own-change replays <=1e-10; exact-zero absolute <=1e-12.
B: separate Python -I process, no project imports, compensation required.
No fitting, GPU, native forwards or new native-fidelity claim.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from complete_response_ports_v1 import project_ports
from contracted_qk_response_v1 import features
from composed_key_span_v1 import scalar
from research_phase_clock_v1 import mark

P = Path(__file__).resolve().parent
PACKAGE = P/'extracted_circuits/complete_response_ports_v2'
SOURCE = P/'COMPOSED_KEY_SPAN_V1_PROGRAM.pt'
PRIMITIVES = ['readers', 'left', 'right', 'direction', 'gain', 'key_coordinates']
STRENGTHS = [-2, -1, 0, .5, 1, 2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


WORKER = r'''
import importlib.util,json,sys
from pathlib import Path
import torch
torch.set_num_threads(2)
root=Path(sys.argv[1]); original=Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('exported_response',root/'execute.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
p=module.load(root/'program.pt')
cases=torch.load(root/'fixture.pt',weights_only=True)
checks=[]; missing=False
def error(actual,reference):
    delta=float((actual-reference).norm());norm=float(reference.norm())
    return delta/norm if norm else delta
with torch.no_grad():
 for index,case in enumerate(cases):
    ports=module.project(case['z'],case['h'],case['u'],p)
    projection=max(error(ports[k],v) for k,v in case['ports'].items())
    zero,_=module.execute(ports,case['amp']*0,p)
    if index==0:
        broken=dict(ports);del broken['cu']
        try:module.execute(broken,case['amp'],p)
        except KeyError:missing=True
    for strength,ref,rho in zip([-2,-1,0,.5,1,2],case['references'],case['norms']):
        value,norm=module.execute(ports,case['amp']*strength,p)
        own=ref-case['references'][2]
        checks.append(dict(row=index,strength=strength,projection_error=projection,
            scalar_error=error(value,ref),norm_error=error(norm,rho),
            own_change_error=error(value-zero,own),
            zero_own_pass=float((value-zero-own).abs().max())<=1e-12 if not bool(own.norm()) else True))
leaks=[]
for name,loaded in list(sys.modules.items()):
 file=getattr(loaded,'__file__',None)
 if file and Path(file).resolve().is_relative_to(original):leaks.append(name)
result=dict(records=checks,missing_compensation_rejected=missing,project_imports=leaks,
    primitive_scalars=sum(p[k].numel() for k in ['readers','left','right','direction','gain','key_coordinates']),
    resident_scalars=sum(t.numel() for t in p.values()),
    scope='Independent Python -I in temporary directory; only explicit package and fixture paths supplied.')
(root/'isolated_result.json').write_text(json.dumps(result))
'''


@torch.no_grad()
def main():
    out = P/'COMPLETE_PORTS_EXTRACTION_V2_RESULT.json'
    if out.exists() or (PACKAGE/'program.pt').exists():
        raise FileExistsError('Preserve prior exports and receipts')
    torch.set_num_threads(2)
    mark(P/'RESEARCH_ACTIVITY_2026-09-10_1614.jsonl', 'complete_ports_extraction_v2_export', 'implementation')
    program = torch.load(SOURCE, weights_only=True)
    torch.save({k: program[k].clone() for k in PRIMITIVES}, PACKAGE/'program.pt')
    checkpoint = Path('/home/loganriggs/.local/share/bilin18/hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
    sd = torch.load(checkpoint, weights_only=True, mmap=True)
    lam = sd['transformer.h.9.lambdas'].double()
    bias = sd['transformer.h.8.mlp.Down_bias'].double()
    cache_path = P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt'
    row_path = P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json'
    cache = torch.load(cache_path, weights_only=True, mmap=True)
    rows = json.loads(row_path.read_text())['rows']
    cases = []
    for i, row in enumerate(rows):
        n = len(row['ids'])
        ids = torch.tensor([row['ids']])
        z = cache['z8'][0, i, :n][None].double()
        h = cache['raw9'][0, i, :n][None].double()
        amp = cache['amplitude8'][0, i, :n][None]
        x0 = F.rms_norm(F.embedding(ids, sd['transformer.wte.weight']), (1152,)).double()
        u = (h-lam[1]*x0)/lam[0]-z-bias
        references, norms = [], []
        for strength in STRENGTHS:
            rd, rho = features(z, h, u, amp*strength, program)
            references.append(scalar(rd, rho, program))
            norms.append(rho)
        cases.append(dict(z=z, h=h, u=u, amp=amp, ports=project_ports(z,h,u,program),
                          references=references, norms=norms))
    mark(P/'RESEARCH_ACTIVITY_2026-09-10_1614.jsonl', 'complete_ports_extraction_v2_isolation', 'validation')
    with tempfile.TemporaryDirectory(prefix='complete-ports-v2-') as temp:
        root = Path(temp)
        for name in ['execute.py', 'program.pt']:
            shutil.copy2(PACKAGE/name, root/name)
        torch.save(cases, root/'fixture.pt')
        (root/'check.py').write_text(WORKER)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        child = subprocess.run([sys.executable, '-I', str(root/'check.py'), str(root), str(P.parent.parent)],
                               cwd=root, env=env, capture_output=True, text=True, timeout=120)
        if child.returncode:
            raise RuntimeError(child.stderr)
        isolated = json.loads((root/'isolated_result.json').read_text())
    manifest = dict(source_artifact=SOURCE.name, source_sha256=sha(SOURCE),
                    runtime_files={name: dict(sha256=sha(PACKAGE/name), bytes=(PACKAGE/name).stat().st_size)
                                   for name in ['execute.py', 'program.pt']},
                    external_inputs=['pristine z/h/bias-free u or their declared ports',
                                     'intervention amplitude; native prefix and suffix not included'],
                    precision='FP64 CPU, native FP32 RMS epsilon and BF16-rounded rotary constants',
                    assumption='Unchanged rank64 directional approximation')
    (PACKAGE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    records = isolated['records']
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
                  pred_a=len(records)==432 and all(max(r[k] for k in ['projection_error','scalar_error','norm_error','own_change_error'])<=1e-10 and r['zero_own_pass'] for r in records),
                  pred_b=not isolated['project_imports'] and isolated['missing_compensation_rejected'],
                  max_errors={k:max(r[k] for r in records) for k in ['projection_error','scalar_error','norm_error','own_change_error']},
                  records=records, primitive_scalars=isolated['primitive_scalars'],
                  resident_scalars=isolated['resident_scalars'],
                  required_runtime_bytes=sum(v['bytes'] for v in manifest['runtime_files'].values()),
                  total_package_bytes=sum(f.stat().st_size for f in PACKAGE.iterdir() if f.is_file()),
                  dependency_hashes={f.name:sha(f) for f in [SOURCE,row_path,cache_path,Path(__file__),PACKAGE/'execute.py',PACKAGE/'program.pt']},
                  project_imports=isolated['project_imports'],
                  scope='Standalone conditional extraction, 72 old contexts, six strengths; no new native behavior, OOD or full-model compression.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','dependency_hashes']}, indent=2))
    if not result['pred_a'] or not result['pred_b']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
