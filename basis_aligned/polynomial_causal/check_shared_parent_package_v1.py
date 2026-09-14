"""Isolated portable runtime audit; no model, native fitting or repository imports."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

P=Path(__file__).resolve().parent


def main():
    out=P/'SHARED_PARENT_PACKAGE_V1_RESULT.json';assert not out.exists()
    root=P/'extracted_circuits/sparse_even_key_producers_8_2_9_8_v1'
    source=(P/'shared_parent_runtime_v1.py').read_text()
    portable=(root/'shared_execute.py').read_text()
    assert portable.split('\n\ndef load_shared')[0].rstrip()==source.rstrip()
    files=['execute.py','program.pt','shared_execute.py']
    original=json.loads((root/'manifest.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(root/'execute.py')==original['runtime_sha256']
    assert digest(root/'program.pt')==original['program_sha256']
    code="""
import importlib.util,json,torch
from pathlib import Path
root=Path.cwd()
def load(name):
 spec=importlib.util.spec_from_file_location(name,root/(name+'.py'))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('execute');new=load('shared_execute')
model=new.load_shared(root/'program.pt');torch.manual_seed(170233200);torch.set_num_threads(2)
x=torch.randn(2,19,1152);ids=torch.randint(0,50000,(2,19));errors=[]
for i in [0,1]:
 a=old.scalar(x,ids,model.p,i);b=model.scalar(x,ids,i)
 errors.append(float((a-b).norm()/a.norm()))
assert max(errors)<=1e-10
print(json.dumps(dict(errors=errors,adapter_bytes=sum(t.numel()*t.element_size() for pair in model.adapters for t in pair))))
"""
    with tempfile.TemporaryDirectory(prefix='bilin18-shared-parent-') as temp:
        for name in files:shutil.copyfile(root/name,Path(temp)/name)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        child=subprocess.run([sys.executable,'-I','-c',code],cwd=temp,env=env,text=True,capture_output=True,timeout=60,check=True)
        replay=json.loads(child.stdout)
    manifest=dict(version=1,entrypoint='shared_execute.py:load_shared',
                  files={name:dict(sha256=digest(root/name),bytes=(root/name).stat().st_size) for name in files},
                  program_weights_unchanged=True,extra_resident_adapter_bytes=replay['adapter_bytes'],
                  native_validated_class_sha256=hashlib.sha256(source[source.index('class SharedParent:'):].encode()).hexdigest(),
                  scope='Optional exact shared runtime; original manifest and weights remain unchanged. Native normalized contexts and suffix external. CPU speed evidence is not GPU/full-model speed.')
    manifest['required_file_bytes']=sum(v['bytes'] for v in manifest['files'].values())
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=max(replay['errors'])<=1e-10,pred_b=True,
                isolated_replay=replay,manifest=manifest,scope='Temporary independent Python -I process with only three package files. No new native evidence.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    with (root/'shared_manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
