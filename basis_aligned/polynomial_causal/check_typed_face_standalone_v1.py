"""Isolated package-only replay of captured native head8.2 writes."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,tempfile
import torch
P=Path(__file__).resolve().parent
D=P/'extracted_circuits/odd_attention8h2_typed_face_v1'

def main():
    source=P/'REGIONAL_ENDPOINT_BATCHING_V1_ARTIFACT.pt'
    representatives=torch.load(source,map_location='cpu',weights_only=True)['representatives']
    with tempfile.TemporaryDirectory(prefix='typed-face-isolated-') as tmp:
        root=Path(tmp)
        for name in ('native.py','program.pt'):shutil.copyfile(D/name,root/name)
        torch.save(representatives,root/'fixtures.pt')
        (root/'check.py').write_text('''from pathlib import Path
import importlib.util,json,torch
p=Path(__file__).resolve().parent
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location("native",p/"native.py")
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
program=torch.load(p/"program.pt",map_location="cpu",weights_only=True)
errors=[]
for item in torch.load(p/"fixtures.pt",map_location="cpu",weights_only=True):
    target=item.pop("expected_write")
    actual=n.execute(program,**item)
    errors.append(float((actual-target).norm()/target.norm().clamp_min(1e-8)))
assert len(errors)==8 and max(errors)<=1e-5,errors
try:n.inherited(program,-1)
except ValueError:pass
else:raise AssertionError("unknown city silently accepted")
print(json.dumps({"pred_a":True,"native_write_relative_errors":errors,"representatives":len(errors),"native_state_ports":2,"unknown_city_rejected":True}))
''')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
        result=subprocess.run(['/venv/main/bin/python','-I',str(root/'check.py')],cwd=root,env=env,text=True,capture_output=True,check=True)
        receipt=json.loads(result.stdout)
    receipt.update({'fixture_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'program_sha256':hashlib.sha256((D/'program.pt').read_bytes()).hexdigest(),'scope':'Isolated CPU execution of head8.2 write on eight captured native inputs. Does not extract upstream state generators or downstream suffix; earlier all-readout strict replay failure remains.'})
    (P/'TYPED_FACE_STANDALONE_V1_RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
