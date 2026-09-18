"""Run the exported program with only its files and declared inputs in a tempdir."""
import json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
    source=P/'extracted_circuits/city_residual6_single_input_v1'
    captured=torch.load(P/'CITY_SOURCE7_FRESH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
    fixtures=[{'inputs':f['inputs'],'expected':f['candidate_delta']} for f in captured]
    with tempfile.TemporaryDirectory(prefix='city-residual6-isolated-') as directory:
        temp=Path(directory)
        paths={'execute.py':P/'city_source7_present_generator_v1.py','city_attention7_full_v1.py':P/'city_attention7_full_v1.py','city_mlp7_readers_v1.py':P/'city_mlp7_readers_v1.py','attention7.pt':P/'CITY_SOURCE7_FRESH_V1_ATTENTION7.pt','readers.pt':P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt','head8.pt':P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt'}
        for name,path in paths.items():shutil.copyfile(path,temp/name)
        torch.save(fixtures,temp/'fixtures.pt')
        (temp/'check.py').write_text('''import json,time
from pathlib import Path
import torch
import execute,city_attention7_full_v1,city_mlp7_readers_v1
torch.set_num_threads(2)
start=time.perf_counter()
assert all(Path(m.__file__).resolve().parent==Path.cwd() for m in [execute,city_attention7_full_v1,city_mlp7_readers_v1])
program={n:torch.load(n+'.pt',weights_only=True) for n in ['attention7','readers','head8']}
fixtures=torch.load('fixtures.pt',weights_only=True);errors=[];outside=[];finite=True
for f in fixtures:
    got=execute.execute(program,**f['inputs']);errors.append(float((got-f['expected']).norm()/f['expected'].norm()))
    outside.append(float(got[:,~f['inputs']['destination']].abs().max()));finite=finite and bool(torch.isfinite(got).all())
inputs=dict(fixtures[0]['inputs']);inputs['token_ids']=inputs['token_ids'].clone();inputs['token_ids'][0,0]=-1
unknown=False
try:execute.execute(program,**inputs)
except ValueError:unknown=True
r={'passes':max(errors)<=1e-10 and max(outside)==0 and finite and unknown and len(fixtures)==40,
   'fixtures':len(fixtures),'max_relative_write_error':max(errors),'max_outside':max(outside),'unknown_token_rejected':unknown,
   'imports_outside_repository':True,'seconds':time.perf_counter()-start,'scope':'Isolated CPU declared-input replay; fresh residual6 source-group fixtures; three copied Python modules and declared weight files only; compares generated MLP7-present writes.'}
print(json.dumps(r))
''')
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='')
        run=subprocess.run([sys.executable,str(temp/'check.py')],cwd=temp,env=environment,capture_output=True,text=True,check=True,timeout=120)
        result=json.loads(run.stdout)
    (P/'CITY_SOURCE7_FRESH_V1_ISOLATED_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
