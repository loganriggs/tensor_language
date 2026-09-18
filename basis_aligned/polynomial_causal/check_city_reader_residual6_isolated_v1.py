"""Run the exported program with only its files and declared inputs in a tempdir."""
import json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
    source=P/'extracted_circuits/city_reader_residual6_v1'
    prefix=torch.load(P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt',weights_only=True)['fixtures']
    reference=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
    fixtures=[]
    for f,r in zip(prefix,reference,strict=True):
        city=f['city'];c=r['inputs']
        fixtures.append({'inputs':{'residual6':f['residual6'][:,:city+1],'token_ids':f['token_ids'][:,:city+1],'city':city,
                                   'rotated_queries':c['rotated_queries'],'destination':c['destination'],'input_rms':c['input_rms']},
                         'expected':r['expected_native_delta']})
    with tempfile.TemporaryDirectory(prefix='city-residual6-isolated-') as directory:
        temp=Path(directory)
        for name in ['execute.py','attention7.py','readers.py','citywrite.py','attention7.pt','readers.pt','head8.pt']:shutil.copyfile(source/name,temp/name)
        torch.save(fixtures,temp/'fixtures.pt')
        (temp/'check.py').write_text('''import json,time
from pathlib import Path
import torch
import execute,attention7,readers,citywrite
torch.set_num_threads(2)
start=time.perf_counter()
assert all(Path(m.__file__).resolve().parent==Path.cwd() for m in [execute,attention7,readers,citywrite])
program={n:torch.load(n+'.pt',weights_only=True) for n in ['attention7','readers','head8']}
fixtures=torch.load('fixtures.pt',weights_only=True);errors=[];outside=[];finite=True
for f in fixtures:
    got=execute.execute(program,**f['inputs']);errors.append(float((got-f['expected']).norm()/f['expected'].norm()))
    outside.append(float(got[:,~f['inputs']['destination']].abs().max()));finite=finite and bool(torch.isfinite(got).all())
inputs=dict(fixtures[0]['inputs']);inputs['token_ids']=inputs['token_ids'].clone();inputs['token_ids'][0,0]=-1
unknown=False
try:execute.execute(program,**inputs)
except ValueError:unknown=True
r={'passes':max(errors)<=1e-4 and max(outside)==0 and finite and unknown and len(fixtures)==40,
   'fixtures':len(fixtures),'max_relative_write_error':max(errors),'max_outside':max(outside),'unknown_token_rejected':unknown,
   'imports_outside_repository':True,'seconds':time.perf_counter()-start,'scope':'Isolated CPU declared-input replay; native query/RMS and residual6 inputs supplied.'}
print(json.dumps(r))
''')
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='')
        run=subprocess.run([sys.executable,str(temp/'check.py')],cwd=temp,env=environment,capture_output=True,text=True,check=True,timeout=120)
        result=json.loads(run.stdout)
    (P/'CITY_READER_RESIDUAL6_V1_ISOLATED_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
