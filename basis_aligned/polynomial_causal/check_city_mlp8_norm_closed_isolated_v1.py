"""Copy only the package and declared fixtures outside the repository and execute."""
import json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 out=P/'CITY_MLP8_NORM_CLOSED_V1_ISOLATED_RESULT.json';assert not out.exists()
 source=P/'extracted_circuits/city_mlp8_norm_closed_v1'
 with tempfile.TemporaryDirectory(prefix='mlp8-value-isolated-') as directory:
  temp=Path(directory)
  for name in ['execute.py','program.pt']:shutil.copyfile(source/name,temp/name)
  shutil.copyfile(P/'CITY_MLP8_NORM_CLOSED_V1_ARTIFACT.pt',temp/'fixtures.pt')
  (temp/'check.py').write_text('''import torch,json,time
from pathlib import Path
import execute
torch.set_num_threads(2);start=time.perf_counter();assert Path(execute.__file__).resolve().parent==Path.cwd()
p=execute.prepare(torch.load('program.pt',weights_only=True));f=torch.load('fixtures.pt',weights_only=True)['fixtures'];errors=[];outside=[]
with torch.no_grad():
 for x in f:
  got=execute.execute(p,**x['inputs']);errors.append(float((got-x['delta']).norm()/x['delta'].norm()));mask=x['inputs']['delta'].abs().sum(-1)>0;outside.append(float(got[~mask].abs().max()))
r={'passes':len(f)==40 and max(errors)<=1e-10 and max(outside)==0,'fixtures':len(f),'max_relative_error':max(errors),'max_outside':max(outside),'imports_outside_repository':True,'seconds':time.perf_counter()-start,'scope':'Isolated declared-input mediator replay; native z8/intervention/tokens supplied; editedRMS9 generated.'}
print(json.dumps(r))
''')
  run=subprocess.run([sys.executable,str(temp/'check.py')],cwd=temp,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=''),capture_output=True,text=True,check=True,timeout=120);r=json.loads(run.stdout)
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
