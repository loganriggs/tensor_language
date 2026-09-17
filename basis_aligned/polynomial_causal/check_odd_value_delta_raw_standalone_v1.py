"""Isolated native-fixture replay for the reduced head9 write package."""
import hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'extracted_circuits/odd_value_delta_raw_v1'
CHILD=r'''
import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from execute import Program
torch.set_num_threads(2)
p=Program(torch.load('program.pt',map_location='cpu',weights_only=True))
rows=torch.load('fixtures.pt',map_location='cpu',weights_only=True)
errors=[];zero=[]
with torch.no_grad():
 for r in rows:
  value=p.execute(r['raw'],r['delta'],r['lambda90'],r['mask'])
  errors.append(float((value-r['expected']).norm()/r['expected'].norm()))
  zero.append(float(p.execute(r['raw'],torch.zeros_like(r['delta']),r['lambda90'],r['mask']).abs().max()))
result={'pred_d':max(errors)<=1e-4 and max(zero)==0,'max_write_error':max(errors),'zero_delta_max_abs':max(zero),'fixtures':len(rows)}
print(json.dumps(result))
'''
def main():
 source=P/'ODD_VALUE_DELTA_RAW_V1_PROGRAM.pt'
 if not (D/'program.pt').exists():shutil.copy2(source,D/'program.pt')
 assert (D/'program.pt').read_bytes()==source.read_bytes()
 with tempfile.TemporaryDirectory(prefix='odd_raw_standalone_') as td:
  q=Path(td);shutil.copy2(D/'program.pt',q/'program.pt');shutil.copy2(D/'execute.py',q/'execute.py');shutil.copy2(P/'ODD_VALUE_DELTA_RAW_V1_FIXTURES.pt',q/'fixtures.pt');(q/'check.py').write_text(CHILD)
  proc=subprocess.run([sys.executable,'-I',str(q/'check.py')],cwd=q,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=''),text=True,capture_output=True,timeout=120)
  if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
  result=json.loads(proc.stdout)
 result['program_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();result['scope']='Isolated CPU export replay on opened native fixtures; raw9 and delta8 remain explicit. No model import or inherited-first/initial-state input.'
 (P/'ODD_VALUE_DELTA_RAW_V1_STANDALONE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
