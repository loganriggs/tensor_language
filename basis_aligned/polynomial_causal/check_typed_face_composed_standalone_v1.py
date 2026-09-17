"""Three-state program replay in an isolated CPU process without repo imports."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile
P=Path(__file__).resolve().parent;D=P/'extracted_circuits/typed_face_composed_raw_v1';STEM='TYPED_FACE_COMPOSED_RAW_V1'
CHILD=r'''
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from execute import Program
torch.set_num_threads(2)
p=torch.load('program.pt',weights_only=True,map_location='cpu');program=Program(p['head8'],p['head9'],p['lambda90'])
rows=torch.load('fixtures.pt',weights_only=True,map_location='cpu');errors=[];zero=[]
with torch.no_grad():
 for r in rows:
  args=(r['current'],r['donor'],r['raw'],r['rt'],r['dt'],r['city'],r['mask'])
  value=program.execute(*args);errors.append(float((value-r['expected']).norm()/r['expected'].norm()))
  zero.append(float(program.execute(*args,strength=0).abs().max()))
 rejected=False
 try:program.execute(r['current'],r['donor'],r['raw'],r['rt'],-1,r['city'],r['mask'])
 except ValueError:rejected=True
result={'pred_d':max(errors)<=1e-4 and max(zero)==0 and rejected,'max_write_error':max(errors),'zero_strength_max_abs':max(zero),'unknown_token_rejected':rejected,'fixtures':len(rows)}
print(json.dumps(result))
'''
def main():
 with tempfile.TemporaryDirectory(prefix='regional_composed_isolated_') as td:
  q=Path(td)
  for name in ['execute.py','head8.py','head9.py','program.pt']:shutil.copy2(D/name,q/name)
  shutil.copy2(P/(STEM+'_FIXTURES.pt'),q/'fixtures.pt');(q/'check.py').write_text(CHILD)
  proc=subprocess.run([sys.executable,'-I',str(q/'check.py')],cwd=q,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=''),text=True,capture_output=True,timeout=120)
  if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
  result=json.loads(proc.stdout)
 result['program_sha256']=hashlib.sha256((D/'program.pt').read_bytes()).hexdigest();result['scope']='Isolated native-fixture replay of current8,donor-city8,raw9 to head9 write. Three native-state arrays remain. No model or repo imports, no new OOD/selectivity/causal composition certification.'
 (P/(STEM+'_STANDALONE_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
