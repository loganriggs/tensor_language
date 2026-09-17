"""Isolated CPU replay: only copied package, fixtures and installed PyTorch."""
import hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent
D=P/'extracted_circuits/odd_attention8h2_mlp7_donor_v1'
STEM='TYPED_FACE_MLP7_DONOR_V1'
CHILD=r'''
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import execute,donor
p=torch.load('program.pt',map_location='cpu',weights_only=True)
rows=torch.load('fixtures.pt',map_location='cpu',weights_only=True)
torch.set_num_threads(2)
se=[];we=[]
with torch.no_grad():
 for r in rows:
  initial=p['donor']['initial_table'][p['donor']['token_ids'].tolist().index(r['donor_token'])].unsqueeze(0)
  state=donor.generate(p['donor'],r['g7'],initial)
  value=execute.execute(p,r['current'],r['g7'],r['recipient_token'],r['donor_token'],r['city'],r['destination'])
  se.append(float((state-r['state']).norm()/r['state'].norm()))
  we.append(float((value-r['write']).norm()/r['write'].norm()))
 rejected=False
 try:execute.execute(p,r['current'],r['g7'],r['recipient_token'],-1,r['city'],r['destination'])
 except ValueError:rejected=True
result={'max_state_error':max(se),'max_write_error':max(we),'fixture_count':len(rows),'unknown_token_rejected':rejected,'pred_d':max(se)<=1e-5 and max(we)<=1e-5 and rejected}
print(json.dumps(result))
assert result['pred_d']
'''
def main():
 with tempfile.TemporaryDirectory(prefix='mlp7_donor_isolated_') as td:
  target=Path(td)
  for name in ['program.pt','execute.py','donor.py','native.py']:shutil.copy2(D/name,target/name)
  shutil.copy2(P/(STEM+'_FIXTURES.pt'),target/'fixtures.pt')
  (target/'check.py').write_text(CHILD)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='')
  proc=subprocess.run([sys.executable,'-I',str(target/'check.py')],cwd=target,env=env,text=True,capture_output=True,timeout=120)
  if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
  result=json.loads(proc.stdout)
 result['program_sha256']=hashlib.sha256((D/'program.pt').read_bytes()).hexdigest()
 result['scope']='Isolated CPU native state/write replay on 16 opened fixtures; two native-state inputs, no model import. No fresh behavioral certification.'
 (P/(STEM+'_STANDALONE_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
