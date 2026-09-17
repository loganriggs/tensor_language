"""Replay native joint-delta fixtures from a copied package, without repo imports."""
from pathlib import Path
import tempfile,shutil,subprocess,json
P=Path(__file__).resolve().parent;STEM='TYPED_FACE_REDUCED_RESIDUAL7_V1'
def main():
 with tempfile.TemporaryDirectory(prefix='coupled-isolated-') as tmp:
  root=Path(tmp);shutil.copytree(P/'extracted_circuits/typed_face_reduced_residual7_v1',root/'package',ignore=shutil.ignore_patterns('__pycache__'))
  shutil.copyfile(P/(STEM+'_ARTIFACT.pt'),root/'fixtures.pt')
  (root/'check.py').write_text('''from pathlib import Path
import sys,json,torch
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'package'))
import execute
p={'local':torch.load(root/'package/local_program.pt',weights_only=True,map_location='cpu'),'reentry':torch.load(root/'package/reentry_program.pt',weights_only=True,map_location='cpu'),'context':torch.load(root/'package/context_program.pt',weights_only=True,map_location='cpu')};fixtures=torch.load(root/'fixtures.pt',weights_only=True,map_location='cpu')['fixtures'];torch.set_num_threads(2)
errors=[]
for f in fixtures:
 actual=execute.execute(p,**f['inputs']);target=f['expected_native_delta'];errors.append(float((actual-target).norm()/target.norm()))
x=dict(fixtures[0]['inputs']);zero=float(execute.execute(p,**x,strength=0).abs().max());x['donor_token']=-1
try:execute.execute(p,**x);rejected=False
except ValueError:rejected=True
r={'pred_d':len(errors)==40 and max(errors)<=1e-4 and zero==0 and rejected,'fixtures':len(errors),'max_native_delta_error':max(errors),'zero_strength_max':zero,'unknown_token_rejected':rejected,'scope':'Isolated copied package, learned weights and supplied native fixtures only; no repo/model import. Native states and whole suffix remain external.'}
(root/'result.json').write_text(json.dumps(r,indent=2)+'\\n')
''')
  subprocess.run([sys.executable,'-I',str(root/'check.py')],cwd=root,check=True)
  result=json.loads((root/'result.json').read_text());(P/(STEM+'_STANDALONE_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert result['pred_d']
if __name__=='__main__':
 import sys
 main()
