"""Frozen input-corner fields for Q7 by first-token-value interaction."""
from pathlib import Path
import json,torch,importlib.util
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);a=torch.load(P/'FIRST_TOKEN_PATH_FRESH_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];pr=torch.load(P/'extracted_circuits/first_token_value_path_v1/program.pt',weights_only=True);f=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in f if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);s=importlib.util.spec_from_file_location('tok',P/'extracted_circuits/first_token_value_path_v1/execute.py');exe=importlib.util.module_from_spec(s);s.loader.exec_module(exe);fields=torch.zeros(96,4,a['fields'].shape[-1],dtype=torch.float64);hybrid=[]
 for i,row in enumerate(rows):
  j=row['donor_id'];n=len(row['ids']);ids=torch.tensor(row['ids']);dids=torch.tensor(rows[j]['ids']);c=int(torch.where(ids!=dids)[0].item());F=exe.token_readings(ids,sd['transformer.wte.weight'],pr);dF=exe.token_readings(dids,sd['transformer.wte.weight'],pr)-F;q=a['q7'][i,:n];dq=a['q7'][j,:n]-q;h=a['gamma8'][i,:n,:n]@F;dh=a['gamma8'][i,:n,:n]@dF;mask=torch.arange(n)>c;scale=2*pr['head9_gain']*mask/(a['rho8_squared'][i,:n]*a['rho9'][i,:n]);vF=(q*pr['eigenvalues']*dh).sum(-1);vQ=(dq*pr['eigenvalues']*h).sum(-1);vM=(dq*pr['eigenvalues']*dh).sum(-1)
  for arm,v in enumerate([vF,vQ,vF+vQ+vM,vM]):fields[i,arm,:n]=a['gamma9'][i,:n,:n]@(scale*v)
  if i>=84 and i%2:hybrid.append(dict(row=i,nativeF_final=float(fields[i,0,n-1]),F_afterQ_final=float(fields[i,0,n-1]+fields[i,3,n-1])))
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));res=dict(F_replay=rel(fields[:,0],a['fields'][:,1]),sum_identity=rel(fields[:,0]+fields[:,1]+fields[:,3],fields[:,2]),Baltimore_hybrid_final_writes=hybrid,scope='Recipientrouting/norms; Q7 andfirstvaluesdonatedonlyinspecifiedcrosspath. Algebraicfields awaitingphysicalsuffix. No fullQ-containingpath/module edit.')
 assert res['F_replay']<1e-5 and res['sum_identity']<1e-10;torch.save(dict(fields=fields),P/'FIRST_TOKEN_Q_INTERACTION_V1_FIELDS.pt');(P/'FIRST_TOKEN_Q_INTERACTION_V1_CONTROL.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
if __name__=='__main__':main()
