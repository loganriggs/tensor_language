"""Opened diagnostic separating native removal sign from surrogate prediction."""
from pathlib import Path
import torch,json,hashlib
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);rp=P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_ROWS.json';ap=P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_ARTIFACT.pt';doc=json.loads(rp.read_text());rows=doc['rows'];v=torch.load(ap,weights_only=True)['values'];records=[]
 for cell in doc['contexts']:
  idx=[i for i,r in enumerate(rows) if r['context_id']==cell['context_id']];base=v[0,idx,0][::2]-v[0,idx,0][1::2];record={**cell,'cities':[rows[idx[0]]['city'],rows[idx[1]]['city']],'native_pair_margins':base.tolist()}
  for arm,name in [(1,'native_removal'),(3,'candidate')]:
   pair=v[arm,idx,0][::2]-v[arm,idx,0][1::2];atten=(base-pair)/base;record[name]={'attenuations':atten.tolist(),'positive_endpoints':int((atten>0).sum()),'mean_attenuation':float(atten.mean())}
  records.append(record)
 result={'scope':'Post-result opened diagnostic. Registered fresh pred_d fails; no changed gates or removed cells. Native removal sign is distinct from approximation fidelity.','cells':records,'native_negative_cells':[r['context_id'] for r in records if r['native_removal']['positive_endpoints']<6],'candidate_negative_cells':[r['context_id'] for r in records if r['candidate']['positive_endpoints']<6],'source_shas':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [rp,ap]}}
 (P/'CITY_REMOVAL_ENDPOINT_V1_SIGN_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('cells','source_shas')},indent=2))
 for r in records:
  if r['candidate']['positive_endpoints']<6 or r['native_removal']['positive_endpoints']<6:print(json.dumps(r,indent=2))
if __name__=='__main__':main()
