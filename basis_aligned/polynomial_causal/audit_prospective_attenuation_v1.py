"""Opened diagnostic of the failed minimum attenuation gate; no new threshold."""
from pathlib import Path
import hashlib,json
import torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 rows=json.loads((P/'TYPED_FACE_PROSPECTIVE_V1_ROWS.json').read_text())['rows']
 path=P/'TYPED_FACE_PROSPECTIVE_V1_ARTIFACT.pt';v=torch.load(path,weights_only=True,map_location='cpu')['values'];result={}
 for family in dict.fromkeys(r['variant'] for r in rows):
  idx=[i for i,r in enumerate(rows) if r['variant']==family];z=v[:,idx]
  gap=z[0,::2,0]-z[0,1::2,0];after=z[3,::2,0]-z[3,1::2,0];damage=gap-after
  result[family]={'native_gap_mean_logits':float(gap.mean()),'paired_damage_mean_logits':float(damage.mean()),'paired_damage_rms_logits':float(damage.square().mean().sqrt()),'mean_fraction':float((damage/gap).mean()),'ratio_of_means':float(damage.mean()/gap.mean()),
  'cell_means':[{'context_id':cid,'gap':float(gap[[j for j,r in enumerate([rows[k] for k in idx][::2]) if r['context_id']==cid]].mean()),'damage':float(damage[[j for j,r in enumerate([rows[k] for k in idx][::2]) if r['context_id']==cid]].mean())} for cid in dict.fromkeys(rows[k]['context_id'] for k in idx)]}
 out={'families':result,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Opened damage-versus-native-gap diagnostic. Mean ratio is the original gate; ratio of means is descriptive only and cannot rescue it. Does not localize downstream response or identify causal modules.'}
 (P/'PROSPECTIVE_ATTENUATION_V1_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:{a:b for a,b in x.items() if a!='cell_means'} for k,x in result.items()},indent=2))
if __name__=='__main__':main()
