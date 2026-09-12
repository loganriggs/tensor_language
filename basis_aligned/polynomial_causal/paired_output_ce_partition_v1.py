"""Exact loss partition for component removal; no fit or new data."""
from pathlib import Path
import json,torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 binding=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 path=P/'QUARTIC_OUTER32_FRESH_V1_PORTS.pt';receipt=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_RESULT.json').read_text());assert digest(path)==receipt['artifact_sha256']
 cache=torch.load(path,weights_only=True);ports=cache['ports'];h=ports['pre']+ports['native_output'];write=cache['reference_write'].float()
 rows=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_ROWS.json').read_text())['rows'];targets=torch.tensor([r[s+'_answer_id'] for r in rows for s in ('base','donor')]);foils=torch.tensor([r[s+'_foil_id'] for r in rows for s in ('base','donor')])
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True);u=state['lm_head.weight'].float()
 terms=[];identities=[]
 for states in (h,h-write):
  records=[]
  for i in range(0,len(h),32):
   z=(30*torch.tanh(F.linear(F.rms_norm(states[i:i+32],(1152,)),u)/30)).double()
   target=targets[i:i+32];foil=foils[i:i+32];a=z.gather(1,target[:,None]).flatten();b=z.gather(1,foil[:,None]).flatten()
   ce=F.cross_entropy(z,target,reduction='none');binary=F.softplus(b-a);mass=torch.logsumexp(z,-1)-torch.logaddexp(a,b)
   identities.append(float((ce-binary-mass).abs().max()));records.append(torch.stack([ce,binary,mass],1))
  terms.append(torch.cat(records))
 delta=terms[1]-terms[0];families=[]
 for family in sorted({r['family'] for r in rows}):
  ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten();d=delta[ep]
  families.append(dict(family=family,mean_signed_ce_binary_pairmass=d.mean(0).tolist(),mean_absolute_ce_binary_pairmass=d.abs().mean(0).tolist(),positive_binary_damage_fraction=float((d[:,1]>0).double().mean()),pairmass_fraction_of_signed_mean=float(d[:,2].mean()/d[:,0].mean())))
 r=dict(pred_a=max(identities)<1e-10,maximum_identity_error=max(identities),families=families,per_endpoint_effects=delta.tolist(),cache_sha256=digest(path),scope='Exact reference component removal on already inspected rows. Full CE damage partitions into answer/foil binary CE and mass outside that pair; neither alone identifies a semantic circuit. Native float32 logits with float64 loss accounting, no fit.')
 out=P/'PAIRED_OUTPUT_CE_PARTITION_V1.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='per_endpoint_effects'},indent=2));assert r['pred_a']
if __name__=='__main__':main()
