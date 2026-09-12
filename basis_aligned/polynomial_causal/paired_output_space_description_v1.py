"""Describe the frozen two-output write subspace; token weights are not causality."""
from pathlib import Path
import json,torch
from tokenizers import Tokenizer
from sparse_path_stability_atlas_v1 import digest
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 b=json.loads((P/'PRODUCER_METRIC_OUTER32_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in b.items())
 p=torch.load(P/'PRODUCER_METRIC_OUTER32_V1_PROGRAM.pt',weights_only=True)
 state=torch.load(next(k for k in b if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True);u=state['lm_head.weight'].double();w=p['output_writers'];z=u@w;centered=z-z.mean(0)
 energy=centered.square().sum(-1);order=energy.argsort(descending=True)
 tokenizer=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
 def label(i):return tokenizer.decode([i]) if i<50257 else '<unused_'+str(i)+'>'
 top=[dict(id=i,token=label(i),energy_fraction=float(energy[i]/energy.sum()),loading=centered[i].tolist()) for i in order[:64].tolist()]
 rows=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_ROWS.json').read_text())['rows'];families=[]
 for family in ('A1','A2','past','progressive'):
  vectors=[]
  for row in rows:
   if row['family']!=family:continue
   vector=centered[row['donor_answer_id']]-centered[row['base_answer_id']]
   sign=1
   if family=='A1':sign=1 if row['group_number']%2 else -1
   elif family=='A2':sign=-1 if row['group_number']%2 else 1
   vectors.append(sign*vector)
  v=torch.stack(vectors);unit=v/v.norm(dim=1)[:,None];mean=unit.mean(0)
  families.append(dict(family=family,mean_unit_contrast=mean.tolist(),directional_coherence=float(mean.norm()),median_loading_contrast_norm=float(v.norm(dim=1).median())))
 q,_=torch.linalg.qr(torch.tensor([[.4,.7],[-.8,.3]],dtype=torch.float64));rotated=(centered@q).square().sum(-1)
 result=dict(pred_a=float((centered.T@centered-torch.eye(2)).norm())<1e-8 and float((energy-rotated).abs().max())<1e-12,
  centered_metric_identity_error=float((centered.T@centered-torch.eye(2)).norm()),top64=top,energy_concentration={str(n):float(energy[order[:n]].sum()/energy.sum()) for n in (16,64,128,1024)},unused_row_energy_fraction=float(energy[50257:].sum()/energy.sum()),families=families,
  scope='Centered logit loading of a fixed Gc-orthonormal two-output basis. Token energy invariant under basis rotation; individual coordinates are not canonical. Contrast coherence describes weights, not behavior or new semantic circuits.')
 out=P/'PAIRED_OUTPUT_SPACE_DESCRIPTION_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({**{k:v for k,v in result.items() if k!='top64'},'top_tokens':[x['token'] for x in top[:32]]},indent=2));assert result['pred_a']
if __name__=='__main__':main()
