"""Validate a weight-defined two-writer shortcut on existing native transport cache.
A: original writer span preserves raw9 change to <=10% error each family.
B: projected change, with RMS recomputed, preserves each QK/value read change
   to <=10% error each family. No fitted span or fresh/OOD claim.
"""
from pathlib import Path
import json,torch
import torch.nn.functional as F
from writer_span_coordinates_v1 import prepare
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
 program=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
 rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
 basis=prepare(program['writers'])['basis'];basis=basis/basis.shape[-1]**.5
 raw=cache['raw9'].double();delta=raw[1]-raw[0];projected=(delta@basis.T)@basis
 eps=torch.finfo(torch.float32).eps
 def norm(x):return F.rms_norm(x.float(),(1152,),eps=eps).double()
 base=norm(raw[0]);truth=norm(raw[1])-base;pred=norm(raw[0]+projected)-base
 readers={k:program[k][1].double() for k in ['q1','q2','k1','k2']};readers['value']=program['current_value_reader'].double()[None]
 records=[]
 for family in sorted(set(r['family'] for r in rows)):
  mask=torch.zeros(delta.shape[:2],dtype=torch.bool)
  for i,r in enumerate(rows):
   if r['family']==family:mask[i,:len(r['ids'])]=True
  def error(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
  reads={k:error(pred[mask]@v.T,truth[mask]@v.T) for k,v in readers.items()}
  records.append(dict(family=family,positions=int(mask.sum()),raw_change_relative_error=error(projected[mask],delta[mask]),normalized_change_relative_error=error(pred[mask],truth[mask]),reader_change_errors=reads))
 out={'pred_a':all(r['raw_change_relative_error']<=.1 for r in records),'pred_b':all(max(r['reader_change_errors'].values())<=.1 for r in records),'records':records,'scope':'Existing 72-prompt native removal cache; original two physical writers define span without fitting. Raw9 after original head8 path, before attention9. Recomputed RMS; linear QK reads before per-head normalization/RoPE/product, not routing/logit behavior. No claim that richer transported response bank fails.'}
 (P/'TRANSPORTED_WRITER_SPAN_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
