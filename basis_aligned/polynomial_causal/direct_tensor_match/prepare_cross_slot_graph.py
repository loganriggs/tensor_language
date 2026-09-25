"""Instantiate the frozen mixed-slot topology proposal from learned native readers."""
from pathlib import Path
import json,torch
from local_shared_reader_graph import expand,factor_bundle,price
from pack_reader_graph_artifacts import packed,counts
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
def main():
 proposal=json.loads((P/'OVERLAP_CROSS_SLOT_TOPOLOGY_V1.json').read_text());programs=torch.load(P/'JOINT_OVERLAP_PROGRAMS_V1.pt',weights_only=True);out={};records=[]
 for row in proposal['proposals']:
  parent=programs[row['source_program']];bundle=expand(parent);basis=torch.linalg.qr(parent['input_basis'],mode='reduced').Q;selections={k:torch.tensor(v,dtype=torch.int64) for k,v in row['shared_indices'].items()};new=packed(factor_bundle(bundle,basis,selections));actual=price(new);assert actual==row['price'] and counts(new)['backing_storage_floats']==996876
  cross=[]
  for j in range(3):
   p=new['pairs'][str(j)];mask=torch.zeros(384,dtype=torch.bool);mask[p['shared_indices']]=True;i,k,kind=p['product_indices'];cross.append(int(((mask[i]!=mask[k])&(kind==2)).sum()))
  assert cross==[c['proposed_cross_products'] for c in row['pair_cross_counts']]
  key=row['geometry']+'_128_160';out[key]=new;records.append(dict(key=key,parent=row['source_program'],cross_counts=cross,**actual))
 torch.save(out,P/'CROSS_SLOT_INITIAL_PROGRAMS_V1.pt');(P/'CROSS_SLOT_INITIAL_V1.json').write_text(json.dumps(dict(records=records,scope='Wiring and initialization only, no fitted result. Native source targets and all later component parameters unchanged. Shared-slot reprojection changes the approximate source function; readout and affine corrections are refitted by the existing optimizer.'),indent=2)+'\n');print(records)
if __name__=='__main__':main()
