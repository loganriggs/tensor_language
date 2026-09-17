"""CPU compositional implementation check, not causal composition evidence."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from typed_face_composed_raw_v1 import Program
from extracted_circuits.odd_attention8h2_typed_face_v1 import native
from even_value_shared_graph_v1 import SharedGraph
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from odd_source_positions_v1 import select_sources
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(17092219)
 p8=torch.load(P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_PROGRAM.pt',weights_only=True)
 p9=torch.load(P/'ODD_VALUE_DELTA_RAW_V1_PROGRAM.pt',weights_only=True)
 lam=torch.load(P/'ODD_VALUE_DELTA_RAW_V1_FIXTURES.pt',weights_only=True)[0]['lambda90']
 program=Program(p8,p9,lam);graph=SharedGraph(p9);errors=[]
 for T in [20,24,32]:
  current=F.rms_norm(torch.randn(1,T,1152),(1152,));donor=F.rms_norm(torch.randn(1,1152),(1152,));x=torch.randn_like(current);initial=torch.randn_like(current);raw=lam*x+.37*initial
  city=3;mask=(torch.arange(T)>city)&(torch.arange(T)<T-1);rt,dt=p8['token_ids'][:2].tolist()
  delta=.5*native.execute(p8,current,donor,rt,dt,city,mask)
  z0=F.rms_norm(raw,(1152,));z1=F.rms_norm(lam*(x+delta)+.37*initial,(1152,));first=torch.randn(1,T,128)
  routing,_=source_factors(graph,z0,z0,z0,first);a,_=value_parts(graph,z0,first);b,_=value_parts(graph,z1,first)
  expected=select_sources(routing*(b-a),mask[None])@p9['output'].double().T
  actual=program.execute(current,donor,raw,rt,dt,city,mask);errors.append(float((actual-expected).norm()/expected.norm()))
 result={'pred_a':max(errors)<=1e-4,'relative_errors':errors,'native_state_arrays':3,'native_state_scalars_T32':2*32*1152+1152,
 'stored_weight_scalars_including_lambda':sum(v.numel() for v in p8.values() if v.is_floating_point())+sum(v.numel() for v in p9.values())+1,
 'scope':'Synthetic CPU implementation composition versus legacy conditional path, with FP32 reassociation. No native end-to-end replay yet, no new causal composition/additivity or token-only closure; suffix external.'}
 (P/'TYPED_FACE_COMPOSED_RAW_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
if __name__=='__main__':main()
