"""CPU identity against the old conditional value path; no new model execution."""
from pathlib import Path
import hashlib,json
import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from odd_source_swap_interaction_v1 import source_factors
from odd_source_positions_v1 import select_sources
from odd_value_source_split_v1 import value_parts
from odd_value_delta_from_raw_v1 import execute
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(17092212)
 path=P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt';g=SharedGraph(torch.load(path,weights_only=True,map_location='cpu'))
 errors=[];changed_first_errors=[];zero=[]
 for T in [13,29,32]:
  raw=torch.randn(1,T,1152);delta=.01*torch.randn_like(raw);mask=torch.arange(T)%3!=0;delta[:,~mask]=0
  current=F.rms_norm(raw,(1152,));changed=F.rms_norm(raw+.83*delta,(1152,));first=torch.randn(1,T,9,128)
  routing,_=source_factors(g,current,current,current,first);a,_=value_parts(g,current,first);b,_=value_parts(g,changed,first)
  old=select_sources(routing*(b-a),mask[None])@g.p['output'].double().T
  new=execute(g,raw,delta,.83,mask);errors.append(float((new-old).norm()/old.norm()))
  other=first*100+7;a2,_=value_parts(g,current,other);b2,_=value_parts(g,changed,other)
  changed_first_errors.append(float(((b2-a2)-(b-a)).abs().max()))
  zero.append(float(execute(g,raw,torch.zeros_like(delta),.83,mask).abs().max()))
 result={'pred_a':max(errors)<=1e-10,'pred_b':max(changed_first_errors)==0,'pred_c':max(zero)==0,'relative_errors':errors,'first_value_independence_max_abs':max(changed_first_errors),'zero_delta_max_abs':max(zero),
  'scope':'CPU exact odd-value difference identity. Removes first-value input and combines block9 residual/initial into one raw-mixed native port. FP32 reassociation from original reentry is not yet verified on native data; full composed extraction/suffix remains incomplete.',
  'program_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 (P/'ODD_VALUE_DELTA_RAW_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert all(result[k] for k in ['pred_a','pred_b','pred_c'])
if __name__=='__main__':main()
