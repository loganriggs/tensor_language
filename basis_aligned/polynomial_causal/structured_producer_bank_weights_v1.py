"""Frozen consumer-metric value component for every selected producer head."""
import json,time,torch
from pathlib import Path
from consumer_pullback_metric_v1 import weighted_component
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_RESULT.json';assert not out.exists()
 assert json.loads((P/'CONSUMER_PULLBACK_V1_CONTROL.json').read_text())['pred_a']
 prior=torch.load(P/'CONSUMER_PULLBACK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');H=prior['metric']
 binding=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files'];state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');C=p['current_readers'];projectors=[];sources=[];writers=[];rows=[]
 for layer in (8,9,13):
  prefix=f'transformer.h.{layer}.attn.';scale=1.
  for j in range(layer+1,18):scale*=float(state[f'transformer.h.{j}.lambdas'][0])
  mix=float(state[prefix+'lamb']);fold=(scale*C@state[prefix+'c_proj.weight'].double()).reshape(4,9,128)
  current=state[prefix+'c_v.weight'].double().reshape(9,128,1152);first=state['transformer.h.0.attn.c_v.weight'].double().reshape(9,128,1152)
  for head in range(9):
   M=fold[:,head]@torch.cat([(1-mix)*current[head],mix*first[head]],1)
   selected,projector,reader,singular=weighted_component(M,H)
   rows.append(dict(layer=layer,head=head,weighted_value_rank1_capture=float(singular[0].square()/singular.square().sum()),projected_map_replay=float((projector@M-selected).norm()/selected.norm()),projector_idempotence=float((projector@projector-projector).norm()/projector.norm())))
   projectors.append(projector);sources.append(reader);writers.append(M@reader)
 projectors=torch.stack(projectors);replay=float((projectors[18]-prior['projector']).norm()/prior['projector'].norm())
 result={'pred_a':max(r['projected_map_replay'] for r in rows)<=1e-10,'pred_b':max(r['projector_idempotence'] for r in rows)<=1e-10,'pred_c':replay<=1e-10,'head13_0_projector_replay':replay,'rows':rows,'seconds':time.perf_counter()-tic,'scope':'One fixed weights-only component per producer head under identical consumer metric. No behavior-based component selection, fitted data, or rank sweep. Native QK and input generators retained.'}
 torch.save(dict(projectors=projectors,source_readers=torch.stack(sources),output_writers=torch.stack(writers),metric=H),P/'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_ARTIFACT.pt')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
