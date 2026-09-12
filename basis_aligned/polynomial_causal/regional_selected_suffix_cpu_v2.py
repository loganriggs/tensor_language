"""CPU replay of completed query/source edits using selected vocabulary rows.
Bar: baseline and effects relative<=1e-4. Selected logits, not full CE.
"""
from pathlib import Path
import torch,json,time
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'REGIONAL_SELECTED_SUFFIX_CPU_V2_RESULT.json';assert not out.exists()
 binding=json.loads((P/'REGIONAL_QUERY_SOURCE_EFFECT_V1_BINDING.json').read_text())['files'];ck=next(k for k in binding if k.endswith('pytorch_model.bin'));state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
 a=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True);b=torch.load(P/'REGIONAL_QUERY_SOURCE_ACCOUNTING_V1_ARTIFACT.pt',weights_only=True);prior=torch.load(P/'REGIONAL_QUERY_SOURCE_EFFECT_V1_ARTIFACT.pt',weights_only=True)
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[])
 ids=torch.tensor([[r['uk_id'],r['us_id'],*r['control_ids']] for r in rows]);vocab,inverse=torch.unique(ids,sorted=True,return_inverse=True);u=state['lm_head.weight'][vocab].float()
 l,r,d=[state['transformer.h.17.mlp.'+k+'.weight'].float() for k in ('Left','Right','Down')];bias=state['transformer.h.17.mlp.Down_bias'].float();pre=a['pre'];base=b['base_write'];times=[]
 def margins(delta):
  start=time.perf_counter();z=pre+delta;x=F.rms_norm(z,(1152,));h=z+F.linear(F.linear(x,l)*F.linear(x,r),d,bias);logits=30*torch.tanh(F.linear(F.rms_norm(h,(1152,)),u)/30);picked=logits.gather(1,inverse);result=torch.stack([picked[:,0]-picked[:,1],picked[:,2]-picked[:,3]],1).double();times.append(time.perf_counter()-start);return result
 baseline=margins(torch.zeros_like(base));effects=[]
 for k,bit in enumerate((1,2)):
  donor=torch.arange(96)^bit;effects.append(torch.stack([margins(base[donor]-base)-baseline,margins(b['hybrid_writes'][k,0]-base)-baseline,margins(b['hybrid_writes'][k,1]-base)-baseline]))
 effects=torch.stack(effects)
 def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 checks=dict(baseline=rel(baseline,prior['baseline']),effects=rel(effects,prior['effects']),baseline_maxabs=float((baseline-prior['baseline']).abs().max()),effects_maxabs=float((effects-prior['effects']).abs().max()))
 result=dict(pred_a=checks['baseline']<=1e-4 and checks['effects']<=1e-4,checks=checks,selected_token_count=len(vocab),suffix_seconds=times,total_seconds=time.perf_counter()-tic,weight_tensor_bytes=sum(x.numel()*x.element_size() for x in (l,r,d,bias,u)),scope='CPU selected-logit oracle for cached last-MLP inputs and edits, validated against a completed GPU experiment. Not a duplicate of queued read/norm test, not full vocabulary CE or upstream extraction.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
