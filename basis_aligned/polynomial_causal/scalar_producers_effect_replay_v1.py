"""Fresh fixed producer components, shared output, joint and removal effects."""
from pathlib import Path
import torch,json,time
import torch.nn.functional as F
from compiled_reading_head_v1 import execute_reading_head
from folded_normalized_router_v1 import rotary
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();stem='SCALAR_PRODUCERS_EFFECT_REPLAY_V1';out=P/(stem+'_RESULT.json');assert not out.exists()
 native=json.loads((P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_RESULT.json').read_text());assert native['pred_a'] and native['pred_c']
 c=torch.load(P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');bank=torch.load(P/'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');merged=torch.load(P/'STRUCTURED_PRODUCER_SHARED_OUTPUT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];validate(rows);ix=[2,17];heads=c['head_reads'][ix]
 components=torch.einsum('hnti,hji->hntj',heads,bank['projectors'][ix]);merge=torch.einsum('hnti,hji->hntj',heads,merged['projectors'])
 compiled_result=json.loads((P/'SCALAR_PRODUCERS_NATIVE_V1_RESULT.json').read_text());assert all(compiled_result[k] for k in ('pred_a','pred_b','pred_c'))
 compiled=torch.load(P/'SCALAR_PRODUCERS_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['contributions']
 base_contributions=torch.cat([merge,merge.sum(0,keepdim=True),compiled,compiled.sum(0,keepdim=True)])
 contributions=torch.cat([base_contributions,base_contributions])
 assert contributions.shape==(12,48,22,4)
 writes=torch.zeros(13,48,1152);checks=[]
 for length in sorted(set(len(r['ids']) for r in rows)):
  ids=[i for i,r in enumerate(rows) if len(r['ids'])==length];donors=[i^1 for i in ids];query=c['current_states'][ids,length-1];qr=rotary(length-1,128)
  for pos in range(length):
   current=c['current_states'][ids,pos];tokens=torch.tensor([rows[i]['ids'][pos] for i in ids]);rotation=(qr.T@rotary(pos,128)).float();f=current.double()@p['current_readers'].T+p['token_reads'][tokens]
   base=execute_reading_head(query,current,f,rotation,p);writes[0,ids]+=base;ref=c['position_writes'][ids,pos];checks.append(float((base-ref).norm()/ref.norm().clamp_min(1e-30)))
   for arm in range(12):
    delta=contributions[arm,donors,pos]-contributions[arm,ids,pos] if arm<6 else -contributions[arm,ids,pos]
    writes[arm+1,ids]+=execute_reading_head(query,current,f+delta/c['rho'][ids,pos],rotation,p)
 binding=json.loads((P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_BINDING.json').read_text())['files'];state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 ids=torch.tensor([[r['uk_id'],r['us_id'],*r['control_ids'],*r['newline_control_ids']] for r in rows]);vocab,inverse=torch.unique(ids,sorted=True,return_inverse=True);u=state['lm_head.weight'][vocab].float();l,r,d=[state['transformer.h.17.mlp.'+k+'.weight'].float() for k in ('Left','Right','Down')];bias=state['transformer.h.17.mlp.Down_bias'].float()
 def margins(delta):
  z=c['pre']+delta;x=F.rms_norm(z,(1152,));h=z+F.linear(F.linear(x,l)*F.linear(x,r),d,bias);logits=30*torch.tanh(F.linear(F.rms_norm(h,(1152,)),u)/30);v=logits.gather(1,inverse);return torch.stack([v[:,0]-v[:,1],v[:,2]-v[:,3],v[:,4]-v[:,5]],1).double()
 baseline=margins(torch.zeros_like(writes[0]));effects=torch.stack([margins(writes[i+1]-writes[0])-baseline for i in range(12)])
 def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 baseline_error=rel(baseline,c['baseline_margins']);cells=[]
 for family in range(2):
  ids=[i for i,row in enumerate(rows) if row['family']==family]
  for mode,offset in [('swap',0),('removal',6)]:
   for head in range(3):
    reference=effects[offset+head,ids];actual=effects[offset+3+head,ids]
    cells.append(dict(family=family,mode=mode,component=head,relative_effect_error=rel(actual,reference),regional_error=rel(actual[:,0],reference[:,0]),maximum_absolute_error=float((actual-reference).abs().max())))
 A=max(checks)<=1e-5 and baseline_error<=1e-4
 result={'pred_a':A,'pred_b':A and all(c['relative_effect_error']<=1e-4 for c in cells if c['mode']=='swap'),'pred_c':A and all(c['relative_effect_error']<=1e-4 for c in cells if c['mode']=='removal'),'baseline_replay':baseline_error,'noedit_replay':max(checks),'cells':cells,'seconds':time.perf_counter()-tic,'scope':'Compiled scalar producers versus frozen shared-output projected native contributions, individual/joint swaps and removals on48freshcontexts. Native query/current generators and downstream background remain required; execution fidelity, not another semantic discovery.'}
 torch.save(dict(baseline=baseline,effects=effects),P/(stem+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
