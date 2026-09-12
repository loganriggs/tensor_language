"""Frozen initial/pilot quartic effects using established native-tail developmental scorer."""
from pathlib import Path
import torch,json
import torch.nn.functional as F
from coupled_quartic_writer_v1 import features
from sparse_path_stability_atlas_v1 import digest
from stable_path_native_cache_v1 import relative
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 out=P/'QUARTIC_CONTINUED_NATIVE_EFFECTS_V1.json';assert not out.exists()
 bindings=json.loads((P/'COUPLED_QUARTIC_NONLINEAR_V2_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in bindings.items())
 result=json.loads((P/'COUPLED_QUARTIC_NONLINEAR_V2_RESULT.json').read_text());ap=P/'COUPLED_QUARTIC_NONLINEAR_V2_PROGRAM.pt';assert digest(ap)==result['artifact_sha256']
 initial=torch.load(P/'COUPLED_QUARTIC_NONLINEAR_V1_PROGRAM.pt',weights_only=True);pilot=torch.load(ap,weights_only=True)
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double();old=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports'];den=old['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps;h=old['pre']+old['native_output']
 ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0]
 def write(p):return features(p['input_readers'],p['inner_weights'],x)@p['mixing']@p['output_writers'].T/den[:,None]
 writes=[ref,write(initial),write(pilot)]
 rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows'];targets=torch.tensor([r[s+'_answer_id'] for r in rows for s in ('base','donor')])
 state=torch.load(next(k for k in bindings if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True);u=state['lm_head.weight'].float()
 def ce(states):
  values=[]
  for i in range(0,len(states),32):
   logits=30*torch.tanh(F.linear(F.rms_norm(states[i:i+32],(1152,)),u)/30)
   values.append(F.cross_entropy(logits,targets[i:i+32],reduction='none').double())
  return torch.cat(values)
 baseline=ce(h);effects=[]
 for w in writes:
  zero=ce(h-w.float())-baseline;swapped=h[::2]+(w[1::2]-w[::2]).float();values=[]
  for i,r in enumerate(rows):
   readers=u[[r['donor_answer_id'],r['donor_foil_id']]]
   z=30*torch.tanh(F.linear(F.rms_norm(torch.stack([h[2*i],swapped[i]]),(1152,)),readers)/30)
   values.append(float((z[1,0]-z[1,1])-(z[0,0]-z[0,1])))
  effects.append((zero,torch.tensor(values,dtype=torch.float64)))
 prior=next(r for r in json.loads((P/'QUARTIC_GROUP_LIFTED_NATIVE_V1.json').read_text())['effects'] if r['name']=='lifted')
 replay=max(float((effects[0][0]-torch.tensor(prior['zero_ce'][0],dtype=torch.float64)).abs().max()),float((effects[0][1]-torch.tensor(prior['swaps'][0],dtype=torch.float64)).abs().max()))
 reports=[]
 for idx,name in [(1,'initial'),(2,'continued')]:
  zero,swaps=effects[idx];rz,rs=effects[0];families=[]
  for family in sorted({r['family'] for r in rows}):
   ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten();a,b=swaps[ids],rs[ids];live=torch.maximum(a.abs(),b.abs())>=1e-4
   families.append(dict(family=family,swap_relative_rms=relative(a,b),swap_live=int(live.sum()),swap_sign_agreement=float((a[live].sign()==b[live].sign()).double().mean()),zero_ce_meanabs_disagreement=float((zero[ep]-rz[ep]).abs().mean()),zero_ce_mean=[float(zero[ep].mean()),float(rz[ep].mean())]))
  reports.append(dict(name=name,physical_error=float((writes[idx]-ref).norm()/ref.norm()),families=families))
 errors=[abs(reports[0]['physical_error']-result['initial_native_error']),abs(reports[1]['physical_error']-result['final_native_error'])];fams=reports[1]['families']
 r={'pred_a':max(errors)<=1e-8 and replay<=1e-5,'pred_b':all(f['swap_relative_rms']<=.1 and f['swap_sign_agreement']>=.9 and f['swap_live']>=4 for f in fams),'pred_c':all(f['zero_ce_meanabs_disagreement']<=.02 for f in fams),'reports':reports,'write_replay_errors':errors,'prior_reference_effect_max_error':replay,'artifact_sha256':digest(ap),'scope':'Frozen partialcomponent fidelity on reuseddevelopmentalrows, nativebackground retained; no OOD/selectivity proof.'}
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert r['pred_a']
if __name__=='__main__':main()
