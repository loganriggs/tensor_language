"""Nine ordered upstream products for the three complete city-side readers."""
import hashlib,itertools,json
from pathlib import Path
import torch
from city_mlp7_integrated_v2 import execute,write_from_readers
P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def main():
 program=torch.load(P/'CITY_MLP7_READERS_V1_PROGRAM.pt',weights_only=True)
 head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
 tables=torch.load(P/'CITY_FULL_PILE_V2_TABLES.pt',weights_only=True);lookup={int(t):i for i,t in enumerate(tables['token_ids'])}
 capture=torch.load(P/'CITY_SOURCE7_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 refs=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 integrated=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
 binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files']
 checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'))
 state=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
 lam7=state['transformer.h.7.lambdas'].float().double();lam8=tables['lambdas8'][0].double()
 L,R,C,c=[program[k].double() for k in ['left','right','folded_down','folded_bias']]
 labels=['residual6','initial7','attention7'];all_terms=[];full_readers=[];self_readers=[];closures=[];full_errors=[];self_errors=[];writes=[];helper_errors=[]
 eps=torch.finfo(torch.float32).eps
 for f,ref,cache in zip(capture,refs,integrated,strict=True):
  city=ref['candidate_inputs']['city'];token=int(ref['candidate_inputs']['token_ids'][0,city])
  initial=lam7[1]*tables['initial_table'][lookup[token]].double()[None]
  mixed=f['sources'][0].double()/lam8;attention=f['sources'][1].double()/lam8
  sources=torch.stack([mixed-initial,initial,attention]);u=sources.sum(0);rho=u.square().mean(-1,keepdim=True)+eps
  left=sources@L.T;right=sources@R.T
  terms={f'{labels[i]}*{labels[j]}':(left[i]*right[j])@C.T/rho for i,j in itertools.product(range(3),repeat=2)}
  total=sum(terms.values())+c;direct=((u@L.T)*(u@R.T))@C.T/rho+c
  self_only=sum(terms[f'{name}*{name}'] for name in labels)+c
  closures.append(float((total-direct).norm()/direct.norm()))
  context={k:v for k,v in cache['inputs'].items() if k!='normalized_mlp7_city'}
  got=write_from_readers(total,head,**context);lean=write_from_readers(self_only,head,**context)
  replay=execute(program,head,**cache['inputs'])
  helper_errors.append(float((replay-cache['delta']).norm()/cache['delta'].norm()))
  target=ref['inputs']['delta'].double();full_errors.append(float((got-target).norm()/target.norm()));self_errors.append(float((lean-target).norm()/target.norm()))
  writes.append({'full':got,'self_only':lean,'native':target});all_terms.append(terms);full_readers.append(total);self_readers.append(self_only)
 full=torch.cat(full_readers);lean=torch.cat(self_readers);pair=full[::2]-full[1::2];leanpair=lean[::2]-lean[1::2]
 errors=[float((a-b).norm()/a.norm()) for a,b in zip(pair.split(128,-1),leanpair.split(128,-1))]
 den=float(pair.square().sum());stats={}
 for name in all_terms[0]:
  values=torch.cat([d[name] for d in all_terms]);contrast=values[::2]-values[1::2]
  stats[name]={'norm_ratio':float(contrast.norm()/pair.norm()),'aligned_fraction':float((contrast*pair).sum()/den)}
 target=torch.cat([w['native'] for w in writes]);leanwrite=torch.cat([w['self_only'] for w in writes]);aggregate=float((leanwrite-target).norm()/target.norm())
 result={'pred_a':max(closures)<=1e-10,'pred_b':max(full_errors)<=1e-4,'pred_c':min(errors)>=.10,'pred_d':aggregate>=.10,
         'max_closure_relative':max(closures),'max_full_write_relative':max(full_errors),'helper_v2_replay_relative':max(helper_errors),
         'self_only_paired_reader_errors':dict(zip(['k1','k2','current_value'],errors)),'self_only_aggregate_write_error':aggregate,
         'self_only_max_sequence_write_error':max(self_errors),'paired_terms':stats,'effective_document_pairs':20,
         'scope':'Opened conditional input-source fold. Native query/key RMS/mixed8 RMS supplied. Residual6 source recovered by subtraction, not independently captured. No causal omission or independent-composition claim.',
         'source_shas':{str(P/name):hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['CITY_MLP7_INPUT_TERMS_V1_PREREGISTRATION.md','city_mlp7_input_terms_v1.py','city_mlp7_integrated_v2.py','CITY_MLP7_READERS_V1_PROGRAM.pt']}}
 torch.save({'writes':writes,'reader_terms':all_terms},P/'CITY_MLP7_INPUT_TERMS_V1_ARTIFACT.pt')
 (P/'CITY_MLP7_INPUT_TERMS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2))
if __name__=='__main__':main()
