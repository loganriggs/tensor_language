"""Prune unused attention weights from the freshly tested normalization candidate."""
from pathlib import Path
import json,hashlib,sys,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);out=P/'extracted_circuits/typed_face_single_head_norm_v1';out.mkdir(exist_ok=False)
 local=torch.load(P/'extracted_circuits/typed_face_mlp8_coupled_v1/program.pt',weights_only=True)
 tables=torch.load(P/'SINGLE_HEAD_FRESH_V1_TABLES.pt',weights_only=True)
 local['head8']['token_ids']=tables['token_ids'];local['head8']['first_table']=tables['first_table'][:,256:384].clone()
 program={**local,'entry':{k:tables[k] for k in ['token_ids','initial_table','lambdas8']}}
 torch.save(program,out/'program.pt')
 (out/'head8.py').write_text((P/'extracted_circuits/typed_face_reduced_fused_v1/head8.py').read_text())
 channels=(P/'attention8_single_head_channels_v1.py').read_text().replace("p['first_table'][:,256:384]","p['first_table']").replace('p[name][256:384]','p[name]').replace("p['value'][256:384]","p['current_value']")
 (out/'attention.py').write_text(channels)
 (out/'execute.py').write_text('''"""Single-head frozen-norm approximation, with two declared native inputs."""
import torch
import torch.nn.functional as F
import head8
from attention import channels

def execute(p,residual7,donor_city7,token_ids,recipient_token,donor_token,city,destination,strength=.5):
 entry=p['entry'];head=p['head8'];lookup={int(t):i for i,t in enumerate(entry['token_ids'].tolist())}
 try:
  idx=torch.tensor([[lookup[int(t)] for t in row] for row in token_ids.tolist()],device=residual7.device)
  donor_idx=lookup[int(donor_token)]
 except KeyError as e:raise ValueError('Token outside frozen initial-state table') from e
 initial=F.embedding(idx,entry['initial_table']).to(residual7.dtype);lam=entry['lambdas8']
 mixed=lam[0]*residual7+lam[1]*initial
 current8=F.rms_norm(mixed,(1152,))
 donor_initial=entry['initial_table'][donor_idx][None].to(donor_city7.dtype)
 donor8=F.rms_norm(lam[0]*donor_city7+lam[1]*donor_initial,(1152,))
 values=channels(head,current8,token_ids).reshape(*current8.shape[:2],128)
 g=mixed+F.linear(values,head['output'])
 d=(strength*head8.execute(head,current8,donor8,recipient_token,donor_token,city,destination)).double()
 z=g.double();mlp=p['mlp8'];L=mlp['left'].double();R=mlp['right'].double();D=mlp['down'].double()
 ld=d@L.T;rd=d@R.T;lr=z@L.T;rr=z@R.T
 s=z.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps
 return d+((ld*rr+lr*rd+ld*rd)/s)@D.T
''')
 sys.path.insert(0,str(out));import execute
 fixtures=torch.load(P/'SINGLE_HEAD_FRESH_V1_ARTIFACT.pt',weights_only=True)['fixtures'];errors=[];zeros=[]
 for f in fixtures:
  x=f['candidate_inputs'];got=execute.execute(program,**x);ref=f['expected_candidate_delta'];errors.append(float((got-ref).norm()/ref.norm()));zeros.append(float(execute.execute(program,**x,strength=0).abs().max()))
 def floats(v):return sum(floats(x) for x in v.values()) if isinstance(v,dict) else (v.numel() if v.is_floating_point() else 0)
 count=floats(program)
 result={'pred_a':len(fixtures)==40 and max(errors)<=1e-4,'pred_b':max(zeros)==0,'fixtures':len(fixtures),'max_relative_write_error':max(errors),'max_zero_strength':max(zeros),'float_scalars':count,'native_state_arrays':2,'native_state_scalars_T32':38016,'supported_sequence_tokens':len(tables['token_ids']),'scope':'Pruned program CPU replay against saved fresh-run candidate writes, now opened. Native readout and isolated import replay of this package pending. Fresh scientific evidence belongs to unchanged formula; no composition or token-only claim.'}
 assert result['pred_a'] and result['pred_b']
 (P/'SINGLE_HEAD_PRUNED_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 manifest={'name':out.name,'entrypoint':'execute.execute','float_scalars':count,'supported_sequence_tokens':len(tables['token_ids']),'native_inputs':['residual7[B,T,1152]','donor_city7[B,1152]'],'other_inputs':['token_ids','recipient_token','donor_token','city','destination','strength=.5'],'external_suffix':'native blocks9–17 and readout','counterfactual':'approximate retained head8.2 donor half-write coupled to MLP8 with head8.2-only frozen normalization','validation':result,'four_traits':{'prediction':'fresh formula test passed; same endpoints','extraction':'CPU replay passes; independent native/layout certificate pending','selectivity':'fresh formula same-site null/four-reader test passed','composition':False},'complete_circuit':False,'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (out/'README.md').write_text('# Single-head regional normalization candidate\n\nTwo native residual7 inputs; later suffix external. Only head8.2 attention weights are stored. Normalization uses mixed residual8 plus head8.2, frozen across the intervention. This approximates the full-context response; it is not an exact fold of that response.\n\nLoad `program.pt` with PyTorch and call `execute.execute(program, **inputs)`. Supported token IDs are explicitly enumerated; others are rejected.\n\nCPU fixture replay passes. Independent native readout and isolated-import checks of this layout remain pending. Fresh formula evidence: SINGLE_HEAD_FRESH_V1_RESULT.json (all six gates); same endpoints, paired intervention, no composition claim.\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
