#!/usr/bin/env python3
# BQGATE:52prefixes;576suffixes;180seconds.
"""pred_a anchor fields<=1e-4 and h9<=1e-6.
pred_b each fresh family native-interaction error<=2%target/5%control.
pred_c each family incremental error<=1%target/2.5%control.
pred_d positive native paired cue capability each family.
Price52prefixes/576suffixes/180sec; fresh templates, frozen weights, live fields.
"""
from pathlib import Path
from hashlib import sha256
import torch,json,sys,os,time,signal
import torch.nn.functional as F
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from fastload import load_model_fast
from live_crossfirst_prefix_v1 import prepare as live_prefix,assembled
from composed_mlp10_context_v1 import prepare
from composed_joint_response_v1 import branch
from diagonal_mlp10_branch_v1 import prepare as prepare_exact,evaluate as exact_post
from polynomial_mlp10_branch_v1 import prepare as prepare_three,evaluate as three_post
from regional_cue_row_check_v1 import validate
@torch.no_grad()
def main():
 files=json.loads((P/'THREE_TERM_FRESH_TEMPLATE_V1_BINDING.json').read_text())['files']
 assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
 rows=json.loads((P/'THREE_TERM_FRESH_TEMPLATE_V1_ROWS.json').read_text())['rows'];validate(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('52prefixes;576suffixes;180seconds');return
 out=P/'THREE_TERM_FRESH_TEMPLATE_V1_RESULT.json';assert not out.exists()
 signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
 model=load_model_fast().cuda().eval();weights=assembled.load_weights(model.state_dict(),'cuda')
 program={k:v.cuda() for k,v in torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True).items()}
 oldrows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'];child=torch.load(P/'CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt',weights_only=True)['fields'];parents=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True)['parent_fields']
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));anchors=[]
 for index in (0,24,48,72):
  ids=torch.tensor([oldrows[index]['ids']],device='cuda');live=live_prefix(model,ids,weights)
  x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
  for block in model.transformer.h[:10]:x,v1=block(x,v1,x0)
  anchors.append(dict(row=index,child_error=rel(live['child'][...,0],child[index].cuda()),parent_error=rel((live['child']+live['remainder'])[...,0],parents[index].cuda()),h9_error=rel(live['h9'],x)))
 A=all(r['child_error']<=1e-4 and r['parent_error']<=1e-4 and r['h9_error']<=1e-6 for r in anchors)
 if not A:
  out.write_text(json.dumps({'pred_a':False,'pred_b':False,'pred_c':False,'pred_d':False,'anchors':anchors,'scope':'Invalid live-prefix instrument; fresh rows not evaluated.'},indent=2)+'\n');return
 b9,b10=model.transformer.h[9],model.transformer.h[10];w=program['direction']
 matrices=[getattr(b10.attn,k).weight.double() for k in ('c_q','c_k','c_q2','c_k2','c_v')]
 mlp_matrices=[getattr(b10.mlp,k).weight.double() for k in ('Left','Right','Down')];cells=[]
 def post(z):
  z=z.float();return z+b10.mlp(F.rms_norm(z,(1152,)))
 for index,row in enumerate(rows):
  ids=torch.tensor([row['ids']],device='cuda');live=live_prefix(model,ids,weights)
  x0,v1=live['x0'],live['v1'];raw9,att9,z9,m9,h9=[live[k] for k in ('raw9','att9','z9','m9','h9')];a,b=live['child'],live['remainder']
  raw0=b10.lambdas[0]*h9+b10.lambdas[1]*x0;z0=raw0+b10.attn(F.rms_norm(raw0,(1152,)),v1)[0]
  context=prepare(z9,m9.double()-b9.mlp.Down_bias.double(),raw0,z0,v1.double(),program,float(b10.lambdas[0]),matrices,float(b10.attn.lamb),b10.attn.c_proj.weight.double())
  exact=prepare_exact(context,*mlp_matrices,b10.mlp.Down_bias.double());three=prepare_three(context,*mlp_matrices,b10.mlp.Down_bias.double())
  native=[post(z0)];full=[native[0]];compressed=[native[0]]
  for amplitude in (a,b,a+b):
   za=raw9+(att9-(amplitude*w).to(att9.dtype));ha=za+b9.mlp(F.rms_norm(za,(1152,)));raw=b10.lambdas[0]*ha+b10.lambdas[1]*x0;state=raw+b10.attn(F.rms_norm(raw,(1152,)),v1)[0];native.append(post(state))
   zgen=branch(amplitude,context).float().double();full.append(exact_post(zgen,amplitude,exact).float());compressed.append(three_post(zgen,amplitude,three).float())
  outcomes=[]
  for initial in native+full+compressed:
   state=initial
   for block in model.transformer.h[11:]:state,_=block(state,v1,x0)
   logits=(30*torch.tanh(model.lm_head(F.rms_norm(state[:,-1],(1152,)))/30))[0]
   outcomes.append([float(logits[row['uk_id']]-logits[row['us_id']]),float(logits[row['control_ids'][0]]-logits[row['control_ids'][1]])])
  scores=torch.tensor(outcomes,dtype=torch.float64).reshape(3,4,2);effects=scores[:,3]-scores[:,1]-scores[:,2]+scores[:,0]
  cells.append(dict(row=index,family=row['family'],scores=scores.tolist(),interactions=effects.tolist(),max_state_error=max(rel(c,n) for c,n in zip(compressed,native))))
 groups=[]
 for family in range(4):
  sub=cells[family*12:(family+1)*12];effects=torch.tensor([c['interactions'] for c in sub],dtype=torch.float64);n,e,c=effects.unbind(1)
  native_scores=torch.tensor([r['scores'][0][0][0] for r in sub],dtype=torch.float64)
  groups.append(dict(family=family,compressed_relative_errors=[rel(c[:,j],n[:,j]) for j in range(2)],exact_relative_errors=[rel(e[:,j],n[:,j]) for j in range(2)],incremental_relative_errors=[rel(c[:,j],e[:,j]) for j in range(2)],reference_norms=n.norm(dim=0).tolist(),capability=float((native_scores[::2]-native_scores[1::2]).mean()),material_sign_flips=(((n*c)<0)&(n.abs()>=1e-5)).sum(0).tolist()))
 result={'pred_a':A,'pred_b':all(g['compressed_relative_errors'][0]<=.02 and g['compressed_relative_errors'][1]<=.05 for g in groups),'pred_c':all(g['incremental_relative_errors'][0]<=.01 and g['incremental_relative_errors'][1]<=.025 for g in groups),'pred_d':all(g['capability']>0 for g in groups),'anchors':anchors,'groups':groups,'cells':cells,'seconds':time.perf_counter()-tic,'scope':'48 fresh constructed templates, six existing lexical contrasts, live assembled upstream scalar fields; native prefix/background/suffix. Three-term vs exact five-bank and native original interaction. Not independent text-only circuit or corpus OOD.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
if __name__=='__main__':main()
