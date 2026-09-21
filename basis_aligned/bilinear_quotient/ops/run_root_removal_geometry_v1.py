#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_scalar pred_c_geometry
"""Same-root-removal scalar/Jacobian diagnostic; no fit.
Replay previous8errors within1e-4; rawscalarerror>=.10everycell;
weakalpha.25 weightederrorwithin20%actualeachcell.384products322048coeff768indices.
Null: scalarerrornotprimary, orlocalgeometryfails. No semanticadoption.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def summarize(rows):
 result=[]
 for domain,condition,alpha in sorted({(r['domain'],r['condition'],r['alpha']) for r in rows}):
  rr=[r for r in rows if (r['domain'],r['condition'],r['alpha'])==(domain,condition,alpha)];s=lambda k:sum(r[k] for r in rr);ratio=lambda n,d:math.sqrt(s(n)/s(d)) if s(d)>0 else None
  result.append(dict(domain=domain,condition=condition,alpha=alpha,n=s('n'),scalar_error=ratio('scalar_error_energy','scalar_reference_energy'),weighted_scalar_error=ratio('weighted_error_energy','weighted_reference_energy'),actual_error=ratio('actual_error_energy','actual_reference_energy'),baseline_linear_discrepancy=ratio('baseline_linear_error_energy','actual_reference_energy'),local_error_prediction_discrepancy=ratio('local_error_residual_energy','actual_error_energy')))
 return result

def main():
 import torch,tiktoken
 import torch.nn.functional as F
 sys.path.insert(0,str(P));from logit_directional_response import derivative,controls
 from audit_root_feature_conditions import root_features
 from native_quartic_branch import pure_branch
 from paired_root_compiler import cast
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checks=controls();row=dict(domain='toy',condition='newline',alpha=.25,n=3,scalar_error_energy=1.,scalar_reference_energy=100.,weighted_error_energy=1.,weighted_reference_energy=100.,actual_error_energy=1.,actual_reference_energy=100.,baseline_linear_error_energy=0.,local_error_residual_energy=0.)
  assert summarize([row])[0]['scalar_error']==.1
  print(json.dumps(dict(derivative_checks=checks,aggregation='pass')));return
 from native_feature_capture import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'ROOT_REMOVAL_GEOMETRY_V1.json';assert not out.exists()
 path=P/'EXPANDED_ROOT_EMPIRICAL_V1.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()=='f50ab7fe62295fd338e883c486ca1d77d3dbf9d3d21883f8b7b95393bf6fb2df'
 def move(x):
  if torch.is_tensor(x):return x.cuda()
  if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
  if isinstance(x,list):return [move(v) for v in x]
  return x
 program=move(cast(torch.load(path,weights_only=True),torch.float32));tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];enc=tiktoken.get_encoding('gpt2');w=program['writer'][:,1];uw=model.lm_head(w);reader=model.lm_head.weight.T@uw/(uw@uw);unembedding=model.lm_head.weight
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 rows=[];scalar_rows=[];zeros=[]
 for domain,documents in tokens.items():
  for doc,row in enumerate(documents[:16]):
   c=capture(model,row[None,:256].cuda());x,m,h,final=[c[k].flatten(0,1)[16:255] for k in ['x16','m16','h17','final']];l,r,d=[getattr(b17.mlp,k).weight for k in ['Left','Right','Down']]
   exact=pure_branch(m,b16.mlp.Down_bias,b17.lambdas[0],l,r,d)@reader;approx=root_features(program,x)[:,1];den=h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps;direction=-w/den;base=logits(final);j0=derivative(final,direction,unembedding).double();weight=j0.square().sum(-1);zeros.append(float((logits(final+0*exact[:,None]*direction)-base).abs().max()))
   mask=torch.tensor([b'\n' in enc.decode_single_token_bytes(int(t)) for t in row[16:255]],device='cuda');delta=(approx-exact).double()
   for pos in range(len(x)):scalar_rows.append(dict(domain=domain,document=doc,position=pos+16,newline=bool(mask[pos]),native=float(exact[pos]),candidate=float(approx[pos]),sensitivity=float(weight[pos])))
   for alpha in [.25,1.]:
    state=final-alpha*exact[:,None]*w/den;ze=logits(state);za=logits(final-alpha*approx[:,None]*w/den);ref=(ze-base).double();err=(za-ze).double();jlocal=derivative(state,direction,unembedding).double();linear=alpha*exact.double()[:,None]*j0;errlinear=alpha*delta[:,None]*jlocal
    for condition,selected in [('newline',mask),('other',~mask)]:
     n=int(selected.sum())
     if not n:continue
     energy=lambda z:float(z[selected].double().square().sum())
     rows.append(dict(domain=domain,document=doc,condition=condition,alpha=alpha,n=n,scalar_error_energy=energy(delta),scalar_reference_energy=energy(exact),weighted_error_energy=float((delta.square()*weight)[selected].sum()),weighted_reference_energy=float((exact.double().square()*weight)[selected].sum()),actual_error_energy=energy(err),actual_reference_energy=energy(ref),baseline_linear_error_energy=energy(linear-ref),local_error_residual_energy=energy(errlinear-err)))
 summary=summarize(rows);old=json.loads((P/'ROOT_CASE_ABLATION_V1.json').read_text());lookup={(r['domain'],r['condition'],r['alpha']):r['error'] for r in old['summary']};replay=max(abs(r['actual_error']-lookup[r['domain'],r['condition'],r['alpha']]) for r in summary)
 pred=dict(pred_a_replay=replay<1e-4 and max(zeros)==0,pred_b_scalar=all(r['scalar_error']>=.1 for r in summary),pred_c_geometry=all(abs(r['weighted_scalar_error']/r['actual_error']-1)<=.2 for r in summary if r['alpha']==.25))
 result=dict(predictions=pred,summary=summary,rows=rows,scalar_rows=scalar_rows,previous_error_replay=replay,zeroedit=max(zeros),seconds=time.monotonic()-start)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','scalar_rows']},indent=2))
if __name__=='__main__':main()
