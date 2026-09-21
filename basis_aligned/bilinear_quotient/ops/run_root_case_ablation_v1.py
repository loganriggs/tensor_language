#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_fidelity pred_c_selectivity
"""Frozenroot1 nativeprojected quartic removal, strengths.25/1.
Replay<1e-5,zeroedit0;candidate fullresponseerror<=.10 allcells andnewline>=20rows/5prefixes.
FineWebnativecasecontrastchange<=-.01newline,absnewline>=2absnonnewlinebothstrengths.
Null: descriptive root is not selectively causal.384products322048coeff+768indices.
"""
import os,sys,json,time,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def case_pairs(enc):
 vocab={enc.decode_single_token_bytes(i):i for i in range(enc.n_vocab) if i!=enc.eot_token};pairs=[]
 for b,i in vocab.items():
  offset=1 if b.startswith(b' ') else 0
  if len(b)>offset and 97<=b[offset]<=122:
   upper=b[:offset]+bytes([b[offset]-32])+b[offset+1:]
   if upper in vocab:pairs.append((vocab[upper],i))
 return sorted(pairs)

def summarize(rows):
 result=[]
 for domain,alpha,condition in sorted({(r['domain'],r['alpha'],r['condition']) for r in rows}):
  rr=[r for r in rows if (r['domain'],r['alpha'],r['condition'])==(domain,alpha,condition)];n=sum(r['n'] for r in rr);ref=sum(r['reference_energy'] for r in rr)
  result.append(dict(domain=domain,alpha=alpha,condition=condition,n=n,prefixes=len(rr),error=math.sqrt(sum(r['error_energy'] for r in rr)/ref) if ref else None,reference_energy=ref,native_case_mean=sum(r['native_case_sum'] for r in rr)/n,native_case_mean_abs=sum(r['native_case_abs_sum'] for r in rr)/n,candidate_case_mean=sum(r['candidate_case_sum'] for r in rr)/n))
 return result

def main():
 import torch,tiktoken
 import torch.nn.functional as F
 sys.path.insert(0,str(P));from audit_root_feature_conditions import root_features
 from paired_root_compiler import cast
 from native_quartic_branch import bilinear,pure_branch
 torch.set_num_threads(2);torch.set_grad_enabled(False);enc=tiktoken.get_encoding('gpt2');pairs=case_pairs(enc)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  assert len(pairs)>100 and len(set(pairs))==len(pairs)
  for u,l in pairs:assert enc.decode_single_token_bytes(u).lower()==enc.decode_single_token_bytes(l).lower()
  torch.manual_seed(17);u=torch.randn(11,5,dtype=torch.float64);w=torch.randn(5,dtype=torch.float64);v=u@w;r=u.T@v/(v@v);assert abs(float(r@w)-1)<1e-12
  x=torch.randn(7,5,dtype=torch.float64);h=torch.randn_like(x);t=x@r;den=h.square().mean(-1,keepdim=True)+torch.finfo(x.dtype).eps
  assert torch.allclose((x-t[:,None]*w/den)@r,x@r-t/den[:,0])
  row=dict(domain='toy',alpha=.25,condition='newline',n=2,reference_energy=4.,error_energy=.04,native_case_sum=-.2,native_case_abs_sum=.2,candidate_case_sum=-.19)
  assert abs(summarize([row])[0]['error']-.1)<1e-12
  print(json.dumps(dict(case_pairs=len(pairs),projection_smoke='pass',aggregation_smoke='pass')));return
 from native_feature_capture import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'ROOT_CASE_ABLATION_V1.json';assert not out.exists();start=time.monotonic()
 path=P/'EXPANDED_ROOT_EMPIRICAL_V1.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()=='f50ab7fe62295fd338e883c486ca1d77d3dbf9d3d21883f8b7b95393bf6fb2df'
 program=cast(torch.load(path,weights_only=True),torch.float32)
 def move(x):
  if torch.is_tensor(x):return x.cuda()
  if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
  if isinstance(x,list):return [move(v) for v in x]
  return x
 program=move(program);tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17]
 w=program['writer'][:,1];uw=model.lm_head(w);reader=model.lm_head.weight.T@uw/(uw@uw);reader_replay=abs(float(reader@w)-1)
 upper=torch.tensor([u for u,l in pairs],device='cuda');lower=torch.tensor([l for u,l in pairs],device='cuda')
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 def contrast(z):return torch.logsumexp(z[:,upper],1)-torch.logsumexp(z[:,lower],1)
 rows=[];checks=[];zeros=[]
 for domain,documents in tokens.items():
  for doc,row in enumerate(documents[:16]):
   c=capture(model,row[None,:256].cuda());x,m,h,final=[c[k].flatten(0,1)[16:255] for k in ['x16','m16','h17','final']]
   l,r,d=[getattr(b17.mlp,k).weight for k in ['Left','Right','Down']];native=pure_branch(m,b16.mlp.Down_bias,b17.lambdas[0],l,r,d)
   independent=bilinear(b17.lambdas[0]*bilinear(x,b16.mlp.Left.weight,b16.mlp.Right.weight,b16.mlp.Down.weight),l,r,d)
   checks.append(float((native-independent).norm()/native.norm()));exact=native@reader;approx=root_features(program,x)[:,1];den=h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps
   base=logits(final);basecontrast=contrast(base);zeros.append(float((logits(final-0*exact[:,None]*w/den)-base).abs().max()))
   mask=torch.tensor([b'\n' in enc.decode_single_token_bytes(int(t)) for t in row[16:255]],device='cuda')
   for alpha in [.25,1.]:
    ze=logits(final-alpha*exact[:,None]*w/den);za=logits(final-alpha*approx[:,None]*w/den);ref=(ze-base).double();error=(za-ze).double();ec=(contrast(ze)-basecontrast).double();ac=(contrast(za)-basecontrast).double()
    for condition,selected in [('newline',mask),('other',~mask)]:
     n=int(selected.sum())
     if n:rows.append(dict(domain=domain,document=doc,alpha=alpha,condition=condition,n=n,reference_energy=float(ref[selected].square().sum()),error_energy=float(error[selected].square().sum()),native_case_sum=float(ec[selected].sum()),native_case_abs_sum=float(ec[selected].abs().sum()),candidate_case_sum=float(ac[selected].sum())))
 summary=summarize(rows);lookup={(r['domain'],r['alpha'],r['condition']):r for r in summary};fw=[r for r in summary if r['domain']=='fineweb' and r['condition']=='newline'];newlines=[r for r in summary if r['condition']=='newline']
 pred=dict(pred_a_instrument=max(checks)<1e-5 and reader_replay<1e-5 and max(zeros)==0,pred_b_fidelity=all(r['error'] is not None and r['error']<=.1 for r in summary) and len(newlines)==4 and all(r['n']>=20 and r['prefixes']>=5 for r in newlines),pred_c_selectivity=len(fw)==2 and all(r['native_case_mean']<=-.01 and r['native_case_mean_abs']>=2*lookup[r['domain'],r['alpha'],'other']['native_case_mean_abs'] for r in fw))
 result=dict(predictions=pred,summary=summary,rows=rows,case_pairs=pairs,reader_replay=reader_replay,maximum_native_replay=max(checks),maximum_zero_edit=max(zeros),seconds=time.monotonic()-start,scope='Output-projected purequartic removal; fixeddenominator/otherbranches; originalbaseline; actualfinalnormalizationandsoftcap; openedpanels,noadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','case_pairs']},indent=2))
if __name__=='__main__':main()
