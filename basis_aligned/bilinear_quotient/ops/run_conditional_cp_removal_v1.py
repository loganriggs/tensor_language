#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_absolute pred_c_retention
"""Conditional rank256 root1 finite removal, two seeds, original32prefixes.
Replay<1e-5/zeroedit0. All16seed/domain/strength/conditioncells<=10%error.
Relativecontrol: allcells<=1.10frozenCP512parenterror.3584products850960floats.
Native case selectivity already failed; this is fidelity, not semantic promotion.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def cp_values(program,x):
 from conditional_quartic_cp import evaluate
 return evaluate(program,x)

def main():
 import torch,tiktoken
 import torch.nn.functional as F
 sys.path.insert(0,str(P));sys.path.insert(0,str(ROOT/'basis_aligned/bilinear_quotient/ops'))
 from run_root_case_ablation_v1 import case_pairs,summarize
 from native_quartic_branch import bilinear,pure_branch
 torch.set_num_threads(2);torch.set_grad_enabled(False);enc=tiktoken.get_encoding('gpt2');pairs=case_pairs(enc)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  torch.manual_seed(11700);x=torch.randn(9,12,dtype=torch.float64);program=dict(factors=[torch.randn(512,12,dtype=x.dtype) for _ in range(4)],coefficients=torch.randn(16,512,dtype=x.dtype))
  from conditional_quartic_cp import construct
  program=construct(program['factors'],program['coefficients'],torch.eye(12,dtype=x.dtype),torch.zeros(12,dtype=x.dtype),torch.eye(12,dtype=x.dtype))
  ref=torch.stack([torch.stack([(x[i]@a.T) for a in program['factors']]).prod(0)@program['coefficients'].T for i in range(len(x))]);assert torch.allclose(cp_values(program,x),ref)
  example=dict(domain='toy',alpha=.25,condition='newline',n=2,reference_energy=4.,error_energy=.04,native_case_sum=-.2,native_case_abs_sum=.2,candidate_case_sum=-.19)
  summary=[dict(seed=seed,**r) for seed in [1001,1002] for r in summarize([example])];assert len(summary)==2 and all(abs(r['error']-.1)<1e-12 for r in summary)
  actual=dict(input_projection=torch.randn(256,1152),factors=[torch.randn(512,256) for _ in range(4)],biases=[torch.randn(512) for _ in range(4)],pair_weights=torch.randn(6,512),coefficients=torch.randn(16,512),constant=torch.randn(16))
  assert cp_values(actual,torch.randn(9,1152)).shape==(9,16)
  assert len(pairs)==8805;print(json.dumps(dict(full512values=True,seed_reporting=True,case_pairs=len(pairs))));return
 from native_feature_capture import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'CONDITIONAL_CP_REMOVAL_V1.json';assert not out.exists();start=time.monotonic()
 frozen=json.loads((P/'CONDITIONAL_CP_REMOVAL_INPUTS_V1.json').read_text())
 for file,sha in frozen.items():assert hashlib.sha256((P/file).read_bytes()).hexdigest()==sha
 programs={};hashes={}
 for seed in [1001,1002]:
  path=P/f'CONDITIONAL_CP_SEED{seed}_RANK256_V1.pt';hashes[seed]=hashlib.sha256(path.read_bytes()).hexdigest();p=torch.load(path,weights_only=True)
  programs[seed]={k:([a.cuda().float() for a in v] if isinstance(v,list) else v.cuda().float() if isinstance(v,torch.Tensor) else v) for k,v in p.items()}
 assert torch.equal(programs[1001]['writer'],programs[1002]['writer'])
 oldwriter=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'];assert torch.equal(programs[1001]['writer'].cpu(),oldwriter.float())
 tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17]
 w=programs[1001]['writer'][:,1];uw=model.lm_head(w);reader=model.lm_head.weight.T@uw/(uw@uw);reader_replay=abs(float(reader@w)-1)
 upper=torch.tensor([u for u,l in pairs],device='cuda');lower=torch.tensor([l for u,l in pairs],device='cuda')
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 def contrast(z):return torch.logsumexp(z[:,upper],1)-torch.logsumexp(z[:,lower],1)
 rows=[];checks=[];zeros=[]
 for domain,documents in tokens.items():
  for doc,row in enumerate(documents[:16]):
   capture_data=capture(model,row[None,:256].cuda());x,m,h,final=[capture_data[k].flatten(0,1)[16:255] for k in ['x16','m16','h17','final']]
   l,r,d=[getattr(b17.mlp,k).weight for k in ['Left','Right','Down']];native=pure_branch(m,b16.mlp.Down_bias,b17.lambdas[0],l,r,d)
   independent=bilinear(b17.lambdas[0]*bilinear(x,b16.mlp.Left.weight,b16.mlp.Right.weight,b16.mlp.Down.weight),l,r,d);checks.append(float((native-independent).norm()/native.norm()))
   exact=native@reader;approx={seed:cp_values(p,x)[:,1] for seed,p in programs.items()};den=h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps
   base=logits(final);basecontrast=contrast(base);zeros.append(float((logits(final-0*exact[:,None]*w/den)-base).abs().max()))
   mask=torch.tensor([b'\n' in enc.decode_single_token_bytes(int(t)) for t in row[16:255]],device='cuda')
   for alpha in [.25,1.]:
    ze=logits(final-alpha*exact[:,None]*w/den);ref=(ze-base).double();ec=(contrast(ze)-basecontrast).double()
    for seed in programs:
     za=logits(final-alpha*approx[seed][:,None]*w/den);error=(za-ze).double();ac=(contrast(za)-basecontrast).double()
     for condition,selected in [('newline',mask),('other',~mask)]:
      n=int(selected.sum())
      if n:rows.append(dict(seed=seed,domain=domain,document=doc,alpha=alpha,condition=condition,n=n,reference_energy=float(ref[selected].square().sum()),error_energy=float(error[selected].square().sum()),native_case_sum=float(ec[selected].sum()),native_case_abs_sum=float(ec[selected].abs().sum()),candidate_case_sum=float(ac[selected].sum())))
 summary=[dict(seed=seed,**r) for seed in programs for r in summarize([r for r in rows if r['seed']==seed])]
 baseline=json.loads((P/'MIXED_CP_REMOVAL_V1.json').read_text());old={(r['seed'],r['domain'],r['alpha'],r['condition']):r for r in baseline['summary']}
 native_replay=max(abs(r['reference_energy']-old[r['seed'],r['domain'],r['alpha'],r['condition']]['reference_energy'])/old[r['seed'],r['domain'],r['alpha'],r['condition']]['reference_energy'] for r in summary)
 pred=dict(pred_a_instrument=max(checks)<1e-5 and reader_replay<1e-5 and max(zeros)==0 and native_replay<1e-4,pred_b_absolute=len(summary)==16 and all(r['error'] is not None and r['error']<=.1 for r in summary),pred_c_retention=all(r['error']<=1.10*old[r['seed'],r['domain'],r['alpha'],r['condition']]['error'] for r in summary))
 result=dict(predictions=pred,summary=summary,rows=rows,program_sha256=hashes,reader_replay=reader_replay,maximum_native_replay=max(checks),maximum_zero_edit=max(zeros),prior_reference_energy_replay=native_replay,seconds=time.monotonic()-start,scope='Frozen two CP candidates; native root1 projected quartic removal at quarter/full strength, actual final RMSNorm/softcap, same original denominator/otherbranches. OpenedFineWeb/code panels. Selectivity previouslyfailed andnotreset. No OOD or semanticadoption claim. Eachcandidate3584products850960storedfloatsincludingcommonwriter; implicitGaussianconditionalmeanrank256 withquadratic/constantcorrections. Compareparent512CPcost1536products2385920floats. Openedpanelretention, notfreshvalidation.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':main()
