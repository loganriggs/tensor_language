#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selectivity pred_c_writer
"""Frozen group2 continuation hypothesis on32unused FineWeb rows112:144.
pred_a_instrument hashes, finite effects, >=32continuation/space sites.
pred_b_selectivity real removal CEadded>.05continuation and abs(space CE)<.02.
pred_c_writer real decrease in bare-vs-spaced-word logodds exceeds matched
sham's decrease by>.02nats on continuation sites. Sham uses same scalar with
component3 writer, norm-matched in fixed vocabulary-centered geometry.
Secondary prespecified: within same-current-token strata, continuation-minus-
space mean real logodds decrease>.01, requiring>=16sites per class.
No fitting or regrouping, original native background, final normalization explicit.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN_NAME='MIDPOINT_CONTINUATION_GROUP_PLAN_V1.json'
TOKEN_NAME='MIDPOINT_CONTINUATION_GROUP_TOKENS_V1.pt'
OUTPUT_NAME='MIDPOINT_CONTINUATION_GROUP_NATIVE_V1.json'
FORWARDS=32
SCOPE='Frozen hypothesis from reused discovery panel, new FineWeb rows for confirmation.'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=FORWARDS,arms=['real','sham'],context=256,fit=False)));return
 import torch
 import torch.nn.functional as F
 import tiktoken
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 from token_boundary_conditions import annotate,vocabulary_masks
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/OUTPUT_NAME;assert not out.exists();start=time.perf_counter();enc=tiktoken.get_encoding('gpt2')
 plan=json.loads((P/PLAN_NAME).read_text());artifact=P/'MIDPOINT_STABLE_GROUP_REMOVAL_PROGRAMS_V1.pt';assert hashlib.sha256(artifact.read_bytes()).hexdigest()==plan['program_sha256']
 tokens=torch.load(P/TOKEN_NAME,weights_only=True);assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==plan['token_sha256']
 e={k:v.cuda().double() for k,v in torch.load(artifact,weights_only=True)['half0'].items()};S=torch.load(P/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].cuda().double()
 writers=dict(real=e['W'][:,2],sham=e['W'][:,3]*(S@e['W'][:,2]).norm()/(S@e['W'][:,3]).norm())
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');invru=torch.linalg.inv(ru)
 bare,spaced=vocabulary_masks(enc,50304);bare=torch.tensor(bare,device='cuda');spaced=torch.tensor(spaced,device='cuda')
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 odds=lambda z:torch.logsumexp(z[:,bare],1)-torch.logsumexp(z[:,spaced],1)
 records=[]
 for doc,row in enumerate(tokens):
  labels=annotate(row.tolist(),enc);c=capture(model,row[None,:256].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=((h-m/2)/scale)[16:];m=(m/scale)[16:];state=c['final'].flatten(0,1)[16:];base=logits(state);target=row[17:257].cuda();ce=F.cross_entropy(base,target,reduction='none');baseodds=odds(base)
  scalar=(n@e['A'][:,2]-e['left_mean'][2])*(m@e['B'][:,2]-e['right_mean'][2])
  for arm,w in writers.items():
   changed=logits(state-(scalar[:,None]*(invru@w)[None,:]).float());damage=(F.cross_entropy(changed,target,reduction='none')-ce).cpu().tolist();decrease=(baseodds-odds(changed)).cpu().tolist();values=scalar.cpu().tolist()
   for j in range(240):
    pos=j+16;records.append(dict(document=int(plan['documents'][doc]),position=pos,arm=arm,input_token=int(row[pos]),**labels[pos],scalar=values[j],ce_added=damage[j],continuation_logodds_decrease=decrease[j]))
 summary={}
 for arm in writers:
  summary[arm]={}
  for label in ['continuation','spaced_word','utf8_pending']:
   rr=[v for v in records if v['arm']==arm and v[label]]
   summary[arm][label]=dict(sites=len(rr),ce_added=sum(v['ce_added'] for v in rr)/len(rr) if rr else None,mean_logodds_decrease=sum(v['continuation_logodds_decrease'] for v in rr)/len(rr) if rr else None)
 rr=[v for v in records if v['arm']=='real'];current=set(v['input_token'] for v in rr if v['continuation'])&set(v['input_token'] for v in rr if v['spaced_word']);matched=[]
 for token in sorted(current):
  a=[v['continuation_logodds_decrease'] for v in rr if v['input_token']==token and v['continuation']];b=[v['continuation_logodds_decrease'] for v in rr if v['input_token']==token and v['spaced_word']];matched.append(dict(input_token=token,continuation_count=len(a),spaced_count=len(b),difference=sum(a)/len(a)-sum(b)/len(b),weight=min(len(a),len(b))))
 total=sum(v['weight'] for v in matched);contrast=sum(v['weight']*v['difference'] for v in matched)/total if total else None
 secondary=dict(matched_token_types=len(matched),matched_continuation_sites=sum(v['continuation_count'] for v in matched),matched_spaced_sites=sum(v['spaced_count'] for v in matched),min_count_weighted_logodds_contrast=contrast,strata=matched)
 secondary['coverage_sufficient']=min(secondary['matched_continuation_sites'],secondary['matched_spaced_sites'])>=16
 secondary['beyond_token_gate']=contrast>.01 if secondary['coverage_sufficient'] else None
 real=summary['real'];sham=summary['sham'];pred=dict(pred_a_instrument=min(real['continuation']['sites'],real['spaced_word']['sites'])>=32 and all(torch.isfinite(torch.tensor(v['ce_added'])) for v in records),pred_b_selectivity=real['continuation']['ce_added']>.05 and abs(real['spaced_word']['ce_added'])<.02,pred_c_writer=real['continuation']['mean_logodds_decrease']-sham['continuation']['mean_logodds_decrease']>.02)
 result=dict(predictions=pred,summary=summary,same_token_secondary=secondary,records=records,program_sha256=plan['program_sha256'],token_sha256=plan['token_sha256'],seconds=time.perf_counter()-start,scope=SCOPE+' Continuation/space classes use realized next token and prefix eligibility; selective average effects, not randomized manipulation of an independent continuation variable. Same-token contrast controls current identity, not all context/difficulty confounds. No fit or semantic monosemanticity claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
