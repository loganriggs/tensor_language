#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_replay pred_c_sparse_branch
"""Explicit native five-factor attention reader; see ATTENTION_READER_FOLD_V1_PREREGISTRATION.md."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups'
PLAN=dict(prefix=12,double_suffix=4,reverse=36)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch,tiktoken
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_write,guard_torch_save
 sys.path.insert(0,str(P))
 from refined_native_sources import build_refined
 from native_source_observables import source_observables,attention_write64
 from attention_reader_fold import fold
 out=A/'attention_reader_fold_v1_result.json';pt=A/'attention_reader_fold_v1_tensors.pt';assert not out.exists() and not pt.exists()
 torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
 for p in model.parameters():p.requires_grad_(False)
 binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());extra=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text());enc=tiktoken.get_encoding('gpt2')
 modal=torch.tensor([[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]+extra['token_ids'],device='cuda')
 prior=torch.load(A/'residual_reader_transfer_v1_tensors.pt',weights_only=False,map_location='cpu');counts={k:0 for k in PLAN};errors=[];replays=[];records=[];groups=[]
 for context in binding['contexts']:
  c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];n,t,d=c['raw'].shape
  with torch.enable_grad():
   raw=c['raw'].double().detach().requires_grad_();capture={}
   values=source_observables(model,raw,c['x0'],c['first'],torch.zeros(n,1,d,device='cuda',dtype=torch.float64),torch.zeros(n,device='cuda',dtype=torch.long),c['read'],c['pairs'],torch.zeros(n,1,device='cuda',dtype=torch.float64),capture);counts['double_suffix']+=1
   nodes=[raw]+[capture['mlp_inputs'][l] for l in range(11,18)]+[capture['mlp_outputs'][l] for l in range(11,17)]
   gradients=[torch.autograd.grad(values[:,o].sum(),nodes,retain_graph=o<8) for o in range(9)];counts['reverse']+=9
  readers=[torch.stack([g[j] for g in gradients],1).detach() for j in range(len(nodes))]
  for l in range(11,18):
   block=model.transformer.h[l];att=block.attn
   x=raw.detach() if l==11 else block.lambdas[0].double()*capture['mlp_outputs'][l-1].detach()+block.lambdas[1].double()*c['x0'].double()
   q=readers[1+l-11];truth=readers[0] if l==11 else readers[8+l-12]/block.lambdas[0].double()
   cos,sin=att.rotary(torch.zeros(n,t,att.n_head,att.head_dim,device='cuda',dtype=torch.float32))
   parts,y=fold(x,c['first'].double(),[getattr(att,k).weight.double() for k in ['c_q','c_k','c_q2','c_k2','c_v']],att.c_proj.weight.double(),att.lamb.double(),cos.double(),sin.double(),att.n_head,q,torch.finfo(torch.float32).eps)
   errors.append(dict(layer=l,forward=float((y-attention_write64(block,x,c['first'])).abs().max()),backward=float((q+parts.sum(0)-truth).abs().max())))
   if l!=11:continue
   allparts=torch.cat([q[None],parts],0)
   for role in ['subject','attractor']:
    pos,ds=c['source_components'][role];site=allparts[:,:, :, :, :].permute(1,3,0,2,4)[c['batch'],pos]
    g=torch.einsum('bfod,bkd->bfok',site,ds);full=g.sum(1)
    old=next(v for v in prior['groups'] if v['panel']==context['panel'] and v['template']==context['template'] and v['role']==role)
    canonical=site.clone();canonical[:,:,0]*=old['orientation'].to('cuda')[:,None,None]
    replays.append(float((canonical.sum(1)-old['reader'].to('cuda')).abs().max()))
    for j,name in enumerate(['residual','Q1','K1','Q2','K2','V']):
     error=(g[:,j].square().sum((0,2))/full.square().sum((0,2)).clamp_min(1e-24)).sqrt()
     records.append(dict(panel=context['panel'],template=context['template'],role=role,omitted=name,relative_errors=error.tolist()))
    groups.append(dict(panel=context['panel'],template=context['template'],role=role,parts=canonical.cpu()))
 instrument=counts==PLAN and max(max(v['forward'],v['backward']) for v in errors)<=1e-8
 passing=[name for name in ['residual','Q1','K1','Q2','K2','V'] if all(max(r['relative_errors'])<=.1 for r in records if r['omitted']==name)]
 result=dict(plan=PLAN,counts=counts,errors=errors,max_prior_replay=max(replays),records=records,passing_omissions=passing,predictions=dict(pred_a_instrument=instrument,pred_b_replay=instrument and max(replays)<=1e-8,pred_c_sparse_branch=instrument and bool(passing)),seconds=time.perf_counter()-tic)
 guard_torch_save(dict(groups=groups),str(pt));payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=out.name);out.write_text(payload);print(json.dumps({k:result[k] for k in ['predictions','max_prior_replay','passing_omissions','seconds']}))
if __name__=='__main__':main()
