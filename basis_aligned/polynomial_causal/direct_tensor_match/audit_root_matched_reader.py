"""Same-token root/native scalar prediction; ROOT_MATCHED_READER_PLAN_V1.md."""
import json,hashlib,time
from pathlib import Path
import torch,tiktoken
from audit_root_feature_conditions import root_features
from paired_root_compiler import cast,evaluate
P=Path(__file__).resolve().parent;BQ=P.parents[1]/'bilinear_quotient'
CK='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'

def matched(tokens,enc):
 flat=tokens.flatten().tolist();eligible=[i for i,t in enumerate(flat) if b'\n' in enc.decode_single_token_bytes(t)];rec=[];don=[]
 for i in eligible:
  candidates=[j for j in eligible if flat[j]==flat[i] and j//64!=i//64]
  if candidates:
   j=min(candidates,key=lambda j:(abs(j%64-i%64),j));rec.append(i);don.append(j)
 return torch.tensor(rec,dtype=torch.long),torch.tensor(don,dtype=torch.long)

def stats(pred,true):
 pred=pred.double();true=true.double();energy=float(true.square().sum());corr=lambda a,b:float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30))
 return dict(error=float((pred-true).norm()/true.norm().clamp_min(1e-30)),cosine=corr(pred,true),correlation=corr(pred-pred.mean(),true-true.mean()),norm_ratio=float(pred.norm()/true.norm().clamp_min(1e-30)),reference_energy=energy)

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();enc=tiktoken.get_encoding('gpt2')
 path=P/'EXPANDED_ROOT_EMPIRICAL_V1.pt';program=cast(torch.load(path,weights_only=True),torch.float64)
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].double()
 a,b,d=weight(16,'Left'),weight(16,'Right'),weight(16,'Down');l,r,e=weight(17,'Left'),weight(17,'Right'),weight(17,'Down');lam=state['transformer.h.17.lambdas'][0].double()
 vocab=state['lm_head.weight'].double();vw=vocab@program['writer'];gram=vw.T@vw;reader=vocab.T@vw[:,1]/gram[1,1];del vocab,vw
 folded=e.T@reader
 def native(x,unfold=False):
  m=((x@a.T)*(x@b.T))@(lam*d).T;roots=(m@l.T)*(m@r.T)
  return (roots@e.T)@reader if unfold else roots@folded
 panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];rows=[];replays=[];native_replays=[]
 for name,panel,cache in zip(['calibration','evaluation'],panels,['fineweb_n96_skip1200.pt','fineweb_n192_skip7000.pt']):
  tokens=torch.load(BQ/'.rowcache'/cache,weights_only=True)[:32,:64];rec,don=matched(tokens,enc)
  assert len(rec)>0 and torch.equal(tokens.flatten()[rec],tokens.flatten()[don]) and bool((rec//64!=don//64).all())
  ids=torch.unique(torch.cat([rec,don]));x=panel['rows'][ids].double();h=root_features(program,x);teacher=native(x);full=evaluate(program,x)@reader
  replays.append(float((full-h[:,1]).norm()/full.norm()));check=native(x,True);native_replays.append(float((teacher-check).norm()/check.norm()))
  pos={int(j):i for i,j in enumerate(ids)};ri=torch.tensor([pos[int(j)] for j in rec]);di=torch.tensor([pos[int(j)] for j in don]);true=teacher[di]-teacher[ri];pred=h[di,1]-h[ri,1];fp=full[di]-full[ri]
  rows.append(dict(panel=name,pairs=len(rec),recipient_prefixes=len(set((rec//64).tolist())),pairs_flat=list(zip(rec.tolist(),don.tolist())),response=stats(pred,true),full_graph_response=stats(fp,true),constant_token_response=stats(torch.zeros_like(true),true),ordinary_values=stats(h[:,1],teacher),pair_rows=[dict(recipient=int(i),donor=int(j),reference=float(t),predicted=float(v)) for i,j,t,v in zip(rec,don,true,pred)]))
 ev=rows[1];predictions=dict(pred_a_integrity=max(replays)<1e-5 and max(native_replays)<1e-10,pred_b_reader=ev['response']['reference_energy']>1e-20 and ev['response']['error']<=.1 and ev['response']['cosine']>=.9 and ev['pairs']>=20 and ev['recipient_prefixes']>=10,pred_c_shape=all(abs(row['response']['correlation'])>=.9 and .8<=row['response']['norm_ratio']<=1.2 for row in rows))
 result=dict(predictions=predictions,rows=rows,graph_reader_replays=replays,native_reader_replays=native_replays,writer_gram_max_offdiag=float((gram-torch.diag(gram.diag())).abs().max()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-start,scope='Frozenfeature1, purequartic native vocabularymetric reader; identical currentnewline-token acrossdifferentprefixes; openedstates, no feature-removal or native-logit causal claim.')
 (P/'ROOT_MATCHED_READER_V1.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
 for row in rows:print(json.dumps({k:v for k,v in row.items() if k not in ['pairs_flat','pair_rows']},indent=2))
if __name__=='__main__':main()
