"""Frozen feature condition screen; plan ROOT_CONDITION_TRANSFER_PLAN_V1.md."""
import codecs,json,hashlib
from pathlib import Path
import numpy as np
import torch,tiktoken
from audit_root_feature_conditions import root_features
from paired_root_compiler import evaluate,cast
P=Path(__file__).resolve().parent;BQ=P.parents[1]/'bilinear_quotient'

def auc(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s);n=int(y.sum());m=len(y)-n
 if not n or not m:return None
 _,inv,counts=np.unique(s,return_inverse=True,return_counts=True)
 ranks=np.cumsum(counts)-.5*(counts-1)
 return float((ranks[inv][y].sum()-n*(n+1)/2)/(n*m))

def labels(tokens,enc):
 result={'newline':[],'utf8_pending':[],'invalid_prefix':[]}
 for row in tokens.tolist():
  decoder=codecs.getincrementaldecoder('utf-8')('strict');invalid=False
  for token in row:
   b=enc.decode_single_token_bytes(token)
   try:decoder.decode(b,final=False)
   except UnicodeDecodeError:invalid=True;decoder.reset()
   result['newline'].append(b'\n' in b)
   result['utf8_pending'].append(bool(decoder.getstate()[0]) and not invalid)
   result['invalid_prefix'].append(invalid)
 return {k:np.asarray(v) for k,v in result.items()}

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 assert auc([0,1],[0,1])==1 and auc([0,1],[1,0])==0 and auc([0,1],[1,1])==.5
 enc=tiktoken.get_encoding('gpt2');program_path=P/'EXPANDED_ROOT_EMPIRICAL_V1.pt';program=cast(torch.load(program_path,weights_only=True),torch.float64)
 panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)['rows']
 xs=[torch.cat([panels[0]['rows'],extra]).double(),panels[1]['rows'].double()]
 ts=[torch.load(BQ/'.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:96,:64],torch.load(BQ/'.rowcache/fineweb_n192_skip7000.pt',weights_only=True)[:32,:64]]
 hs=[root_features(program,x) for x in xs];replays=[float((h@program['writer'].T-evaluate(program,x)).norm()/evaluate(program,x).norm()) for h,x in zip(hs,xs)]
 ys=[labels(t,enc) for t in ts];features=[h.numpy() for h in hs];rows=[]
 for name in ('newline','utf8_pending'):
  valid=[~y['invalid_prefix'] if name=='utf8_pending' else np.ones(len(y[name]),dtype=bool) for y in ys]
  train=[auc(ys[0][name][valid[0]],features[0][valid[0],g]) for g in range(16)]
  selected=max(range(16),key=lambda g:abs(train[g]-.5));sign=1 if train[selected]>=.5 else -1
  evals=[auc(ys[1][name][valid[1]],features[1][valid[1],g]) for g in range(16)]
  tok0=ts[0].flatten().numpy();tok1=ts[1].flatten().numpy();h0=features[0][:,selected];h1=features[1][:,selected]
  counts=np.bincount(tok0,minlength=50304);sums=np.bincount(tok0,weights=h0,minlength=50304)
  means=np.full(50304,h0.mean());np.divide(sums,counts,out=means,where=counts>0)
  pred=means[tok1];r2=float(1-((pred-h1)**2).sum()/((h1-h1.mean())**2).sum())
  rows.append(dict(condition=name,selected_feature=selected,sign=sign,calibration_auc=train[selected] if sign==1 else 1-train[selected],evaluation_auc=evals[selected] if sign==1 else 1-evals[selected],all_calibration_unsigned_auc=train,all_evaluation_unsigned_auc=evals,positive_counts=[int(y[name][v].sum()) for y,v in zip(ys,valid)],negative_counts=[int((~y[name][v]).sum()) for y,v in zip(ys,valid)],invalid_prefix_counts=[int(y['invalid_prefix'].sum()) for y in ys],token_lookup_evaluation_r2=r2,token_lookup_seen_fraction=float((counts[tok1]>0).mean())))
 support={}
 for panel_name,y,t in zip(('calibration','evaluation'),ys,ts):
  support[panel_name]={}
  for condition in ('newline','utf8_pending'):
   counts=y[condition].reshape(len(t),64).sum(1)
   support[panel_name][condition]=dict(positive_prefixes=int((counts>0).sum()),counts_by_prefix=counts.tolist(),max_prefix_share=float(counts.max()/counts.sum()))
 (P/'ROOT_CONDITION_SUPPORT_V1.json').write_text(json.dumps(support,indent=2)+'\n')
 result=dict(program_sha256=hashlib.sha256(program_path.read_bytes()).hexdigest(),replays=replays,conditions=rows,predictions=dict(pred_a_replay=max(replays)<1e-12,pred_b_transfer=all(r['evaluation_auc']>=.9 and min(r['positive_counts']+r['negative_counts'])>=20 for r in rows),pred_c_not_token_lookup=any(r['token_lookup_evaluation_r2']<.5 for r in rows)),scope='Post-inventory descriptive hypotheses, calibration-selected feature/sign. Second panel already opened in earlier research. No causal or semantic identification.')
 (P/'ROOT_CONDITION_TRANSFER_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
