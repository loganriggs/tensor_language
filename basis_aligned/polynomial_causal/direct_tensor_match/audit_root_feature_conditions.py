"""Descriptive conditions and output effects of the frozen16-form parent graph.
Not a semantic label assignment, causal identification or new validation panel.
"""
import json,hashlib
from pathlib import Path
import torch,tiktoken
from paired_root_compiler import evaluate,cast
from quadratic_pair_blocks import products
P=Path(__file__).resolve().parent;BQ=P.parents[1]/'bilinear_quotient'

def root_features(program,x):
 u,v=program['U'],program['V'];m,k,d=u.shape
 q=((x@u.flatten(0,1).T)*(x@v.flatten(0,1).T)).reshape(len(x),m,k).sum(-1)
 output=[]
 for pair in program['pairs']:
  if pair['kind']=='pair':output.append(products(q@pair['input_transform'],pair['product_indices'])@pair['product_weights'])
  else:output.append((torch.einsum('ni,gij->ngj',q,pair['vectors']).square()*pair['values']).sum(-1))
 return torch.cat(output,1)

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 path=P/'EXPANDED_ROOT_EMPIRICAL_V1.pt';program=cast(torch.load(path,weights_only=True),torch.float64)
 old=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)['rows'];x=torch.cat([old,extra]).double();h=root_features(program,x)
 replay=float((h@program['writer'].T-evaluate(program,x)).norm()/evaluate(program,x).norm());assert replay<1e-12
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 directions=state['lm_head.weight'].double()@program['writer'];tokens=torch.load(BQ/'.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:96,:65];enc=tiktoken.get_encoding('gpt2');rows=[]
 for g in range(16):
  mean=h[:,g].mean();std=h[:,g].std(unbiased=False);z=(h[:,g]-mean)/std
  def examples(sign):
   order=torch.argsort(sign*z,descending=True);seen=set();result=[]
   for flat in order.tolist():
    prefix,position=divmod(flat,64)
    if prefix in seen:continue
    seen.add(prefix);result.append(dict(prefix=prefix,position=position,z=float(z[flat]),context=enc.decode(tokens[prefix,max(0,position-23):position+1].tolist()),next_token=enc.decode([int(tokens[prefix,position+1])])) )
    if len(result)==3:break
   return result
  def vocab(sign):
   ids=torch.topk(sign*directions[:,g],10).indices.tolist()
   return [dict(id=i,text=enc.decode([i]),weight=float(directions[i,g])) for i in ids]
  rows.append(dict(feature=g,mean=float(mean),standard_deviation=float(std),positive_writer_tokens=vocab(1),negative_writer_tokens=vocab(-1),high_conditions=examples(1),low_conditions=examples(-1)))
 result=dict(program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),root_readout_replay=replay,features=rows,scope='Fixed SVD-gauge output-shared quartic features on96historical calibrationprefixes. Writer effects before nonlinear finalRMS/softcap; top contexts are descriptive and selected, not semantic/causal evidence. A sum may represent multiple conditions; no monosemantic labels assigned.')
 (P/'ROOT_FEATURE_CONDITIONS_V1.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 for row in rows:
  print(json.dumps(dict(feature=row['feature'],positive=[r['text'] for r in row['positive_writer_tokens'][:5]],negative=[r['text'] for r in row['negative_writer_tokens'][:5]],high=[r['context'] for r in row['high_conditions'][:2]],low=[r['context'] for r in row['low_conditions'][:2]]),ensure_ascii=False))
if __name__=='__main__':main()
