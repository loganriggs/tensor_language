"""Descriptive activation atlas for frozen weight-discovered shared products."""
import json
from pathlib import Path
import torch
import tiktoken
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);source=torch.load(P/'BANK_WIDTH_FRONTIER_V1.pt',weights_only=True);s={k:v.double() for k,v in source['programs'][6].items()};data=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'][0];x=data['rows'].double();tokens=torch.load(P.parent.parent/'bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt',weights_only=True)[32:64,:65];enc=tiktoken.get_encoding('gpt2');values=((x@s['A'].T)*(x@s['B'].T)).reshape(32,64,6);rows=[]
 for k in range(6):
  h=values[:,:,k];mean=h.mean();std=h.std(unbiased=False);between=h.mean(0).var(unbiased=False);total=h.var(unbiased=False);extremes={}
  for name,sign in [('high',1),('low',-1)]:
   order=torch.argsort(sign*h.flatten(),descending=True);seen=set();examples=[]
   for index in order.tolist():
    doc,pos=divmod(index,64)
    if doc in seen:continue
    seen.add(doc);examples.append(dict(document_cache_index=doc+32,position=pos,activation=float(h[doc,pos]),standardized_activation=float((h[doc,pos]-mean)/std),prefix=enc.decode(tokens[doc,max(0,pos-16):pos+1].tolist()),current_token=enc.decode([int(tokens[doc,pos])])) )
    if len(examples)==4:break
   extremes[name]=examples
  rows.append(dict(product=k,left_reader_norm=float(s['A'][k].norm()),right_reader_norm=float(s['B'][k].norm()),bank_writer=s['bank_writer'][:,k].tolist(),empirical_mean=float(mean),empirical_std=float(std),position_mean_variance_fraction=float(between/total),examples=extremes))
 out=dict(features=rows,source='BANK_WIDTH_FRONTIER_V1.pt programs[6]',scope='Descriptive selected-program atlas on32fresh64token documents. Top/bottom examples use distinct documents. Products are not semantic labels; output effect depends on other bank features through the quartic root. Position statistic is descriptive and upward-biased by finite sampling. No feature or fit selected from these examples.');(P/'FROZEN_BANK_FEATURE_ATLAS_V1.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print([{k:f[k] for k in ['product','position_mean_variance_fraction']} for f in rows])
if __name__=='__main__':main()
