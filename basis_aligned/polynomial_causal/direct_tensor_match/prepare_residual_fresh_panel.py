"""Freeze256 distinct new-to-this-evaluation documents; no model calls."""
import hashlib,itertools,json,time
from pathlib import Path
import torch,tiktoken
from datasets import load_dataset
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);start=time.monotonic();out=P/'RESIDUAL_FRESH_TOKENS_V1.pt';assert not out.exists()
 known=set();seen=set();sources=[]
 for path in sorted((P.parents[1]/'bilinear_quotient/.rowcache').glob('fineweb_n*_skip*.pt'))+sorted(P.glob('*TOKENS_V*.pt')):
  a=torch.load(path,weights_only=True,map_location='cpu')
  rows=a.get('fineweb') if isinstance(a,dict) else a
  if isinstance(rows,torch.Tensor) and rows.ndim==2 and rows.shape[1]>=32:
   known.update(tuple(r[:32].tolist()) for r in rows);sources.append(path.name)
 for path in sorted(P.glob('*ROWS_V*.json')):
  try:a=json.loads(path.read_text())
  except json.JSONDecodeError:continue
  if isinstance(a,dict):
   for row in a.get('fineweb_documents',[]):
    seen.update(str(row[k]) for k in ['document_id','text_sha256'] if k in row)
 byfirst={}
 for row in known:byfirst.setdefault(row[0],set()).add(row)
 enc=tiktoken.get_encoding('gpt2');chosen=[];meta=[];reject={'short':0,'known_excerpt':0,'duplicate':0}
 ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',split='train',streaming=True);it=iter(ds)
 for idx,doc in enumerate(itertools.islice(it,20000)):
  text=doc['text'];sha=hashlib.sha256(text.encode()).hexdigest();identity=str(doc.get('id') or sha)
  if identity in seen or sha in seen:reject['duplicate']+=1;continue
  ids=enc.encode_ordinary(text)
  if len(ids)<65:reject['short']+=1;continue
  if any(ids[i] in byfirst and tuple(ids[i:i+32]) in byfirst[ids[i]] for i in range(len(ids)-31)):
   reject['known_excerpt']+=1;continue
  row=torch.tensor(ids[:65],dtype=torch.long);chosen.append(row);seen.update([identity,sha]);first=tuple(ids[:32]);byfirst.setdefault(first[0],set()).add(first)
  meta.append(dict(stream_index=idx,document_id=identity,text_sha256=sha,prefix_sha256=hashlib.sha256(row.numpy().tobytes()).hexdigest()))
  if len(chosen)%32==0:print('accepted',len(chosen),'stream_index',idx,flush=True)
  if len(chosen)==256:break
 assert len(chosen)==256
 if hasattr(it,'close'):it.close()
 tokens=torch.stack(chosen);torch.save(tokens,out)
 info=dict(fineweb_documents=meta,excluded_sources=sources,known_prefixes=len(known),rejections=reject,tokens_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),token_content_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),seconds=time.monotonic()-start,scope='256 source documents, one65tokenprefix each, first64positions scored. Excludes all enumerated local FineWeb cache prefixes and direct_tensor_match prior document metadata. Not a pretraining or repository-wide independence guarantee. No fitting.')
 (P/'RESIDUAL_FRESH_ROWS_V1.json').write_text(json.dumps(info,indent=2)+'\n');print('complete seconds',info['seconds'],flush=True)
if __name__=='__main__':main()
