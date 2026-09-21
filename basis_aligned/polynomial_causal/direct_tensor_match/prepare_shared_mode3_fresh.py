"""Freeze new distinct-document FW panel and existing code controls.
Exclude a candidate document if ANY token span matches a cached FW chunk's
first32 tokens. This is conservative relative to exact full-prefix exclusion.
One257tokenprefix per accepted document; preserve source identity.
"""
from pathlib import Path
import json,hashlib,itertools
import torch,tiktoken
from datasets import load_dataset
P=Path(__file__).resolve().parent;torch.set_num_threads(2)
cache=P.parents[1]/'bilinear_quotient/.rowcache'
known=set()
for path in sorted(cache.glob('fineweb_n*_skip*.pt')):
 rows=torch.load(path,weights_only=True)
 for row in rows:known.add(tuple(row[:32].tolist()))
byfirst={}
for prefix in known:byfirst.setdefault(prefix[0],set()).add(prefix)
enc=tiktoken.get_encoding('gpt2');chosen=[];metadata=[];seen=set();rejections=dict(short=0,known_excerpt=0,duplicate_document=0)
dataset=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',split='train',streaming=True)
for stream_index,doc in enumerate(itertools.islice(dataset,20000)):
 text=doc['text'];text_hash=hashlib.sha256(text.encode()).hexdigest();identity=str(doc.get('id') or text_hash)
 if identity in seen or text_hash in seen:rejections['duplicate_document']+=1;continue
 ids=enc.encode_ordinary(text)
 if len(ids)<257:rejections['short']+=1;continue
 if any(ids[i] in byfirst and tuple(ids[i:i+32]) in byfirst[ids[i]] for i in range(len(ids)-31)):
  rejections['known_excerpt']+=1;continue
 seen.update([identity,text_hash]);chosen.append(torch.tensor(ids[:257],dtype=torch.long))
 metadata.append(dict(stream_index=stream_index,document_id=identity,text_sha256=text_hash,prefix_sha256=hashlib.sha256(chosen[-1].numpy().tobytes()).hexdigest()))
 if len(chosen)%8==0:print('accepted',len(chosen),'stream position',stream_index,flush=True)
 if len(chosen)==32:break
assert len(chosen)==32
code=torch.load(P/'MIDPOINT_STDLIB_CONTINUATION_TOKENS_V1.pt',weights_only=True)
tokens={'fineweb':torch.stack(chosen),'stdlib':code}
torch.save(tokens,P/'SHARED_MODE3_FRESH_TOKENS_V1.pt')
out=dict(fineweb_documents=metadata,known_cached_prefixes=len(known),rejections=rejections,source='HuggingFaceFW/fineweb sample-10BT train stream',scope='New document-indexed panel for this extraction study, one prefix perdocument and no knowncachedFineWeb excerpt anywhere in eachaccepted document. Not a pretraining-overlap or whole-project novelty guarantee. Stdlib16 reused.')
(P/'SHARED_MODE3_FRESH_ROWS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k!='fineweb_documents'})
