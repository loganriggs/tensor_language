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
previous=[]
for version in ['FRESH','SECOND']:
 for row in torch.load(P/f'SHARED_MODE3_{version}_TOKENS_V1.pt',weights_only=True)['fineweb']:known.add(tuple(row[:32].tolist()))
 previous+=json.loads((P/f'SHARED_MODE3_{version}_ROWS_V1.json').read_text())['fineweb_documents']
for row in torch.load(P/'PARTIAL_GRAPH_FRESH_TOKENS_V1.pt',weights_only=True)['fineweb']:known.add(tuple(row[:32].tolist()))
previous+=json.loads((P/'PARTIAL_GRAPH_FRESH_ROWS_V1.json').read_text())['fineweb_documents']
for row in torch.load(P/'PARTIAL_GRAPH_FRESH_TOKENS_V2.pt',weights_only=True)['fineweb']:known.add(tuple(row[:32].tolist()))
previous+=json.loads((P/'PARTIAL_GRAPH_FRESH_ROWS_V2.json').read_text())['fineweb_documents']
for row in torch.load(P/'COMPACT_GROUP_FRESH_TOKENS_V1.pt',weights_only=True)['fineweb']:known.add(tuple(row[:32].tolist()))
previous+=json.loads((P/'COMPACT_GROUP_FRESH_ROWS_V1.json').read_text())['fineweb_documents']
for row in torch.load(P/'DUAL_FRESH_TOKENS_V1.pt',weights_only=True)['fineweb']:known.add(tuple(row[:32].tolist()))
previous+=json.loads((P/'DUAL_FRESH_ROWS_V1.json').read_text())['fineweb_documents']
for row in torch.load(P/'FRONTIER_FRESH_TOKENS_V1.pt',weights_only=True)['fineweb']:known.add(tuple(row[:32].tolist()))
previous+=json.loads((P/'FRONTIER_FRESH_ROWS_V1.json').read_text())['fineweb_documents']
byfirst={}
for prefix in known:byfirst.setdefault(prefix[0],set()).add(prefix)
enc=tiktoken.get_encoding('gpt2');chosen=[];metadata=[];seen={value for row in previous for value in [row["document_id"],row["text_sha256"]]};rejections=dict(short=0,known_excerpt=0,duplicate_document=0)
dataset=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',split='train',streaming=True)
stream_iter=iter(dataset)
for stream_index,doc in enumerate(itertools.islice(stream_iter,20000)):
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
stream_iter.close()
del dataset
import gc
gc.collect()
import ast,copy,sysconfig
excluded={v['file'] for v in json.loads((P/'MIDPOINT_STDLIB_CONTINUATION_PLAN_V1.json').read_text())['sources']}
excluded.update(v['file'] for v in json.loads((P/'SHARED_MODE3_SECOND_ROWS_V1.json').read_text())['code_sources'])
excluded.update(v['file'] for v in json.loads((P/'PARTIAL_GRAPH_FRESH_ROWS_V1.json').read_text())['code_sources'])
excluded.update(v['file'] for v in json.loads((P/'PARTIAL_GRAPH_FRESH_ROWS_V2.json').read_text())['code_sources'])
excluded.update(v['file'] for v in json.loads((P/'COMPACT_GROUP_FRESH_ROWS_V1.json').read_text())['code_sources'])
excluded.update(v['file'] for v in json.loads((P/'DUAL_FRESH_ROWS_V1.json').read_text())['code_sources'])
excluded.update(v['file'] for v in json.loads((P/'FRONTIER_FRESH_ROWS_V1.json').read_text())['code_sources'])
code_rows=[];code_sources=[]
stdlib_root=Path(sysconfig.get_path('stdlib'))
paths=sorted(stdlib_root.glob('*.py'))+sorted(q for q in stdlib_root.glob('*/*.py') if not any(part in ['site-packages','dist-packages','test','tests','__pycache__'] for part in q.relative_to(stdlib_root).parts))
for path in paths:
 if path.name.startswith('_') or str(path) in excluded:continue
 raw=path.read_text();tree=ast.parse(raw)
 for node in sorted((n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))),key=lambda n:n.lineno):
  fn=copy.deepcopy(node)
  if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str):fn.body=fn.body[1:]
  if not fn.body:continue
  text=ast.unparse(fn);ids=enc.encode_ordinary(text)
  if len(ids)<257:continue
  code_rows.append(torch.tensor(ids[:257],dtype=torch.long));code_sources.append(dict(file=str(path),function=node.name,line=node.lineno,source_sha256=hashlib.sha256(raw.encode()).hexdigest()));break
 if len(code_rows)==16:break
assert len(code_rows)==16
code=torch.stack(code_rows)
oldcode=torch.cat([torch.load(P/'MIDPOINT_STDLIB_CONTINUATION_TOKENS_V1.pt',weights_only=True),torch.load(P/'SHARED_MODE3_SECOND_TOKENS_V1.pt',weights_only=True)['stdlib']])
assert not ({tuple(r[:64].tolist()) for r in code}&{tuple(r[:64].tolist()) for r in oldcode})
tokens={'fineweb':torch.stack(chosen),'stdlib':code}
torch.save(tokens,P/'FRONTIER_FRESH_TOKENS_V2.pt')
out=dict(fineweb_documents=metadata,code_sources=code_sources,known_cached_prefixes=len(known),rejections=rejections,source='HuggingFaceFW/fineweb sample-10BT train stream',scope='New document-indexed panel for this extraction study, one prefix perdocument and no knowncachedFineWeb excerpt anywhere in eachaccepted document. Not a pretraining-overlap or whole-project novelty guarantee. Eighth panel: all seven previous FW panels and112previous stdlib files excluded. Same unchanged64-correction graph and both resource-matched323-width baselines; independent replication after firstpanel relative failures. Code search expands to immediate stdlib packages if needed; no site-packages/test directories.')
(P/'FRONTIER_FRESH_ROWS_V2.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k!='fineweb_documents'})
