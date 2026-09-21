"""Deterministic code-shift panel selected without model outcomes."""
from pathlib import Path
import ast,copy,sysconfig,json,hashlib,torch,tiktoken
p=Path(__file__).resolve().parent;out=p/'MIDPOINT_STDLIB_CONTINUATION_PLAN_V1.json';assert not out.exists()
enc=tiktoken.get_encoding('gpt2');root=Path(sysconfig.get_path('stdlib'));rows=[];sources=[]
from token_boundary_conditions import annotate
for f in sorted(root.glob('*.py')):
 if f.name.startswith('_'):continue
 raw=f.read_text(encoding='utf-8');tree=ast.parse(raw)
 functions=sorted((n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))),key=lambda n:n.lineno)
 for node in functions:
  fn=copy.deepcopy(node)
  if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str):fn.body=fn.body[1:]
  if not fn.body:continue
  text=ast.unparse(fn);ids=enc.encode(text)
  if len(ids)<257:continue
  row=torch.tensor(ids[:257]);rows.append(row);sources.append(dict(file=str(f),function=node.name,line=node.lineno,source_sha256=hashlib.sha256(raw.encode()).hexdigest(),transformed_sha256=hashlib.sha256(text.encode()).hexdigest()));break
 if len(rows)==16:break
assert len(rows)==16
tokens=torch.stack(rows);annotations=[v for row in rows for v in annotate(row.tolist(),enc)[16:256]];torch.save(tokens,p/'MIDPOINT_STDLIB_CONTINUATION_TOKENS_V1.pt')
plan=dict(documents=list(range(16)),sources=sources,program_sha256=hashlib.sha256((p/'MIDPOINT_STABLE_GROUP_REMOVAL_PROGRAMS_V1.pt').read_bytes()).hexdigest(),token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),class_counts={k:sum(v[k] for v in annotations) for k in ['continuation','spaced_word','utf8_pending']},scope='First16alphabetical top-level stdlib files containing an eligible function; first function by source order after removing its leading docstring, ast.unparse, first257GPT2tokens. Distinct code shift from FineWeb and earlier repository panels; not evidence of absence from model pretraining.')
out.write_text(json.dumps(plan,indent=2)+'\n');print(plan['class_counts']);print([(v['file'],v['function']) for v in sources])
