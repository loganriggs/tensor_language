"""Export and isolate the fixed-context coupled MLP9/attention10 executable."""
import ast,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from coupled_routing_value_mlp9_v1 import execute as source_execute
from coupled_attention10_ports_v2 import execute as attention_execute


def function(path,name,new_name=None):
    text=path.read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    code=ast.get_source_segment(text,node)
    return code.replace('def '+name+'(', 'def '+(new_name or name)+'(',1)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    destination=P/'extracted_circuits/coupled_interaction_v1'
    out=P/'COUPLED_INTERACTION_EXTRACTION_V1_RESULT.json'
    assert not destination.exists() and not out.exists();destination.mkdir()
    program=torch.load(P/'COUPLED_ATTENTION10_NATIVE_V2_EXAMPLE_PROGRAM.pt',weights_only=True)
    state=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    row=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows'][0]
    ids=torch.tensor([row['ids']])
    program['x0']=F.rms_norm(F.embedding(ids,state['transformer.wte.weight']).float(),(1152,),eps=torch.finfo(torch.float32).eps)
    program['lambdas']=state['transformer.h.10.lambdas'].clone()
    torch.save(program,destination/'program.pt')
    port_source=P/'coupled_attention10_ports_v1.py';text=port_source.read_text();tree=ast.parse(text)
    constants=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['EPS','EXPS9','MAPS'] for t in node.targets):
            constants.append(ast.get_source_segment(text,node))
    code='"""Fixed-context executable; only torch and declared program data required."""\nfrom pathlib import Path\nimport torch\n\n'
    code+='\n'.join(constants)+'\n\n'
    code+=function(P/'coupled_routing_value_mlp9_v1.py','execute','execute_source')+'\n\n'
    for name in ['monomials','normalized_ports','attention_write']:code+=function(port_source,name)+'\n\n'
    code+=function(P/'coupled_attention10_ports_v2.py','execute','execute_attention')+'\n\n'
    code+='''def load(directory=None):
    directory=Path(directory) if directory is not None else Path(__file__).parent
    return torch.load(directory/'program.pt',map_location='cpu',weights_only=True)


def execute(program,a,b):
    post9=execute_source(program['source'],a,b)
    attention10=execute_attention(program['attention'],a,b)
    alpha,beta=program['lambdas'].double()
    postattention10=alpha*post9+beta*program['x0'].double()+attention10
    return dict(post_mlp9=post9,attention10=attention10,post_attention10=postattention10)
'''
    (destination/'execute.py').write_text(code)
    manifest=dict(context='SRO_ARTICLE_CORRECTION_V1 row0, paired donor row1',tokens=ids.shape[1],
                  inputs=['two scalar strengths a,b; mixed strength is a*b'],
                  outputs=['post_mlp9','attention10','post_attention10'],
                  dependencies=['torch','program.pt','execute.py'],
                  external_preparation='Original model weights and native prefix/source-port capture generate this context program; not needed by the exported executor.',
                  remaining_native='MLP10 and all later blocks remain external. This package is fixed-context, not an independent language model.',
                  x0_provenance='Recomputed from checkpoint embedding using native FP32 RMS formula on CPU; isolation compares this explicit frame. PostMLP9/attention10 programs are copied from the native-validated example.',
                  hashes={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in ['program.pt','execute.py']})
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (destination/'README.md').write_text('''# Fixed-context coupled interaction

`load()` reads the included program; `execute(program,a,b)` returns post-MLP9,
attention10 write, and post-attention10 residual states. Mixed strength equals
`a*b`. Runtime requires only PyTorch and these files; it reads no checkpoint.

This program specializes one recorded recipient/donor context. Original model
weights and native prefix computations were needed to prepare it. It does not
accept arbitrary text. MLP10 and the remaining suffix are external. The explicit
x0 frame is recomputed on CPU from weights; see manifest provenance. No new
semantic circuit, global parameter saving or runtime speedup is claimed.
''')
    grid=[(a,b) for a in [-1,0,.5,1,2] for b in [-1,0,.5,1,2]]
    refs=[]
    for a,b in grid:
        y=source_execute(program['source'],a,b);att=attention_execute(program['attention'],a,b)
        alpha,beta=program['lambdas'].double()
        refs.append(torch.stack([y,att,alpha*y+beta*program['x0'].double()+att]))
    with tempfile.TemporaryDirectory(prefix='coupled-isolation-') as temp:
        fixture=Path(temp)/'reference.pt';torch.save(torch.stack(refs),fixture)
        child='''import importlib.util,json,sys,torch
from pathlib import Path
torch.set_num_threads(2)
package=Path(sys.argv[1]);reference=torch.load(sys.argv[2],weights_only=True)
spec=importlib.util.spec_from_file_location('isolated_coupled',package/'execute.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
program=module.load();actual=[]
for a in [-1,0,.5,1,2]:
 for b in [-1,0,.5,1,2]:
  result=module.execute(program,a,b)
  actual.append(torch.stack([result[k] for k in ['post_mlp9','attention10','post_attention10']]))
actual=torch.stack(actual);error=float((actual-reference).norm()/reference.norm())
repo=str(package.parents[3]);outside=[]
for name,value in list(sys.modules.items()):
 filename=getattr(value,'__file__',None)
 if filename and str(filename).startswith(repo) and not str(filename).startswith(str(package)):outside.append(name)
print(json.dumps(dict(error=error,project_imports_outside_package=outside,cases=25,checkpoint_references='absent from standalone runtime source')))
'''
        completed=subprocess.run([sys.executable,'-I','-c',child,str(destination),str(fixture)],capture_output=True,text=True,check=True,timeout=60)
        isolated=json.loads(completed.stdout)
    package_bytes=sum(f.stat().st_size for f in destination.iterdir())
    baseline=json.loads((P/'COUPLED_ATTENTION10_NATIVE_V2_RESULT.json').read_text())['prices'][0]['independent_bytes']
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=isolated['error']<=1e-10,
                pred_b=not isolated['project_imports_outside_package'],pred_c=package_bytes<=.75*baseline,
                isolated=isolated,package_bytes=package_bytes,baseline_payload_bytes=baseline,
                package=str(destination.relative_to(P)),scope=manifest['remaining_native'],x0_provenance=manifest['x0_provenance'])
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
