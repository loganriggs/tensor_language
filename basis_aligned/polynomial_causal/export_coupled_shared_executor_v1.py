"""Extract a shared attention bank and verify two caller-supplied contexts."""
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P, CHECKPOINT
from export_coupled_interaction_v1 import function
from coupled_routing_value_mlp9_v1 import execute as source_execute
from denominator_factored_attention10_v1 import compile_program, execute as port_execute, MAPS


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    destination=P/'extracted_circuits/coupled_shared_executor_v1'
    receipt=P/'COUPLED_SHARED_EXECUTOR_V1_RESULT.json'
    assert not destination.exists() and not receipt.exists()
    destination.mkdir()
    checkpoint=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    native='transformer.h.10.attn.'
    weights={name:checkpoint[native+name+'.weight'].clone() for name in [*MAPS.values(),'c_proj']}
    # The native loaded model uses FP32 for these learned scalars.
    weights['lamb']=checkpoint[native+'lamb'].float().clone()
    short=torch.load(P/'DENOMINATOR_FACTORED_NATIVE_V1_EXAMPLE_PROGRAM.pt',weights_only=True)
    long=torch.load(P/'COUPLED_NATURAL_LENGTH_V1_EXAMPLE_PROGRAM.pt',weights_only=True)
    row=json.loads((P/'COUPLED_NATURAL_LENGTH_V1_ROWS.json').read_text())['rows'][0]
    long['x0']=F.rms_norm(F.embedding(torch.tensor([row['ids']]),checkpoint['transformer.wte.weight']).float(),(1152,),eps=torch.finfo(torch.float32).eps)
    bank=dict(weights=weights,lambdas=short['lambdas'].clone(),bias=short['source']['bias'].clone())
    contexts=[]
    for fixture in [short,long]:
        assert torch.equal(fixture['source']['bias'],bank['bias'])
        contexts.append(dict(source={k:v for k,v in fixture['source'].items() if k!='bias'},
                             x0=fixture['x0'],first_values=fixture['attention']['first_values']))
    torch.save(bank,destination/'bank.pt')
    torch.save(contexts,destination/'contexts.pt')
    code='"""Shared interaction executor. Requires torch and explicit bank/context data."""\nfrom pathlib import Path\nimport torch\nimport torch.nn.functional as F\nEPS=torch.finfo(torch.float32).eps\n'
    code+='MAPS='+repr(MAPS)+'\n\n'
    code+=function(P/'coupled_routing_value_mlp9_v1.py','execute','execute_source')+'\n\n'
    code+=function(P/'coupled_attention10_ports_v1.py','attention_write')+'\n\n'
    code+='''def load(directory=None):
    directory=Path(directory) if directory is not None else Path(__file__).parent
    return torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)


def execute(bank,context,a,b):
    source=dict(context['source'],bias=bank['bias'])
    post9=execute_source(source,a,b)
    batch,tokens,width=post9.shape
    if width!=1152 or context['x0'].shape!=post9.shape:
        raise ValueError('Source and x0 must have matching B,T,1152 shapes')
    first=context['first_values']
    if first.shape not in ((batch,tokens,1152),(batch,tokens,9,128)):
        raise ValueError('First values must have B,T,1152 or B,T,9,128 shape')
    alpha,beta=bank['lambdas'].double()
    raw=alpha*post9+beta*context['x0'].double()
    normalized=F.rms_norm(raw,(1152,),eps=EPS)
    weights=bank['weights'];ports={}
    for name,native in MAPS.items():
        value=(normalized@weights[native].double().T).reshape(batch,tokens,9,128)
        ports[name]=value if name=='v' else F.rms_norm(value,(128,),eps=EPS)
    write=attention_write(ports,weights['c_proj'],weights['lamb'],first.reshape(batch,tokens,1152))
    return dict(post_mlp9=post9,attention10=write,post_attention10=raw+write)
'''
    (destination/'execute.py').write_text(code)
    ast.parse(code)
    manifest=dict(inputs=['bank','context: source numerator/denominator/linear_basis, x0, first_values','two scalar strengths a,b; mixed strength a*b'],
                  outputs=['post_mlp9','attention10','post_attention10'],
                  tokens=[18,84],runtime_dependencies=['torch','execute.py','bank.pt','caller-supplied context'],
                  context_provenance=['Actual native capture, corrected regional row0/donor1','Natural row0/donor1 captured source/first values; x0 reconstructed using native FP32 embedding RMS on CPU'],
                  external_dependencies='Original model/prefix and MLP9 weights generate source coefficients and contexts. MLP10 and later suffix remain external. No arbitrary-text language-model interface.',
                  precision='Original FP32 attention matrices, native FP32 scalar mixture/reentry; FP64 source coefficients and runtime arithmetic. No quantization.',
                  hashes={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in ['execute.py','bank.pt','contexts.pt']})
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (destination/'README.md').write_text('''# Shared coupled interaction executor

Load the bank with `load()`. Call `execute(bank, context, a, b)` to obtain the
post-MLP9 state, attention10 write, and post-attention10 state. One bank serves
different context lengths. `contexts.pt` contains two examples; callers may
supply other context programs with the declared interface. Mixed strength is a*b.

Runtime needs PyTorch and explicit bank/context data. It reads no checkpoint.
The full original model was needed to generate the context programs; those
preparation dependencies have not been compressed away. MLP10 and the later
suffix remain external. The long example uses CPU-reconstructed x0; see manifest.
This is conditional extraction and reuse, not a complete language model or a
new semantic circuit. Runtime casts matrices to FP64; no speedup is claimed.
''')
    grid=[(a,b) for a in [-1,0,.5,1,2] for b in [-1,0,.5,1,2]]
    grid += [(-.75,.25),(.25,-.75),(.125,.875),(.875,.125),(1.25,.75),(.75,1.25),(-.5,1.5)]
    refs=[];projected=[]
    payload=lambda obj:sum(v.numel()*v.element_size() for v in obj.values())
    for context in contexts:
        source=dict(context['source'],bias=bank['bias'])
        port=compile_program(source,context['x0'],bank['lambdas'],weights,context['first_values'])
        projected.append(payload(context['source'])+payload(port)+context['x0'].numel()*context['x0'].element_size())
        local=[]
        for a,b in grid:
            post9=source_execute(source,a,b);write=port_execute(port,a,b)
            raw=bank['lambdas'][0].double()*post9+bank['lambdas'][1].double()*context['x0'].double()
            local.append(dict(post_mlp9=post9,attention10=write,post_attention10=raw+write))
        refs.append(local)
    # Deduplicate output map and mixture; source bias and reentry occur once.
    projected_bytes=sum(projected)-payload({'output':weights['c_proj'],'mixture':weights['lamb']})+payload({'bias':bank['bias'],'lambdas':bank['lambdas']})
    child='''import importlib.util,json,sys,torch
from pathlib import Path
torch.set_num_threads(2)
package=Path(sys.argv[1]);reference=torch.load(sys.argv[2],weights_only=True)
spec=importlib.util.spec_from_file_location('shared_standalone',package/'execute.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
bank=module.load();contexts=torch.load(package/'contexts.pt',weights_only=True)
maxima={k:0. for k in ['post_mlp9','attention10','post_attention10']};layout=0.
for i,context in enumerate(contexts):
 for j,(a,b) in enumerate(reference['grid']):
  actual=module.execute(bank,context,a,b)
  for k,expected in reference['refs'][i][j].items():
   error=float((actual[k]-expected).norm()/expected.norm().clamp_min(1e-30))
   maxima[k]=max(maxima[k],error)
 context=dict(context,first_values=context['first_values'].reshape(1,-1,1152))
 flat=module.execute(bank,context,.125,.875)
 for k in flat:layout=max(layout,float((flat[k]-reference['refs'][i][27][k]).norm()/reference['refs'][i][27][k].norm().clamp_min(1e-30)))
outside=[]
repo=str(package.parents[3])
for name,value in list(sys.modules.items()):
 filename=getattr(value,'__file__',None)
 if filename and str(filename).startswith(repo) and not str(filename).startswith(str(package)):outside.append(name)
print(json.dumps(dict(maxima=maxima,flat_layout_error=layout,project_imports_outside_package=outside,cases=64)))
'''
    with tempfile.TemporaryDirectory(prefix='shared-coupled-isolation-') as temp:
        reference=Path(temp)/'reference.pt';torch.save(dict(grid=grid,refs=refs),reference)
        completed=subprocess.run([sys.executable,'-I','-c',child,str(destination),str(reference)],capture_output=True,text=True,check=True,timeout=60)
        isolated=json.loads(completed.stdout)
    package_bytes=sum(f.stat().st_size for f in destination.iterdir() if f.is_file())
    result=dict(utc=datetime.now(timezone.utc).isoformat(),
                pred_a=max([*isolated['maxima'].values(),isolated['flat_layout_error']])<=1e-10,
                pred_b=not isolated['project_imports_outside_package'],pred_c=package_bytes<=projected_bytes,
                isolated=isolated,package_and_context_file_bytes=package_bytes,
                shared_projected_payload_bytes=projected_bytes,package_to_projected=package_bytes/projected_bytes,
                bank_tensor_bytes=payload(weights)+payload({'lambdas':bank['lambdas'],'bias':bank['bias']}),
                scope=manifest['external_dependencies'],context_provenance=manifest['context_provenance'])
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
