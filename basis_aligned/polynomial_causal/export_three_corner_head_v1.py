"""Export the validated conditional head core; preserve all external dependencies."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import torch
from three_corner_head_interface_v1 import execute
from additive_head_raw_ports_v1 import EPS, project
from head17_source_interface_v1 import CHECKPOINT

P=Path(__file__).resolve().parent
PACKAGE=P/'extracted_circuits/three_corner_head17_interaction_v1'


def main():
    torch.set_num_threads(2)
    PACKAGE.mkdir(exist_ok=True)
    sources={
        'execute.py':'three_corner_head_interface_v1.py',
        'ports.py':'additive_head_raw_ports_v1.py',
        'mixed.py':'joint_attention_mixed_ports_v1.py',
        'compact.py':'joint_attention_three_group_v1.py',
    }
    for target,source in sources.items():
        text=(P/source).read_text()
        for before,after in [('from additive_head_raw_ports_v1','from .ports'),
                             ('from joint_attention_mixed_ports_v1','from .mixed'),
                             ('from joint_attention_three_group_v1','from .compact'),
                             ('from folded_normalized_router_v1','from .rotary')]:
            text=text.replace(before,after)
        (PACKAGE/target).write_text(text)
    rotary=(P/'folded_normalized_router_v1.py').read_text()
    start=rotary.index('def rotary(');end=rotary.index('\n\ndef fold(',start)
    (PACKAGE/'rotary.py').write_text('import torch\n\n'+rotary[start:end]+'\n')
    (PACKAGE/'__init__.py').write_text('from .execute import execute\n\ndef residual_write(*args, output_matrix, **kwargs):\n    return execute(*args, **kwargs) @ output_matrix.T\n')
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    output=sd['transformer.h.17.attn.c_proj.weight'][:,256:384].float().contiguous()
    mixture=sd['transformer.h.17.attn.lamb'].float().clone()
    torch.save(dict(output_matrix=output,mixture=mixture),PACKAGE/'program.pt')
    matrices=[sd[f'transformer.h.17.attn.{k}.weight'].reshape(9,128,1152)[2].double()
              for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    torch.manual_seed(320);n=torch.randn(8,11,1152,dtype=torch.float64)
    c=n+0.1*torch.randn_like(n);r=n+0.3*torch.randn_like(n)
    projections=[project(x,matrices) for x in (n,c,r)]
    norms=[x.square().mean(-1)+EPS for x in (n,c,r)]
    cross=((c-n)*(r-n)).mean(-1);first=torch.randn(8,11,128,dtype=torch.float64)
    sys.path.insert(0,str(PACKAGE.parent));package=importlib.import_module(PACKAGE.name)
    stored=torch.load(PACKAGE/'program.pt',weights_only=True)
    errors={}
    for compact in (False,True):
        args=(projections,norms,cross,first,float(stored['mixture']))
        reference=execute(*args,compact=compact)@output.double().T
        actual=package.residual_write(*args,compact=compact,output_matrix=stored['output_matrix'].double())
        errors[str(compact)]=float((actual-reference).norm()/reference.norm())
    assert max(errors.values())==0
    price=dict(stored_tensor_scalars=output.numel()+mixture.numel(),stored_tensor_bytes=4*(output.numel()+mixture.numel()),
               source_projection_scalars_per_token=3*5*128,norm_and_cross_scalars_per_token=4,first_value_scalars_per_token=128,
               output_residual_width=1152,conditions=['native','child_removed','remainder_removed'],
               excluded_but_required=['Projection generators and all associated native weights','Norm and cross-inner-product generators','Shared first-value generator','Three condition executions and upstream child/remainder definition','Final additive background, final RMS, unembedding, soft-cap and scored-token selection'],
               scope='Local conditional interface price. Native model generation remains required; no wholemodel compression/adoption or autonomous circuit claim.')
    (PACKAGE/'PRICE.json').write_text(json.dumps(price,indent=2)+'\n')
    result=dict(export_relative_errors=errors,files={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in PACKAGE.iterdir() if x.is_file() and x.name!='README.md'},scope='Independent package import and weight reload, identical outputs on actual-weight synthetic projected inputs. Native evidence validates original core; copied package not independently GPU-deployed.')
    (P/'THREE_CORNER_HEAD_EXPORT_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(errors=errors,price=price),indent=2))


if __name__=='__main__':main()
